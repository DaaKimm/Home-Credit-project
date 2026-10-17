"""Reusable, leakage-safe modeling utilities for Home Credit experiments.

The functions in this module keep all learned transformations inside a
scikit-learn ``Pipeline``.  During cross-validation, each cloned pipeline is
fitted only on that fold's training rows before it sees the validation rows.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_preparation import (
    PreparationParameters,
    fit_preparation_parameters,
    transform_application,
)


TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 3
CVSplit = tuple[np.ndarray, np.ndarray]


class ApplicationPreparationTransformer(TransformerMixin, BaseEstimator):
    """Apply the existing application preparation with fold-fitted parameters."""

    def __init__(self, drop_columns: tuple[str, ...] = (ID_COLUMN,)) -> None:
        self.drop_columns = drop_columns

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> ApplicationPreparationTransformer:
        """Learn preparation parameters only from rows passed by the pipeline."""

        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Application preparation requires a pandas DataFrame.")
        missing_drop_columns = sorted(set(self.drop_columns).difference(X.columns))
        if missing_drop_columns:
            raise ValueError(f"Missing columns configured for removal: {missing_drop_columns}")

        self.parameters_: PreparationParameters = fit_preparation_parameters(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.fit_row_count_ = len(X)
        self.fit_index_ = X.index.copy()

        empty_prepared = transform_application(X.iloc[:0], self.parameters_)
        self.feature_names_out_ = np.asarray(
            [column for column in empty_prepared.columns if column not in self.drop_columns],
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a fold while retaining a DataFrame for dtype selectors."""

        check_is_fitted(self, "parameters_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Application preparation requires a pandas DataFrame.")
        missing = sorted(set(self.feature_names_in_).difference(X.columns))
        if missing:
            raise ValueError(f"Application data is missing fitted columns: {missing}")

        prepared = transform_application(
            X.loc[:, self.feature_names_in_.tolist()], self.parameters_
        )
        return prepared.drop(columns=list(self.drop_columns))

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Return prepared feature names after identifier removal."""

        del input_features
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class FoldFitAuditTransformer(TransformerMixin, BaseEstimator):
    """Record which rows reached downstream preprocessing during ``fit``."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> FoldFitAuditTransformer:
        """Capture the fitting index without changing any features."""

        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The preprocessing audit requires a pandas DataFrame.")
        self.fit_row_count_ = len(X)
        self.fit_index_ = X.index.copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pass data through unchanged."""

        check_is_fitted(self, "fit_index_")
        return X

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Return unchanged feature names."""

        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            return self.feature_names_in_.copy()
        return np.asarray(input_features, dtype=object)


@dataclass(frozen=True)
class CrossValidationResult:
    """Metrics, predictions, and leakage-audit details for one experiment."""

    name: str
    fold_metrics: pd.DataFrame
    oof_predictions: pd.Series
    preprocessing_audit: pd.DataFrame

    def summary(self) -> dict[str, object]:
        """Return one comparison-table row."""

        metrics = self.fold_metrics
        return {
            "model": self.name,
            "mean_accuracy": metrics["accuracy"].mean(),
            "std_accuracy": metrics["accuracy"].std(ddof=1),
            "mean_roc_auc": metrics["roc_auc"].mean(),
            "std_roc_auc": metrics["roc_auc"].std(ddof=1),
            "oof_roc_auc": metrics.attrs["oof_roc_auc"],
            "mean_fit_seconds": metrics["fit_seconds"].mean(),
            "total_cv_seconds": metrics["fold_seconds"].sum(),
            "fold_local_preprocessing_verified": bool(
                self.preprocessing_audit["verified"].all()
            ),
        }


@dataclass(frozen=True)
class HistogramTuningResult:
    """Outputs from the reproducible sample-based randomized search."""

    search: RandomizedSearchCV
    sample_indices: pd.Index
    sample_positive_rate: float
    sample_splits: tuple[CVSplit, ...]
    search_seconds: float
    ranking: pd.DataFrame


@dataclass(frozen=True)
class FinalFitResult:
    """Saved final pipeline plus its two-stage iteration-selection details."""

    pipeline: Pipeline
    calibration_fit_seconds: float
    final_fit_seconds: float
    calibration_iteration_cap: int
    stopped_iteration_count: int
    best_validation_iteration: int
    best_validation_score: float
    stopped_early: bool


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load application training data and separate the binary target."""

    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Training data must contain {TARGET_COLUMN!r}.")
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"Training data must contain {ID_COLUMN!r}.")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be non-missing and unique.")

    raw_target = frame.pop(TARGET_COLUMN)
    if raw_target.isna().any() or set(raw_target.unique()) != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must contain exactly the labels 0 and 1.")
    y = raw_target.astype("int8")
    return frame, y


def load_test_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the unlabeled application test table."""

    frame = pd.read_csv(path)
    if TARGET_COLUMN in frame.columns:
        raise ValueError(f"Test data must not contain {TARGET_COLUMN!r}.")
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"Test data must contain {ID_COLUMN!r}.")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"Test {ID_COLUMN} must be non-missing and unique.")
    return frame


def build_feature_preprocessor() -> ColumnTransformer:
    """Create fold-fitted numeric and categorical preprocessing."""

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_dense_tree_preprocessor() -> ColumnTransformer:
    """Create dense fold-fitted preprocessing for histogram boosting.

    Histogram gradient boosting requires dense numeric input. This preserves
    the same imputed numeric and one-hot categorical representation used by the
    other candidates while changing only the output storage format.
    """

    numeric = SimpleImputer(strategy="median", add_indicator=True)
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_logistic_pipeline(
    *,
    class_weight: str | dict[int, float] | None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build a logistic model whose learned preparation stays inside CV."""

    return Pipeline(
        steps=[
            ("application_preparation", ApplicationPreparationTransformer()),
            ("preprocessing_fit_audit", FoldFitAuditTransformer()),
            ("feature_preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=500,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_baseline_models(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, BaseEstimator]:
    """Return the required majority and weighted/unweighted logistic baselines."""

    return {
        "Majority-class dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic regression (unweighted)": build_logistic_pipeline(
            class_weight=None, random_state=random_state
        ),
        "Logistic regression (balanced)": build_logistic_pipeline(
            class_weight="balanced", random_state=random_state
        ),
    }


def build_random_forest_pipeline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build a bounded class-balanced random-forest candidate."""

    return Pipeline(
        steps=[
            ("application_preparation", ApplicationPreparationTransformer()),
            ("preprocessing_fit_audit", FoldFitAuditTransformer()),
            ("feature_preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    max_samples=0.8,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_pipeline(
    *,
    class_weight: str | dict[int, float] | None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build the bounded scikit-learn fallback for LightGBM."""

    return Pipeline(
        steps=[
            ("application_preparation", ApplicationPreparationTransformer()),
            ("preprocessing_fit_audit", FoldFitAuditTransformer()),
            ("feature_preprocessor", build_dense_tree_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    early_stopping=True,
                    scoring="roc_auc",
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_lightgbm_pipeline(
    *,
    class_weight: str | dict[int, float] | None,
    random_state: int,
) -> Pipeline:
    """Build a bounded LightGBM pipeline when the optional package exists."""

    from lightgbm import LGBMClassifier

    return Pipeline(
        steps=[
            ("application_preparation", ApplicationPreparationTransformer()),
            ("preprocessing_fit_audit", FoldFitAuditTransformer()),
            ("feature_preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=random_state,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_candidate_models(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[str, BaseEstimator], str]:
    """Return bounded tree candidates and describe the boosting backend."""

    candidates: dict[str, BaseEstimator] = {
        "Random forest (balanced subsample)": build_random_forest_pipeline(
            random_state=random_state
        )
    }

    if find_spec("lightgbm") is not None:
        candidates.update(
            {
                "LightGBM (unweighted)": _build_lightgbm_pipeline(
                    class_weight=None, random_state=random_state
                ),
                "LightGBM (balanced)": _build_lightgbm_pipeline(
                    class_weight="balanced", random_state=random_state
                ),
            }
        )
        backend = "LightGBM is installed, so the requested LightGBM candidates are used."
    else:
        candidates.update(
            {
                "Histogram gradient boosting (unweighted)": (
                    build_hist_gradient_boosting_pipeline(
                        class_weight=None, random_state=random_state
                    )
                ),
                "Histogram gradient boosting (balanced)": (
                    build_hist_gradient_boosting_pipeline(
                        class_weight="balanced", random_state=random_state
                    )
                ),
            }
        )
        backend = (
            "LightGBM is not installed. scikit-learn's HistGradientBoostingClassifier "
            "is used as the available gradient-boosting alternative because it uses "
            "histogram-binned trees, supports class weights and early stopping, and "
            "scales to this data without an added dependency."
        )
    return candidates, backend


def make_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[CVSplit, ...]:
    """Materialize the shared CV splits used by every experiment."""

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return tuple(splitter.split(X, y))


def make_stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_size: int = 5_000,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a reproducible sample that preserves the binary target ratio."""

    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Features and target must have identical row indices.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must contain exactly the labels 0 and 1.")
    if not 2 <= sample_size <= len(X):
        raise ValueError("sample_size must be between 2 and the full row count.")

    sampler = StratifiedShuffleSplit(
        n_splits=1, train_size=sample_size, random_state=random_state
    )
    sample_positions, _ = next(sampler.split(X, y))
    sample_positions.sort()
    return X.iloc[sample_positions].copy(), y.iloc[sample_positions].copy()


def histogram_boosting_search_space() -> list[dict[str, list[object]]]:
    """Return bounded leaf-limited and depth-limited search branches.

    Searching leaf and depth limits simultaneously wastes trials because a
    shallow depth already caps the possible leaf count. Separate branches let
    each sampled configuration use exactly one structural constraint.
    """

    prefix = "classifier__"
    common: dict[str, list[object]] = {
        f"{prefix}learning_rate": [0.025, 0.04, 0.05, 0.075, 0.1],
        f"{prefix}max_iter": [100, 150, 200, 250],
        f"{prefix}min_samples_leaf": [20, 40, 60, 100],
        f"{prefix}l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
    }
    return [
        {
            **common,
            f"{prefix}max_depth": [None],
            f"{prefix}max_leaf_nodes": [15, 31, 63],
        },
        {
            **common,
            f"{prefix}max_depth": [3, 4, 6, 8],
            f"{prefix}max_leaf_nodes": [None],
        },
    ]


def summarize_random_search(search: RandomizedSearchCV) -> pd.DataFrame:
    """Return a compact ranking with scores clearly labeled as sample CV."""

    results = pd.DataFrame(search.cv_results_)
    split_columns = sorted(
        column
        for column in results
        if column.startswith("split") and column.endswith("_test_score")
    )
    parameter_columns = [
        column for column in results if column.startswith("param_classifier__")
    ]
    ranking = results.loc[
        :,
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "std_fit_time",
            *split_columns,
            *parameter_columns,
        ],
    ].copy()
    ranking = ranking.rename(
        columns={
            "rank_test_score": "sample_rank",
            "mean_test_score": "sample_mean_cv_auc",
            "std_test_score": "sample_cv_auc_std_ddof0",
            "mean_fit_time": "sample_mean_fit_seconds",
            "std_fit_time": "sample_fit_seconds_std",
            **{
                column: f"sample_fold_{index + 1}_auc"
                for index, column in enumerate(split_columns)
            },
            **{
                column: column.removeprefix("param_classifier__")
                for column in parameter_columns
            },
        }
    )
    return ranking.sort_values("sample_rank").reset_index(drop=True)


def tune_hist_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_size: int = 5_000,
    n_iter: int = 20,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> HistogramTuningResult:
    """Tune unweighted histogram boosting on a stratified sample.

    The estimator is the complete preparation/preprocessing/model pipeline, so
    ``RandomizedSearchCV`` refits every learned transformation inside each CV
    training fold. Histogram boosting's early-stopping split is consequently
    nested inside that outer training fold.
    """

    X_sample, y_sample = make_stratified_sample(
        X, y, sample_size=sample_size, random_state=random_state
    )
    sample_splits = make_stratified_splits(
        X_sample,
        y_sample,
        n_splits=n_splits,
        random_state=random_state,
    )
    pipeline = build_hist_gradient_boosting_pipeline(
        class_weight=None, random_state=random_state
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=histogram_boosting_search_space(),
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
        cv=sample_splits,
        verbose=0,
        pre_dispatch=1,
        random_state=random_state,
        error_score="raise",
        return_train_score=False,
    )
    started = time.perf_counter()
    search.fit(X_sample, y_sample)
    search_seconds = time.perf_counter() - started

    application = search.best_estimator_.named_steps["application_preparation"]
    downstream = search.best_estimator_.named_steps["preprocessing_fit_audit"]
    if application.fit_row_count_ != sample_size or downstream.fit_row_count_ != sample_size:
        raise AssertionError("The refitted search pipeline did not see the full tuning sample.")

    return HistogramTuningResult(
        search=search,
        sample_indices=X_sample.index.copy(),
        sample_positive_rate=float(y_sample.mean()),
        sample_splits=sample_splits,
        search_seconds=search_seconds,
        ranking=summarize_random_search(search),
    )


def build_tuned_hist_gradient_boosting_pipeline(
    best_params: Mapping[str, object],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Apply randomized-search parameters to a fresh unweighted pipeline."""

    invalid = sorted(
        parameter for parameter in best_params if not parameter.startswith("classifier__")
    )
    if invalid:
        raise ValueError(f"Tuned parameters must target the classifier step: {invalid}")
    pipeline = build_hist_gradient_boosting_pipeline(
        class_weight=None, random_state=random_state
    )
    return pipeline.set_params(**best_params)


def compare_paired_auc_results(
    default: CrossValidationResult,
    tuned: CrossValidationResult,
    *,
    minimum_mean_gain: float = 0.001,
    minimum_fold_wins: int = 2,
    minimum_oof_gain: float = 0.001,
    maximum_fold_regression: float = 0.0005,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare matched folds and apply a predeclared practical selection rule."""

    default_folds = default.fold_metrics.reset_index(drop=True)
    tuned_folds = tuned.fold_metrics.reset_index(drop=True)
    identity_columns = ["fold", "train_rows", "validation_rows"]
    if not default_folds[identity_columns].equals(tuned_folds[identity_columns]):
        raise ValueError("Default and tuned results do not use identical folds.")

    paired = default_folds.loc[:, identity_columns + ["roc_auc"]].rename(
        columns={"roc_auc": "default_roc_auc"}
    )
    paired["tuned_roc_auc"] = tuned_folds["roc_auc"]
    paired["tuned_minus_default"] = (
        paired["tuned_roc_auc"] - paired["default_roc_auc"]
    )

    mean_gain = float(paired["tuned_minus_default"].mean())
    fold_wins = int((paired["tuned_minus_default"] > 0).sum())
    worst_fold_change = float(paired["tuned_minus_default"].min())
    default_oof = float(default.fold_metrics.attrs["oof_roc_auc"])
    tuned_oof = float(tuned.fold_metrics.attrs["oof_roc_auc"])
    oof_gain = tuned_oof - default_oof
    convincing = (
        mean_gain >= minimum_mean_gain
        and fold_wins >= minimum_fold_wins
        and oof_gain >= minimum_oof_gain
        and worst_fold_change >= -maximum_fold_regression
    )
    decision = {
        "mean_auc_gain": mean_gain,
        "fold_wins": fold_wins,
        "oof_auc_gain": oof_gain,
        "worst_fold_auc_change": worst_fold_change,
        "minimum_mean_gain": minimum_mean_gain,
        "minimum_fold_wins": minimum_fold_wins,
        "minimum_oof_gain": minimum_oof_gain,
        "maximum_fold_regression": maximum_fold_regression,
        "convincing_improvement": convincing,
        "selected_configuration": "tuned" if convincing else "default",
    }
    return paired, decision


def fit_final_hist_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FinalFitResult:
    """Choose an iteration count, then refit the retained default on all rows.

    The calibration fit uses the default internal early-stopping holdout and
    records the iteration at which scikit-learn stops. A fresh pipeline then
    disables early stopping, fixes ``max_iter`` to that count, and fits the
    classifier on every labeled row. Competition-test data are never used.
    """

    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Features and target must have identical row indices.")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must contain exactly the labels 0 and 1.")

    calibration = build_hist_gradient_boosting_pipeline(
        class_weight=None, random_state=random_state
    )
    calibration_cap = int(calibration.named_steps["classifier"].max_iter)
    calibration_started = time.perf_counter()
    calibration.fit(X, y)
    calibration_seconds = time.perf_counter() - calibration_started
    calibration_classifier = calibration.named_steps["classifier"]
    stopped_iterations = int(calibration_classifier.n_iter_)
    validation_scores = np.asarray(
        calibration_classifier.validation_score_, dtype="float64"
    )
    if validation_scores.size < 2 or not np.isfinite(validation_scores[1:]).any():
        raise AssertionError("Early-stopping calibration did not produce validation scores.")
    best_validation_iteration = int(np.nanargmax(validation_scores[1:]) + 1)
    best_validation_score = float(validation_scores[best_validation_iteration])
    stopped_early = stopped_iterations < calibration_cap

    del calibration, calibration_classifier, validation_scores
    gc.collect()

    final_pipeline = build_hist_gradient_boosting_pipeline(
        class_weight=None, random_state=random_state
    ).set_params(
        classifier__early_stopping=False,
        classifier__max_iter=stopped_iterations,
        classifier__validation_fraction=None,
    )
    final_started = time.perf_counter()
    final_pipeline.fit(X, y)
    final_seconds = time.perf_counter() - final_started

    classifier = final_pipeline.named_steps["classifier"]
    application = final_pipeline.named_steps["application_preparation"]
    downstream = final_pipeline.named_steps["preprocessing_fit_audit"]
    if int(classifier.n_iter_) != stopped_iterations:
        raise AssertionError("Final classifier did not use the selected iteration count.")
    if classifier.class_weight is not None:
        raise AssertionError("The retained final classifier must be unweighted.")
    if ID_COLUMN in set(application.get_feature_names_out()):
        raise AssertionError(f"{ID_COLUMN} reached the predictor matrix.")
    if TARGET_COLUMN in set(application.feature_names_in_):
        raise AssertionError(f"{TARGET_COLUMN} reached the predictor pipeline.")
    if application.fit_row_count_ != len(X) or downstream.fit_row_count_ != len(X):
        raise AssertionError("Final preprocessing was not fitted on every labeled row.")

    return FinalFitResult(
        pipeline=final_pipeline,
        calibration_fit_seconds=calibration_seconds,
        final_fit_seconds=final_seconds,
        calibration_iteration_cap=calibration_cap,
        stopped_iteration_count=stopped_iterations,
        best_validation_iteration=best_validation_iteration,
        best_validation_score=best_validation_score,
        stopped_early=stopped_early,
    )


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, object]:
    """Validate Kaggle schema, ID order, uniqueness, and probability bounds."""

    expected_columns = [ID_COLUMN, TARGET_COLUMN]
    if list(sample_submission.columns) != expected_columns:
        raise ValueError(
            f"Sample submission columns must be exactly {expected_columns}; "
            f"found {list(sample_submission.columns)}."
        )
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Submission columns or their order differ from the sample.")
    if len(submission) != len(sample_submission) or len(submission) != len(test):
        raise ValueError("Submission, sample, and test row counts must match.")
    submission_ids = submission[ID_COLUMN].to_numpy()
    if not np.array_equal(submission_ids, sample_submission[ID_COLUMN].to_numpy()):
        raise ValueError("Submission IDs or row order differ from the sample submission.")
    if not np.array_equal(submission_ids, test[ID_COLUMN].to_numpy()):
        raise ValueError("Submission IDs or row order differ from the test data.")
    if submission[ID_COLUMN].isna().any() or submission[ID_COLUMN].duplicated().any():
        raise ValueError("Submission IDs must be non-missing and unique.")

    probabilities = submission[TARGET_COLUMN].to_numpy(dtype="float64")
    if not np.isfinite(probabilities).all():
        raise ValueError("Submission predictions must all be finite.")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("Submission predictions must lie between 0 and 1.")

    return {
        "columns_exact": True,
        "column_order": list(submission.columns),
        "row_count": len(submission),
        "expected_row_count": len(sample_submission),
        "unique_id_count": int(submission[ID_COLUMN].nunique()),
        "ids_unique": True,
        "ids_match_sample_in_order": True,
        "ids_match_test_in_order": True,
        "all_predictions_finite": True,
        "all_predictions_in_unit_interval": True,
        "prediction_min": float(probabilities.min()),
        "prediction_max": float(probabilities.max()),
        "prediction_mean": float(probabilities.mean()),
        "prediction_std": float(probabilities.std()),
    }


def create_kaggle_submission(
    fitted_pipeline: Pipeline,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score test rows and write a sample-compatible submission CSV."""

    application = fitted_pipeline.named_steps["application_preparation"]
    expected_features = application.feature_names_in_.tolist()
    if list(test.columns) != expected_features:
        raise ValueError("Test feature names or order differ from fitted training features.")
    if TARGET_COLUMN in test.columns:
        raise ValueError(f"Test predictors must not contain {TARGET_COLUMN!r}.")

    probabilities = _positive_class_probability(fitted_pipeline, test)
    submission = sample_submission.copy()
    submission[TARGET_COLUMN] = probabilities
    validation = validate_submission(submission, sample_submission, test)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    written = pd.read_csv(output)
    validate_submission(written, sample_submission, test)
    if not np.allclose(
        written[TARGET_COLUMN].to_numpy(),
        probabilities,
        rtol=0.0,
        atol=np.finfo("float64").eps,
    ):
        raise AssertionError("Written probabilities differ from in-memory predictions.")
    validation.update(
        {
            "written_file_revalidated": True,
            "written_probabilities_match_memory": True,
            "submission_path": str(output.resolve()),
            "submission_bytes": output.stat().st_size,
        }
    )
    return submission, validation


def _json_safe(value: object) -> object:
    """Recursively convert common scientific-Python values for JSON output."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for a saved artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_pipeline_and_configuration(
    fitted_pipeline: Pipeline,
    model_path: str | Path,
    configuration: Mapping[str, object],
    configuration_path: str | Path,
) -> dict[str, object]:
    """Persist the fitted pipeline and a JSON configuration beside it."""

    model_output = Path(model_path)
    config_output = Path(configuration_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    dump(fitted_pipeline, model_output, compress=3)
    artifact_details = {
        "model_path": str(model_output.resolve()),
        "model_bytes": model_output.stat().st_size,
        "model_sha256": _sha256(model_output),
        "configuration_path": str(config_output.resolve()),
    }
    complete_configuration = {
        **dict(configuration),
        "artifacts": artifact_details,
    }
    with config_output.open("w", encoding="utf-8") as destination:
        json.dump(
            _json_safe(complete_configuration),
            destination,
            indent=2,
            allow_nan=False,
        )
        destination.write("\n")
    artifact_details["configuration_bytes"] = config_output.stat().st_size
    artifact_details["configuration_sha256"] = _sha256(config_output)
    return artifact_details


def load_saved_pipeline(path: str | Path) -> Pipeline:
    """Load a pipeline written by :func:`save_pipeline_and_configuration`."""

    fitted = load(Path(path))
    if not isinstance(fitted, Pipeline):
        raise TypeError("Saved artifact is not a scikit-learn Pipeline.")
    return fitted


def _positive_class_probability(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Return the probability assigned to class 1."""

    probabilities = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError("The fitted estimator does not expose binary class 1 probabilities.")
    return probabilities[:, positive_positions[0]]


def _audit_pipeline_fit(
    estimator: BaseEstimator,
    training_index: pd.Index,
    validation_index: pd.Index,
) -> dict[str, object]:
    """Verify preparation and downstream preprocessing saw training rows only."""

    if not isinstance(estimator, Pipeline):
        return {
            "application_fit_rows": np.nan,
            "preprocessor_fit_rows": np.nan,
            "validation_rows_seen_during_fit": np.nan,
            "verified": True,
            "note": "No feature preprocessing is used by this estimator.",
        }

    required_steps = {"application_preparation", "preprocessing_fit_audit"}
    if not required_steps.issubset(estimator.named_steps):
        raise AssertionError(
            "A preprocessing pipeline must expose the application preparation "
            "and preprocessing audit steps."
        )

    application = estimator.named_steps["application_preparation"]
    downstream = estimator.named_steps["preprocessing_fit_audit"]
    application_exact = application.fit_index_.equals(training_index)
    downstream_exact = downstream.fit_index_.equals(training_index)
    validation_seen = len(downstream.fit_index_.intersection(validation_index))
    verified = application_exact and downstream_exact and validation_seen == 0
    if not verified:
        raise AssertionError("Learned preprocessing was not isolated to the training fold.")

    return {
        "application_fit_rows": application.fit_row_count_,
        "preprocessor_fit_rows": downstream.fit_row_count_,
        "validation_rows_seen_during_fit": validation_seen,
        "verified": verified,
        "note": "All learned preprocessing was fitted on this fold's training rows.",
    }


def evaluate_model_cv(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    splits: Sequence[CVSplit] | None = None,
) -> CrossValidationResult:
    """Evaluate one classifier with reusable stratified folds and OOF predictions."""

    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Features and target must have identical row indices.")
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds.")

    if splits is None:
        splits = make_stratified_splits(
            X, y, n_splits=n_splits, random_state=random_state
        )
    oof = pd.Series(np.nan, index=y.index, dtype="float64", name=name)
    fold_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for fold, (training_positions, validation_positions) in enumerate(
        splits, start=1
    ):
        X_train = X.iloc[training_positions]
        y_train = y.iloc[training_positions]
        X_validation = X.iloc[validation_positions]
        y_validation = y.iloc[validation_positions]

        fitted = clone(estimator)
        fold_started = time.perf_counter()
        fitted.fit(X_train, y_train)
        fitted_at = time.perf_counter()

        probabilities = _positive_class_probability(fitted, X_validation)
        predictions = (probabilities >= 0.5).astype("int8")
        scored_at = time.perf_counter()
        oof.iloc[validation_positions] = probabilities

        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(training_positions),
                "validation_rows": len(validation_positions),
                "train_positive_rate": y_train.mean(),
                "validation_positive_rate": y_validation.mean(),
                "predicted_positive_rate": np.mean(predictions == 1),
                "accuracy": accuracy_score(y_validation, predictions),
                "roc_auc": roc_auc_score(y_validation, probabilities),
                "fit_seconds": fitted_at - fold_started,
                "score_seconds": scored_at - fitted_at,
                "fold_seconds": scored_at - fold_started,
            }
        )
        audit_rows.append(
            {
                "fold": fold,
                **_audit_pipeline_fit(fitted, X_train.index, X_validation.index),
            }
        )

    if oof.isna().any():
        raise AssertionError("Every row must receive exactly one out-of-fold prediction.")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.attrs["oof_roc_auc"] = roc_auc_score(y, oof)
    return CrossValidationResult(
        name=name,
        fold_metrics=fold_metrics,
        oof_predictions=oof,
        preprocessing_audit=pd.DataFrame(audit_rows),
    )


def evaluate_models_cv(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    splits: Sequence[CVSplit] | None = None,
) -> tuple[pd.DataFrame, dict[str, CrossValidationResult]]:
    """Evaluate models on identical folds and return a ranked comparison."""

    results = {
        name: evaluate_model_cv(
            name,
            estimator,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            splits=splits,
        )
        for name, estimator in models.items()
    }
    comparison = (
        pd.DataFrame([result.summary() for result in results.values()])
        .sort_values("mean_roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    return comparison, results
