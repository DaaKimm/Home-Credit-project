"""Reproducible application-level preparation for Home Credit modeling.

The transformations in this module implement the reversible remediations in
``notebooks/01_eda.ipynb`` ("Remediations applied", cells 86--89).  They work
only on copies of the supplied application frames; raw CSV files are never
changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


TARGET_COLUMN = "TARGET"
EXTREME_VALUE_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OWN_CAR_AGE",
)
LOG_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
)
POSITIVE_DAY_COLUMNS = (
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
)
# EDA question 9 (notebook cells 38--39) defines these applicant-age bands.
AGE_BAND_EDGES = (20, 30, 40, 50, 60, 100)


@dataclass(frozen=True)
class PreparationParameters:
    """Parameters learned from a training application frame only."""

    extreme_value_upper_bounds: dict[str, float]
    missing_indicator_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable parameters for reproducible inference."""

        return {
            "extreme_value_upper_bounds": self.extreme_value_upper_bounds,
            "missing_indicator_columns": list(self.missing_indicator_columns),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PreparationParameters":
        """Restore parameters previously saved with :func:`save_parameters`."""

        bounds = values["extreme_value_upper_bounds"]
        return cls(
            {column: float(bound) for column, bound in bounds.items()},
            tuple(values["missing_indicator_columns"]),
        )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], frame_name: str) -> None:
    """Raise a clear error when a frame lacks columns needed for preparation."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def assert_train_test_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Require exactly matching train/test features, with ``TARGET`` train-only.

    This implements the EDA train/test schema check (notebook cells 13--14 and
    26--27) before and after preparation.  The target is deliberately retained
    in the returned train data and is not allowed in the test data.
    """

    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"Training data must contain {TARGET_COLUMN!r}.")
    if TARGET_COLUMN in test.columns:
        raise ValueError(f"Test data must not contain {TARGET_COLUMN!r}.")

    train_features = [column for column in train.columns if column != TARGET_COLUMN]
    test_features = list(test.columns)
    train_only = sorted(set(train_features).difference(test_features))
    test_only = sorted(set(test_features).difference(train_features))
    if train_only or test_only:
        raise ValueError(
            "Train/test feature columns differ. "
            f"Train-only: {train_only}; test-only: {test_only}"
        )
    if train_features != test_features:
        raise ValueError("Train/test feature columns have different orders.")


def validate_train_test(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """Validate application keys and the train/test schema for modeling.

    The EDA key check (notebook cells 9--10 and 18--19) establishes one row per
    application ID.  The EDA split check (cells 13--14 and 26--27) requires the
    same feature columns, excluding training-only ``TARGET``.
    """

    for frame_name, frame in (("Training data", train), ("Test data", test)):
        _require_columns(frame, ("SK_ID_CURR",), frame_name)
        if frame["SK_ID_CURR"].isna().any():
            raise ValueError(f"{frame_name} contains missing SK_ID_CURR values.")
        duplicate_count = int(frame["SK_ID_CURR"].duplicated().sum())
        if duplicate_count:
            raise ValueError(
                f"{frame_name} must have one row per SK_ID_CURR; "
                f"found {duplicate_count} duplicate IDs."
            )

    assert_train_test_feature_columns(train, test)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_unique_sk_id_curr": int(train["SK_ID_CURR"].nunique()),
        "test_unique_sk_id_curr": int(test["SK_ID_CURR"].nunique()),
        "shared_feature_count": len(train.columns) - 1,
    }


def fit_preparation_parameters(train: pd.DataFrame) -> PreparationParameters:
    """Fit the EDA extreme-value thresholds using the training split only.

    The notebook specifies the 99.9th percentile as a flagging threshold,
    rather than a cap or deletion rule (cell 87 and cell 89).  No target values
    are used to learn this parameter.
    """

    _require_columns(train, EXTREME_VALUE_COLUMNS, "Training data")
    # EDA remediation decision (notebook cells 87 and 89): use the training
    # 99.9th percentile to flag unverified extremes, never to cap or delete them.
    bounds = train.loc[:, EXTREME_VALUE_COLUMNS].quantile(0.999).to_dict()

    # EDA decision 3 (notebook cells 5--6 and 20--21): missingness was assessed
    # before recoding because it can carry information.  Learn which original
    # columns need indicators from training data only; do not infer this from test.
    missing_indicator_columns = tuple(
        column
        for column in train.columns
        if column != TARGET_COLUMN and train[column].isna().any()
    )
    return PreparationParameters(
        {column: float(bound) for column, bound in bounds.items()},
        missing_indicator_columns,
    )


def transform_application(
    frame: pd.DataFrame, parameters: PreparationParameters
) -> pd.DataFrame:
    """Apply EDA-approved application transformations without mutating ``frame``.

    ``parameters`` must be fitted with :func:`fit_preparation_parameters` on a
    training frame.  ``TARGET`` is copied through unchanged when it is present.
    """

    required_columns = tuple(
        set(EXTREME_VALUE_COLUMNS)
        | set(LOG_COLUMNS)
        | set(POSITIVE_DAY_COLUMNS)
        | {"DAYS_EMPLOYED", "CODE_GENDER"}
    )
    _require_columns(frame, required_columns, "Application data")
    parameter_columns = tuple(parameters.extreme_value_upper_bounds)
    _require_columns(
        pd.DataFrame(columns=parameter_columns), EXTREME_VALUE_COLUMNS, "Preparation parameters"
    )
    _require_columns(frame, parameters.missing_indicator_columns, "Application data")

    cleaned = frame.copy()

    # EDA decision 3 (notebook cells 5--6 and 20--21): preserve source
    # missingness as model-visible indicators instead of silently imputing it.
    for column in parameters.missing_indicator_columns:
        cleaned[f"{column}_MISSING_FLAG"] = cleaned[column].isna().astype("int8")

    # EDA cell 87: 365243 contradicts a days-before-application value.  Retain
    # the signal in a flag, then represent the invalid value as missing.
    employed_sentinel = cleaned["DAYS_EMPLOYED"].eq(365243)
    cleaned["DAYS_EMPLOYED_SENTINEL_FLAG"] = employed_sentinel.astype("int8")
    cleaned.loc[employed_sentinel, "DAYS_EMPLOYED"] = np.nan

    # EDA cell 87: XNA is a data-quality signal, not a verified gender value.
    gender_unknown = cleaned["CODE_GENDER"].eq("XNA")
    cleaned["CODE_GENDER_UNKNOWN_FLAG"] = gender_unknown.astype("int8")
    cleaned.loc[gender_unknown, "CODE_GENDER"] = np.nan

    # EDA cell 87: positive values conflict with documented before-application
    # direction.  Preserve their occurrence in flags and set values to missing.
    for column in POSITIVE_DAY_COLUMNS:
        invalid = cleaned[column].gt(0)
        cleaned[f"{column}_POSITIVE_FLAG"] = invalid.astype("int8")
        cleaned.loc[invalid, column] = np.nan

    # EDA cells 87 and 89: thresholds are training-fitted flags only; source
    # values are neither capped nor deleted.
    for column, bound in parameters.extreme_value_upper_bounds.items():
        cleaned[f"{column}_EXTREME_FLAG"] = cleaned[column].gt(bound).astype("int8")

    # EDA cell 87: retain unmodified monetary values and add stable log views.
    for column in LOG_COLUMNS:
        cleaned[f"{column}_LOG1P"] = np.log1p(cleaned[column])

    # EDA question 9 (notebook cells 38--39): age is defined from the negative
    # relative-day field, so convert it to an interpretable years feature.
    cleaned["AGE_YEARS"] = -cleaned["DAYS_BIRTH"] / 365.25

    # EDA question 9 (notebook cells 38--39): reuse the pre-specified age-band
    # edges from the EDA.  Integer codes keep the binned feature model-ready
    # while retaining missing values for ages outside the documented range.
    cleaned["AGE_BAND"] = pd.cut(
        cleaned["AGE_YEARS"], AGE_BAND_EDGES, labels=False, include_lowest=True
    ).astype("Int8")

    # EDA question 9 (cells 38--41) and remediation cell 87: derive employment
    # duration only after the 365243 sentinel has been made missing.
    cleaned["EMPLOYMENT_DURATION_YEARS"] = -cleaned["DAYS_EMPLOYED"] / 365.25

    # EDA questions 6 and 12 (cells 36--37 and 46--47): credit-to-income and
    # annuity-to-income are affordability proxies.  Zero income is undefined,
    # so represent it as missing rather than creating an infinite ratio.
    income = cleaned["AMT_INCOME_TOTAL"].replace(0, np.nan)
    cleaned["CREDIT_TO_INCOME"] = cleaned["AMT_CREDIT"] / income
    cleaned["ANNUITY_TO_INCOME"] = cleaned["AMT_ANNUITY"] / income

    return cleaned


def prepare_train_test(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, PreparationParameters]:
    """Fit on train, transform both splits, and verify their final schemas."""

    validate_train_test(train, test)
    parameters = fit_preparation_parameters(train)
    prepared_train = transform_application(train, parameters)
    prepared_test = transform_application(test, parameters)
    validate_train_test(prepared_train, prepared_test)
    if not prepared_train[TARGET_COLUMN].equals(train[TARGET_COLUMN]):
        raise AssertionError("TARGET changed during preparation.")
    return prepared_train, prepared_test, parameters


def save_parameters(parameters: PreparationParameters, path: str | Path) -> None:
    """Save training-fitted parameters while preserving feature creation order."""

    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(parameters.to_dict(), output, indent=2)


def load_parameters(path: str | Path) -> PreparationParameters:
    """Load thresholds written by :func:`save_parameters`."""

    with Path(path).open(encoding="utf-8") as source:
        return PreparationParameters.from_dict(json.load(source))


def main() -> None:
    """Create prepared train/test CSVs and the training-fitted parameter file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Raw training CSV path")
    parser.add_argument("--test", type=Path, required=True, help="Raw test CSV path")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for derived outputs"
    )
    arguments = parser.parse_args()

    train = pd.read_csv(arguments.train)
    test = pd.read_csv(arguments.test)
    prepared_train, prepared_test, parameters = prepare_train_test(train, test)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_train.to_csv(arguments.output_dir / "application_train_prepared.csv", index=False)
    prepared_test.to_csv(arguments.output_dir / "application_test_prepared.csv", index=False)
    save_parameters(parameters, arguments.output_dir / "application_preparation_parameters.json")

    feature_count = prepared_train.shape[1] - 1
    print(
        "Preparation completed: "
        f"{len(prepared_train):,} train rows, {len(prepared_test):,} test rows, "
        f"{feature_count} shared features."
    )


if __name__ == "__main__":
    main()
