"""Recover the original three OOF folds after explicit user authorization.

This is the only model-card utility that fits estimators. It fits three temporary
fold pipelines, never the final production model, and never overwrites that model.
The user authorized these three fits in the model-card conversation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn

from src.modeling import (
    build_hist_gradient_boosting_pipeline,
    evaluate_model_cv,
    load_training_data,
    make_stratified_splits,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = ROOT / "reports/model_card/tables"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "oof_predictions.csv").exists():
        raise FileExistsError("OOF predictions already exist. Reuse them; this utility must not trigger additional CV fits by accident.")
    model_path = ROOT / "models/final_hist_gradient_boosting_pipeline.joblib"
    config_path = ROOT / "models/final_hist_gradient_boosting_config.json"
    before = {str(p.relative_to(ROOT)): sha256(p) for p in (model_path, config_path)}
    X, y = load_training_data(ROOT / "data/raw/application_train.csv")
    splits = make_stratified_splits(X, y, n_splits=3, random_state=42)
    print("Running exactly three authorized CV fits; final model is untouched.", flush=True)
    result = evaluate_model_cv(
        "Histogram gradient boosting (unweighted)",
        build_hist_gradient_boosting_pipeline(class_weight=None, random_state=42),
        X, y, splits=splits,
    )
    fold_id = np.zeros(len(y), dtype=np.int8)
    for number, (_, validation) in enumerate(splits, start=1):
        fold_id[validation] = number
    predictions = pd.DataFrame({
        "SK_ID_CURR": X["SK_ID_CURR"], "TARGET": y,
        "probability": result.oof_predictions, "fold": fold_id,
    })
    assert predictions["SK_ID_CURR"].is_unique
    assert predictions["probability"].between(0, 1).all()
    predictions_path = output / "oof_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    result.fold_metrics.to_csv(output / "oof_fold_metrics.csv", index=False)
    result.preprocessing_audit.to_csv(output / "oof_preprocessing_audit.csv", index=False)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in (model_path, config_path)}
    assert before == after, "A protected final-model artifact changed."
    metadata = {
        "prediction_kind": "out_of_fold",
        "model": result.name, "n_splits": 3, "random_state": 42,
        "rows": len(predictions), "sklearn_version": sklearn.__version__,
        "training_source_sha256": sha256(ROOT / "data/raw/application_train.csv"),
        "predictions_sha256": sha256(predictions_path),
        "fold_local_preprocessing_verified": bool(result.preprocessing_audit["verified"].all()),
        "mean_cv_auc": float(result.fold_metrics["roc_auc"].mean()),
        "oof_auc": float(result.fold_metrics.attrs["oof_roc_auc"]),
        "fit_count": 3, "authorization": "Explicit user approval in model-card conversation",
        "final_artifact_hashes_before": before, "final_artifact_hashes_after": after,
        "limitations": "Original development folds; not nested validation or a fresh holdout.",
    }
    (output / "oof_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ["rows", "mean_cv_auc", "oof_auc", "fit_count"]}), flush=True)


if __name__ == "__main__":
    main()
