"""Model-card analyses using persisted OOF predictions and the saved final model.

No estimator is fitted in this module. Run notebooks/03_model_card_analysis.ipynb
after the explicitly authorized, separate OOF recovery utility has completed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "home_credit_mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "home_credit_numba"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
import shap
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

from src.modeling import load_saved_pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/model_card"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEED = 42
SAMPLE_SIZE = 1000
ONEMAIN_URL = "https://www.sec.gov/Archives/edgar/data/1584207/000158420726000004/exhibit991earningsreleaseo.htm"
RECOVERY_URL = "https://www.crc.business-school.ed.ac.uk/sites/crc/files/2024-01/Business-Cycle-and-Realized-Losses-in-the-Consumer-Credit-Industry.pdf"
ITALY_URL = "https://www.bancaditalia.it/pubblicazioni/note-stabilita/2025-0048/Note-di-stabilita-finanziaria-e-vigilanza-n-48-ENG.pdf?language_id=1"
SHAP_URL = "https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html"
CALIBRATION_URL = "https://scikit-learn.org/stable/modules/calibration.html"
BENCHMARK_ID = "2025_margin_financing_recovery"
TEAL = "#147D92"
ORANGE = "#C66B23"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    def text(value):
        if pd.isna(value):
            return "[UNDEFINED]"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(text(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def save_table(frame: pd.DataFrame, stem: str, caption: str, *, markdown=True) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{stem}.csv", index=False)
    if markdown:
        (TABLES / f"{stem}.md").write_text(caption + "\n\n" + markdown_table(frame) + "\n")


def save_figure(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_context() -> dict:
    for directory in (OUT, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    model_path = ROOT / "models/final_hist_gradient_boosting_pipeline.joblib"
    config_path = ROOT / "models/final_hist_gradient_boosting_config.json"
    config = json.loads(config_path.read_text())
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (model_path, config_path)}
    assert hashes[str(model_path.relative_to(ROOT))] == config["artifacts"]["model_sha256"]
    pipeline = load_saved_pipeline(model_path)
    assert sklearn.__version__ == config["software"]["scikit_learn"]
    test = pd.read_csv(ROOT / "data/raw/application_test.csv")
    submission = pd.read_csv(ROOT / "submissions/home_credit_hist_gradient_boosting.csv")
    assert submission.columns.tolist() == ["SK_ID_CURR", "TARGET"]
    assert submission["SK_ID_CURR"].is_unique and test["SK_ID_CURR"].is_unique
    assert submission["SK_ID_CURR"].equals(test["SK_ID_CURR"])
    assert submission["TARGET"].between(0, 1).all()
    train_meta = pd.read_csv(ROOT / "data/raw/application_train.csv",
                            usecols=["SK_ID_CURR", "TARGET", "CODE_GENDER", "NAME_EDUCATION_TYPE"])
    assert set(train_meta["SK_ID_CURR"]).isdisjoint(test["SK_ID_CURR"])
    with threadpool_limits(limits=4):
        recomputed = pipeline.predict_proba(test)[:, list(pipeline.classes_).index(1)]
    np.testing.assert_allclose(recomputed, submission["TARGET"], rtol=1e-12, atol=1e-14)
    oof_path = TABLES / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError("[NEEDS INPUT] OOF predictions are absent. Do not substitute training-set predictions or automatically fit models.")
    oof_meta = json.loads((TABLES / "oof_provenance.json").read_text())
    assert oof_meta["predictions_sha256"] == sha256(oof_path)
    assert oof_meta["prediction_kind"] == "out_of_fold"
    assert oof_meta["fold_local_preprocessing_verified"]
    oof = pd.read_csv(oof_path)
    assert oof["SK_ID_CURR"].is_unique and len(oof) == len(train_meta)
    assert oof["probability"].between(0, 1).all() and set(oof["fold"]) == {1, 2, 3}
    oof = oof.merge(train_meta, on="SK_ID_CURR", how="left", validate="one_to_one", suffixes=("", "_raw"))
    assert oof["TARGET"].equals(oof["TARGET_raw"])
    original_auc = config["selection_evidence"]["baseline_and_candidate_results"][0]["oof_roc_auc"]
    assert np.isclose(roc_auc_score(oof["TARGET"], oof["probability"]), original_auc, atol=1e-10, rtol=0)
    return {"pipeline": pipeline, "config": config, "protected_hashes": hashes,
            "test": test, "test_probability": recomputed, "oof": oof,
            "oof_provenance": oof_meta, "train_meta": train_meta}


def economic_scenarios(ctx: dict) -> pd.DataFrame:
    """Compute externally sourced proxies, without calling them loan-level profit."""
    margins = pd.DataFrame([
        {"year": 2024, "interest_income_usd_m": 4965, "interest_expense_usd_m": 1181,
         "operating_expense_usd_m": 1554, "average_net_receivables_usd_m": 22440},
        {"year": 2025, "interest_income_usd_m": 5432, "interest_expense_usd_m": 1270,
         "operating_expense_usd_m": 1687, "average_net_receivables_usd_m": 24028},
    ])
    margins["annual_precredit_pretax_margin_proxy"] = (
        margins["interest_income_usd_m"] - margins["interest_expense_usd_m"] - margins["operating_expense_usd_m"]
    ) / margins["average_net_receivables_usd_m"]
    margins["source_url"] = ONEMAIN_URL
    margins["source_location"] = "C&I segment, fiscal-year columns; C&I additional metrics, average net receivables"
    save_table(margins, "sourced_margin_inputs", "OneMain C&I portfolio proxy, not conditional profit on a repaid Home Credit loan. USD millions; margin is annual per unit average receivables, pre-credit-loss and pretax.")
    recoveries = pd.DataFrame([
        {"recovery_id": "financing", "recovery_rate": .47, "observations": 353891,
         "source_url": RECOVERY_URL, "source_location": "July 25, 2023 working paper, Table 1, printed page 8, Financing row",
         "scope": "Italian collection-agency consumer financing debts, 2007-2019; recovery mandate at most 365 days"},
        {"recovery_id": "banking", "recovery_rate": .48, "observations": 240936,
         "source_url": RECOVERY_URL, "source_location": "Table 1, printed page 8, Banking row",
         "scope": "Italian collection-agency bank overdrafts/personal credits; same source and recovery window"},
        {"recovery_id": "italy_unsecured_2023", "recovery_rate": .28, "observations": "[NEEDS INPUT]",
         "source_url": ITALY_URL, "source_location": "December 2025 note, printed page 2, unsecured positions",
         "scope": "Italian banks' closed unsecured bad-loan positions; firms and households, includes public guarantees"},
        {"recovery_id": "italy_unsecured_2024", "recovery_rate": .37, "observations": "[NEEDS INPUT]",
         "source_url": ITALY_URL, "source_location": "December 2025 note, printed page 2, unsecured positions",
         "scope": "Same mixed bad-loan population; increase materially influenced by public guarantees"},
    ])
    recoveries["gross_loss_share_proxy"] = 1 - recoveries["recovery_rate"]
    save_table(recoveries, "sourced_recovery_inputs", "Recovery benchmarks have different populations, denominators, and horizons. 1 - recovery is a gross unrecovered-share proxy, not a measured net economic LGD for Home Credit.")
    rows = []
    for margin in margins.to_dict("records"):
        for recovery in recoveries.to_dict("records"):
            gain = margin["annual_precredit_pretax_margin_proxy"]
            loss = recovery["gross_loss_share_proxy"]
            rows.append({"scenario": f'{margin["year"]}_margin_{recovery["recovery_id"]}_recovery',
                         "gain_proxy": gain, "loss_proxy": loss,
                         "recovery_rate": recovery["recovery_rate"],
                         "threshold": gain / (gain + loss),
                         "status": "Sourced benchmark sensitivity; not a business recommendation"})
    # Only illustrative: interpreted explicitly as loss:profit = 7:1.
    rows.append({"scenario": "professor_loss_to_profit_7_to_1_illustration",
                 "gain_proxy": 1., "loss_proxy": 7., "recovery_rate": np.nan,
                 "threshold": 1 / 8, "status": "Illustration only; arbitrary units; not empirical"})
    scenarios = pd.DataFrame(rows)
    ctx["scenarios"] = scenarios
    selected = scenarios.set_index("scenario").loc[BENCHMARK_ID]
    ctx["selected_threshold"] = float(selected["threshold"])
    ctx["gain"] = float(selected["gain_proxy"])
    ctx["loss"] = float(selected["loss_proxy"])
    ctx["thresholds"] = np.unique(np.r_[
        [0, .01, .025, .05, .075, .10, .125, .15, .175, .20, .25, .30, .40, .50, .60, .75, 1.0],
        scenarios["threshold"].to_numpy(),
    ])
    save_table(scenarios, "economic_scenarios", "Analytic break-even threshold G/(G+L). The selected analysis benchmark is 2025_margin_financing_recovery. No final lending threshold is established.")
    needs = pd.DataFrame([
        ("Profit per repaid loan", "Loan-level cash flows, term, fees, funding, servicing, origination, taxes and capital costs; external annual portfolio margin is only a proxy."),
        ("Net economic LGD and recovery", "Home Credit recoveries net of collection costs and discounted to default; recovery horizon, collateral, EAD and currency."),
        ("Label-to-loss mapping", "TARGET is payment difficulty with undisclosed X/Y, not charge-off; need economic outcomes conditional on this exact label."),
        ("Time and exposure alignment", "Align probability horizon, amortizing exposure, margin horizon, and LGD denominator; no one-year term or EAD is inferred from AMT_CREDIT."),
        ("Reject/review economics and constraints", "Opportunity costs, review costs, approval/review capacity, risk appetite and acceptable subgroup impacts."),
        ("Deployment validation", "Independent temporal validation and probability calibration assessment on the deployment population; no calibration model is fitted here."),
    ], columns=["needed_input", "why_needed"])
    needs.insert(1, "status", "[NEEDS INPUT]")
    save_table(needs, "business_inputs_needed", "These gaps prevent a final Home Credit business recommendation, even though cited benchmarks support a diagnostic calculation.")
    return scenarios


def threshold_metrics(y: np.ndarray, p: np.ndarray, thresholds) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        pred = p >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision = precision_score(y, pred, zero_division=np.nan)
        recall = recall_score(y, pred, zero_division=np.nan)
        rows.append({"threshold": threshold, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                     "precision": precision, "recall": recall, "predicted_default_rate": pred.mean(),
                     "simulated_selection_rate": (~pred).mean(), "approved_count": int((~pred).sum()),
                     "rejected_count": int(pred.sum()), "observed_difficulty_rate_selected": fn / (tn + fn) if tn + fn else np.nan,
                     "precision_status": "defined" if tp + fp else "undefined: no predicted defaults"})
    frame = pd.DataFrame(rows)
    assert (frame[["TN", "FP", "FN", "TP"]].sum(axis=1) == len(y)).all()
    assert (frame["TP"] + frame["FN"] == y.sum()).all()
    assert np.allclose(frame["simulated_selection_rate"] + frame["predicted_default_rate"], 1)
    return frame


def analyze_thresholds(ctx: dict) -> pd.DataFrame:
    oof = ctx["oof"]
    y, p = oof["TARGET"].to_numpy(), oof["probability"].to_numpy()
    metrics = threshold_metrics(y, p, ctx["thresholds"])
    metrics["population"] = "training_OOF_original_development_folds"
    metrics["selected_benchmark"] = metrics["threshold"].eq(ctx["selected_threshold"])
    save_table(metrics, "oof_threshold_metrics", "Pooled OOF metrics; positive class is TARGET=1 (payment difficulty). Predict positive at p >= threshold; simulate selection at p < threshold. Undefined values are not zeros.")
    ctx["metrics"] = metrics
    # Keep full long-format confusion matrices for easy reuse in model cards.
    matrices = []
    for row in metrics.to_dict("records"):
        for truth, prediction, key in [(0, 0, "TN"), (0, 1, "FP"), (1, 0, "FN"), (1, 1, "TP")]:
            matrices.append({"threshold": row["threshold"], "actual_TARGET": truth,
                             "predicted_TARGET": prediction, "cell": key, "count": row[key]})
    save_table(pd.DataFrame(matrices), "oof_confusion_matrices", "Rows: actual TARGET. Columns: predicted TARGET. Counts are exact pooled OOF counts.", markdown=False)
    test_rows = []
    for t in ctx["thresholds"]:
        approved = ctx["test_probability"] < t
        test_rows.append({"threshold": t, "applications": len(approved), "approved_count": int(approved.sum()),
                          "predicted_default_count": int((~approved).sum()),
                          "simulated_selection_rate": approved.mean(), "predicted_default_rate": (~approved).mean()})
    save_table(pd.DataFrame(test_rows), "test_threshold_outcomes", "Saved final model on unlabeled competition test. Simulated selection rates, not observed approvals; no precision or recall is available here.")
    payoffs = []
    for scenario in ctx["scenarios"].to_dict("records"):
        g, loss = scenario["gain_proxy"], scenario["loss_proxy"]
        for row in metrics.to_dict("records"):
            approve = p < row["threshold"]
            test_approve = ctx["test_probability"] < row["threshold"]
            payoffs.append({"scenario": scenario["scenario"], "threshold": row["threshold"],
                "oof_label_based_payoff_per_applicant": (g * row["TN"] - loss * row["FN"]) / len(y),
                "oof_model_implied_payoff_per_applicant": np.where(approve, (1-p)*g-p*loss, 0).mean(),
                "test_model_implied_payoff_per_applicant": np.where(test_approve, (1-ctx["test_probability"])*g-ctx["test_probability"]*loss, 0).mean(),
                "units": "illustrative units" if scenario["scenario"].startswith("professor") else "proxy exposure units",
                "status": "Benchmark calculation; not realized money or validated financial value"})
    payoff_frame = pd.DataFrame(payoffs)
    save_table(payoff_frame, "profit_sensitivity", "Proxy payoff G*TN - L*FN versus score-implied expectation; reject payoff is zero. All financial values depend on unvalidated mappings and equal exposure. Professor units must not be compared with sourced-margin units.", markdown=False)
    ctx["payoffs"] = payoff_frame

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(metrics.threshold, metrics.precision, label="Precision", marker="o", ms=3)
    axes[0].plot(metrics.threshold, metrics.recall, label="Recall", marker="o", ms=3)
    axes[1].plot(metrics.threshold, metrics.simulated_selection_rate, label="Simulated selection", color=TEAL)
    axes[1].plot(metrics.threshold, metrics.predicted_default_rate, label="Predicted difficulty", color=ORANGE)
    for ax in axes:
        ax.axvline(ctx["selected_threshold"], color=TEAL, ls="--", label=f'Benchmark {ctx["selected_threshold"]:.4f}')
        ax.axvline(.50, color="#777777", ls=":", label="Current cutoff 0.50")
        ax.set(xlabel="Probability threshold", xlim=(0, .6), ylim=(0, 1.03))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle("OOF decision trade-offs | 307,511 applications")
    save_figure(fig, "threshold_tradeoffs")

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for scenario in ["2024_margin_financing_recovery", BENCHMARK_ID,
                     "2025_margin_italy_unsecured_2023_recovery", "2025_margin_italy_unsecured_2024_recovery"]:
        subset = payoff_frame[payoff_frame.scenario.eq(scenario)]
        ax.plot(subset.threshold, subset.oof_label_based_payoff_per_applicant,
                marker="o", ms=3, label=scenario.replace("_", " "))
    ax.axvline(ctx["selected_threshold"], color="black", ls="--", label="Selected analysis benchmark")
    ax.axvline(.50, color="#777777", ls=":", label="Current cutoff 0.50")
    ax.set(xlim=(0, .6), xlabel="Probability threshold", ylabel="Proxy payoff per applicant (equal exposure)",
           title="Sourced benchmark sensitivity on OOF labels\nNot realized lending profit; economic mapping remains unvalidated")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.15)
    save_figure(fig, "profit_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, t, title in zip(axes, [ctx["selected_threshold"], .5], ["Sourced analysis benchmark", "Current cutoff"]):
        matrix = confusion_matrix(y, p >= t, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center",
                        color="white" if matrix[i,j] > matrix.max()/2 else "black", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted TARGET", ylabel="Actual TARGET",
               title=f"{title}\nt = {t:.6f}")
    save_figure(fig, "confusion_matrix_comparison")

    binned = pd.DataFrame({"p": p, "y": y}).assign(bin=pd.qcut(p, 10, duplicates="drop"))
    calibration = binned.groupby("bin", observed=True).agg(n=("y", "size"), mean_probability=("p", "mean"), observed_rate=("y", "mean")).reset_index()
    calibration["bin"] = calibration["bin"].astype(str)
    save_table(calibration, "oof_calibration_bins", "Descriptive OOF reliability bins; no calibrator is fitted. Model-development validation only.")
    fig, ax = plt.subplots(figsize=(5.8, 5), constrained_layout=True)
    ax.plot([0, .35], [0, .35], ls="--", color="#777777", label="Perfect calibration")
    ax.plot(calibration.mean_probability, calibration.observed_rate, "o-", color=TEAL, label="OOF quantile bins")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed payment-difficulty rate",
           title=f"OOF calibration diagnostic\nBrier score = {brier_score_loss(y,p):.6f}")
    ax.legend()
    save_figure(fig, "oof_calibration")
    ctx["brier_score"] = float(brier_score_loss(y, p))
    return metrics


def analyze_groups(ctx: dict) -> pd.DataFrame:
    columns = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]
    populations = [("competition_test_saved_final_model", ctx["test"], ctx["test_probability"]),
                   ("training_OOF_fold_models", ctx["oof"], ctx["oof"]["probability"].to_numpy())]
    rows = []
    for column in columns:
        levels = sorted(set(ctx["train_meta"][column].dropna().astype(str)) | set(ctx["test"][column].dropna().astype(str)))
        levels += ["<MISSING>"]
        for population, frame, probabilities in populations:
            raw_groups = frame[column].astype("string").fillna("<MISSING>")
            for t in ctx["thresholds"]:
                total = 0
                for group in levels:
                    mask = raw_groups.eq(group).to_numpy(dtype=bool)
                    n = int(mask.sum())
                    approved = int((probabilities[mask] < t).sum())
                    total += n
                    rows.append({"population": population, "attribute": column, "group": group,
                                 "threshold": t, "n": n, "simulated_selected": approved,
                                 "predicted_default_count": n-approved,
                                 "simulated_selection_rate": approved/n if n else np.nan,
                                 "rate_status": "observed sample" if n else "NO OBSERVATIONS; rate undefined",
                                 "interpretation": "Simulated selection, not observed approval"})
                assert total == len(frame)
    groups = pd.DataFrame(rows)
    save_table(groups, "selection_rates_all_thresholds", "Separate raw gender and education groups; missing/unknown levels retained. No observed approval or fairness-compliance claim.", markdown=False)
    selected = groups[groups.threshold.eq(ctx["selected_threshold"])].copy()
    for column in columns:
        save_table(selected[selected.attribute.eq(column)], f"selection_rates_{column}",
                   f"Simulated selection rates at sourced analysis benchmark t={ctx['selected_threshold']:.9f}. Both populations are labeled; no Home Credit operational threshold is established. [UNDEFINED] means zero observations.")
    ctx["groups"] = groups
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for ax, column in zip(axes, columns):
        subset = groups[(groups.population.eq("competition_test_saved_final_model")) & groups.attribute.eq(column)]
        base = subset[subset.threshold.eq(ctx["selected_threshold"])]
        comparison = subset[subset.threshold.eq(.5)].set_index("group")
        pos = np.arange(len(base))
        ax.barh(pos-.17, base.simulated_selection_rate, height=.32, label=f'Benchmark {ctx["selected_threshold"]:.4f}', color=TEAL)
        ax.barh(pos+.17, comparison.loc[base.group, "simulated_selection_rate"], height=.32, label="Current cutoff 0.50", color="#AEB7C0")
        labels = [f'{r.group}\n(n={r.n:,})' for r in base.itertuples()]
        for i, row in enumerate(base.itertuples()):
            if not row.n:
                ax.text(.02, i, "No observations", va="center", fontsize=9)
        ax.set(yticks=pos, yticklabels=labels, xlim=(0, 1.03), xlabel="Simulated selection rate", title=column)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        # Include zero-count groups in the axis limits: NaN bars alone do not
        # extend matplotlib's data limits, otherwise the final label hits the axis.
        ax.set_ylim(len(base) - .4, -.6)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Saved final model | 48,744 competition-test applications\nSimulated selection rates, not observed approvals")
    save_figure(fig, "simulated_selection_by_group")
    return selected


def explain_saved_model(ctx: dict) -> pd.DataFrame:
    pipeline = ctx["pipeline"]
    sample = ctx["test"].sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    positions = sample.index.to_numpy()
    # Transform only: preparation, imputation and encoding remain fitted and frozen.
    with threadpool_limits(limits=4):
        transformed = np.asarray(pipeline[:-1].transform(sample), dtype=float)
        classifier = pipeline.named_steps["classifier"]
        names = pipeline.named_steps["feature_preprocessor"].get_feature_names_out()
        explainer = shap.TreeExplainer(classifier, model_output="raw",
                                       feature_perturbation="tree_path_dependent", feature_names=names)
        values = np.asarray(explainer.shap_values(transformed, check_additivity=True))
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        probabilities = classifier.predict_proba(transformed)[:, list(classifier.classes_).index(1)]
    assert values.shape == transformed.shape == (SAMPLE_SIZE, len(names))
    assert np.isfinite(values).all()
    error = float(np.max(np.abs(expit(baseline + values.sum(axis=1)) - probabilities)))
    np.testing.assert_allclose(expit(baseline + values.sum(axis=1)), probabilities, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(probabilities, ctx["test_probability"][positions], rtol=1e-12, atol=1e-14)
    sample_ids = pd.DataFrame({"sample_order": np.arange(SAMPLE_SIZE), "test_row_index": positions,
                              "SK_ID_CURR": sample["SK_ID_CURR"].to_numpy(), "probability": probabilities})
    save_table(sample_ids, "shap_sample_ids", "Exactly 1,000 competition-test rows sampled without replacement using pandas random_state=42. Order matches saved SHAP arrays.", markdown=False)
    np.savez_compressed(OUT / "shap_values.npz", values=values, transformed_features=transformed,
                        feature_names=np.asarray(names, dtype=str), expected_value=baseline,
                        SK_ID_CURR=sample["SK_ID_CURR"].to_numpy())
    feature_rows = []
    for j, name in enumerate(names):
        x, s = transformed[:, j], values[:, j]
        distinct = np.unique(x)
        rho = float(spearmanr(x, s).statistic) if len(distinct) > 1 and np.ptp(s) > 0 else np.nan
        low_cut, high_cut = np.quantile(x, [.25, .75])
        binary = len(distinct) == 2 and set(distinct) == {0, 1}
        if binary:
            low, high = x == 0, x == 1
            contrast = "present (1) versus absent (0)"
            low_cut, high_cut = 0., 1.
        elif low_cut < high_cut:
            low, high = x <= low_cut, x >= high_cut
            contrast = "upper versus lower feature quartile"
        else:
            low = high = np.zeros(len(x), dtype=bool)
            contrast = "quartiles tied; no disjoint tail comparison"
        low_mean = float(s[low].mean()) if low.any() else np.nan
        high_mean = float(s[high].mean()) if high.any() else np.nan
        delta = high_mean-low_mean
        if not np.isfinite(delta) or not np.isfinite(rho) or abs(rho) < .2 or np.sign(delta) != np.sign(rho):
            direction = "Mixed or insufficient variation; inspect dependence plot"
        elif delta > 0:
            direction = "Presence tends to raise predicted risk" if binary else "Higher encoded values tend to raise predicted risk"
        else:
            direction = "Presence tends to lower predicted risk" if binary else "Higher encoded values tend to lower predicted risk"
        feature_rows.append({"feature": name, "mean_abs_SHAP_log_odds": np.abs(s).mean(),
                             "direction": direction, "spearman_feature_SHAP": rho,
                             "contrast": contrast, "low_cut": low_cut, "high_cut": high_cut,
                             "low_n": int(low.sum()), "high_n": int(high.sum()),
                             "low_mean_SHAP": low_mean, "high_mean_SHAP": high_mean,
                             "high_minus_low_mean_SHAP": delta, "share_positive_SHAP": (s > 0).mean()})
    features = pd.DataFrame(feature_rows).sort_values("mean_abs_SHAP_log_odds", ascending=False).reset_index(drop=True)
    features.insert(0, "rank", np.arange(1, len(features)+1))
    save_table(features, "shap_all_encoded_features", "Tree-path-dependent SHAP in log-odds; positive raises modeled TARGET=1 risk relative to the model baseline. Direction is a sample association, not causal or necessarily monotonic.", markdown=False)
    save_table(features.head(20), "shap_top_features", "Top 20 of 404 encoded/engineered predictors by mean absolute SHAP on 1,000 test rows. Direction requires |Spearman rho| >= 0.2 and agreement with disjoint-tail mean SHAP contrast; 0.2 is a descriptive reporting convention, not significance. Correlated/derived features can share importance.")
    shap.summary_plot(values, transformed, feature_names=names, max_display=20,
                      show=False, plot_size=(10, 8.5), rng=np.random.default_rng(SEED))
    fig = plt.gcf()
    plt.xlabel("SHAP value (log-odds of TARGET=1)")
    plt.title("Saved final model: SHAP summary\n1,000 reproducible competition-test applications", pad=16)
    save_figure(fig, "shap_summary")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, features.head(6).feature):
        j = list(names).index(name)
        ax.scatter(transformed[:, j], values[:, j], s=9, alpha=.35, color=TEAL, edgecolors="none")
        ax.axhline(0, color="#777777", lw=.8)
        ax.set(xlabel=name, ylabel="SHAP (log-odds)")
    fig.suptitle("Top six encoded predictors: observed direction and nonlinearity\nAssociations within this model; not causal effects")
    save_figure(fig, "shap_top_dependence")
    ctx["shap_features"] = features
    ctx["shap_validation"] = {"sample_rows": SAMPLE_SIZE, "random_state": SEED,
        "population": "competition_test", "encoded_features": len(names), "shap_version": shap.__version__,
        "model_output": "raw log-odds verified against predict_proba using expit",
        "feature_perturbation": "tree_path_dependent", "background": "Stored training path counts in the fitted model; no new background fit",
        "expected_value_log_odds": baseline, "max_probability_reconstruction_error": error,
        "direction_rule": "abs(Spearman)>=0.2 and sign agrees with high-minus-low mean SHAP; otherwise mixed",
        "sample_ids_sha256": sha256(TABLES / "shap_sample_ids.csv")}
    return features.head(20)


def write_report(ctx: dict) -> str:
    current_hashes = {name: sha256(ROOT/name) for name in ctx["protected_hashes"]}
    assert current_hashes == ctx["protected_hashes"], "Saved final model/configuration changed."
    provenance = {"analysis_date": "2026-10-25", "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "shap": shap.__version__,
        "selected_analysis_threshold": ctx["selected_threshold"], "selection_basis": BENCHMARK_ID,
        "final_business_threshold": "[NEEDS INPUT]", "gain_proxy": ctx["gain"], "loss_proxy": ctx["loss"],
        "final_model_retrained": False, "authorized_temporary_cv_fits": 3,
        "protected_artifact_hashes_before": ctx["protected_hashes"], "protected_artifact_hashes_after": current_hashes,
        "test_data_sha256": sha256(ROOT / "data/raw/application_test.csv"),
        "submission_sha256": sha256(ROOT / "submissions/home_credit_hist_gradient_boosting.csv"),
        "oof_provenance": ctx["oof_provenance"], "brier_score": ctx["brier_score"], "shap": ctx["shap_validation"]}
    (OUT / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    metrics = ctx["metrics"]
    focus = metrics[metrics.threshold.isin([ctx["selected_threshold"], .125, .5])][
        ["threshold", "precision", "recall", "TN", "FP", "FN", "TP", "predicted_default_rate", "simulated_selection_rate"]]
    groups = ctx["groups"]
    groups = groups[groups.threshold.eq(ctx["selected_threshold"]) & groups.population.eq("competition_test_saved_final_model")][
        ["attribute", "group", "n", "simulated_selected", "simulated_selection_rate", "rate_status"]]
    top = ctx["shap_features"].head(10)[["rank", "feature", "mean_abs_SHAP_log_odds", "direction"]]
    report = f"""# Home Credit model-card analysis

The saved final model was not retrained or modified. Following explicit user approval, exactly three temporary folds reproduced the original CV procedure and saved out-of-fold (OOF) probabilities. The selected analysis threshold is **{ctx['selected_threshold']:.9f}**, a sourced economic benchmark, not an approved lending policy. **Final business recommendation: [NEEDS INPUT].**

## Decision rule and sourced economics

Let G be the net gain if a selected applicant repays, L the net loss if the applicant defaults, and rejection have zero incremental payoff. Select when `(1-p)*G - p*L > 0`, equivalently `p < G/(G+L)`. Predict payment difficulty when `p >= threshold`. Both G and L must share an exposure basis and horizon.

OneMain's [FY2025 earnings release]({ONEMAIN_URL}), C&I segment and additional metrics tables, provides the numerator and denominator in `sourced_margin_inputs.csv`. We derive an annual pre-credit-loss, pretax contribution proxy by subtracting interest expense and operating expense from interest income, then dividing by average net receivables. Other revenue and credit-loss provisions are excluded. This portfolio statistic is **not observed profit conditional on a repaid loan**; segment mix, taxes, capital and amortization remain unresolved.

[Distaso, Roccazzella and Vrins, July 2023 working paper]({RECOVERY_URL}), Table 1 (printed page 8), supplies financing recovery 47% (353,891 records) and banking recovery 48% (240,936). These are Italian collection cases from 2007–2019 with mandates of at most 365 days; recovery denominators and costs do not establish Home Credit economic LGD. [Bank of Italy, December 2025, page 2]({ITALY_URL}) supplies broader unsecured recovery benchmarks of 28% and 37% for 2023 and 2024, including firms and public guarantees. They are sensitivity cases, not consumer-only substitutes.

For the main **analysis benchmark**, `G = (5432 - 1270 - 1687)/24028 = {ctx['gain']:.9f}` and `L = 1 - 0.47 = {ctx['loss']:.2f}`. Thus `t = G/(G+L) = {ctx['selected_threshold']:.9f}`. This combines a one-year portfolio margin proxy with a gross recovery-loss proxy on normalized equal exposure. Neither the time/exposure bridge nor the mapping from `TARGET` to economic default is validated. The choice is fixed from external inputs, not optimized against OOF labels. The professor's **loss:profit = 7:1** example gives `t=0.125` and remains illustrative only. Its units are not comparable to the sourced profit curves.

Missing inputs are itemized in [business_inputs_needed.md](tables/business_inputs_needed.md). The dataset lacks the cash-flow horizon and the loss/cure outcome associated with its payment-difficulty label. External evidence cannot resolve these gaps. OOF reliability is shown without fitting a calibrator; see [scikit-learn's calibration documentation]({CALIBRATION_URL}).

## Validation metrics

Positive class means `TARGET=1`, the documented payment-difficulty label. The original model-development folds were recovered, not a fresh or nested holdout. Recovered mean CV AUC is {ctx['oof_provenance']['mean_cv_auc']:.9f}; pooled OOF AUC is {ctx['oof_provenance']['oof_auc']:.9f}. The final fixed-iteration model differs from these fold models.

{markdown_table(focus)}

All tested thresholds: [OOF metrics](tables/oof_threshold_metrics.md). Full confusion matrices are in `tables/oof_confusion_matrices.csv`. TN/FN are selected loans; FP/TP are rejected loans under this simulated rule. Undefined precision at zero predicted positives is preserved as undefined. Profit sensitivity assigns benchmark payoffs to labels; it is not observed monetary profit. Competition-test labels are unavailable.

![OOF threshold trade-offs](figures/threshold_tradeoffs.png)

![Benchmark payoff sensitivity](figures/profit_sensitivity.png)

## SHAP explanations

Exactly 1,000 competition-test rows are sampled without replacement with seed 42. The saved preparation and encoder transform them without fitting. [TreeExplainer]({SHAP_URL}) uses stored training path counts with `tree_path_dependent` perturbation. SHAP values explain log-odds; positive values raise modeled payment-difficulty risk relative to the model baseline. Reconstructing probabilities with the logistic transform verifies additivity (maximum absolute error {ctx['shap_validation']['max_probability_reconstruction_error']:.3g}).

{markdown_table(top)}

Directions describe encoded feature values, using Spearman association plus disjoint high/low SHAP contrasts. The descriptive |rho| >= 0.2 rule is not a significance test; mixed cases are retained. Correlated raw/engineered fields share importance, and categorical effects refer to the specific encoded level. `DAYS_BIRTH` is negative: larger values mean younger applicants. See the top-feature CSV for signed contrasts and support counts. No causal claim is made.

![SHAP summary](figures/shap_summary.png)

## Simulated selection by group

The following uses the saved final model on all 48,744 competition-test applications at **t={ctx['selected_threshold']:.9f}**. These are simulated selection rates, **not observed approvals**. Raw unknown categories and missing groups are retained. Zero-row groups have undefined rates. The OOF population is reported separately in each group CSV. Different populations must not be combined.

{markdown_table(groups)}

![Simulated selection rates](figures/simulated_selection_by_group.png)

The sample contains issued loans and does not represent all potential applicants or rejected-applicant outcomes. Group gaps are descriptive; they do not establish fairness, discrimination, or legal compliance. Small-group results need particular care. The selected threshold is a benchmark for analysis, not a final business recommendation.

## Reproduction and saved artifacts

Run `notebooks/03_model_card_analysis.ipynb` with `.venv/model-card/bin/python` (SHAP 0.50.0, scikit-learn 1.7.2). It reads persisted OOF probabilities and contains no model fitting. `src/recover_model_card_oof.py` is the separately authorized three-fit recovery utility; do not run it as part of routine notebook execution. `analysis_provenance.json` records input hashes, packages, SHAP method and final-model hash equality. Every figure is supplied as PNG and SVG; table CSVs preserve numeric precision and Markdown companions are rounded for reading. `[UNDEFINED]` denotes an undefined statistic, not missing business input. `shap_values.npz` and `tables/shap_sample_ids.csv` support exact auditing.
"""
    (OUT / "model_card_findings.md").write_text(report)
    return report
