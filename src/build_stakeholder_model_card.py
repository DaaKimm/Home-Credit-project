"""Assemble the stakeholder card from completed analysis; never fit or score a model.

The source notebook contains narrative and links to local figures, with no code
outputs or embedded images. The local publication notebook embeds those figures;
the HTML is self-contained and excludes code inputs. Exports live under the
Git-ignored reports/model_card directory.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
from pathlib import Path
import re

import nbformat
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/model_card"
TABLES = OUT / "tables"
RECOMMENDED_THRESHOLD = 0.162724
ONEMAIN = "https://www.sec.gov/Archives/edgar/data/1584207/000158420726000004/exhibit991earningsreleaseo.htm"
RECOVERY = "https://www.crc.business-school.ed.ac.uk/sites/crc/files/2024-01/Business-Cycle-and-Realized-Losses-in-the-Consumer-Credit-Industry.pdf"
ITALY = "https://www.bancaditalia.it/pubblicazioni/note-stabilita/2025-0048/Note-di-stabilita-finanziaria-e-vigilanza-n-48-ENG.pdf?language_id=1"
REASONS = "https://www.consumerfinance.gov/rules-policy/regulations/1002/9/"
EVALUATION = "https://www.consumerfinance.gov/rules-policy/regulations/1002/6/"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    def safe(value):
        return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")
    lines = ["| " + " | ".join(map(safe, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(safe, row)) + " |" for row in rows])


def percent(value):
    return "Not defined — no observations" if pd.isna(value) else f"{value:.2%}"


def main():
    provenance = json.loads((OUT / "analysis_provenance.json").read_text())
    config = json.loads((ROOT / "models/final_hist_gradient_boosting_config.json").read_text())
    protected = provenance["protected_artifact_hashes_before"]
    assert all(digest(ROOT / name) == expected for name, expected in protected.items())
    exact_t = provenance["selected_analysis_threshold"]
    assert round(exact_t, 6) == RECOMMENDED_THRESHOLD
    # Confirm the six-decimal recommendation retains all previously reported decisions.
    rounded_checks = {}
    for name, column in [(TABLES / "oof_predictions.csv", "probability"),
                         (ROOT / "submissions/home_credit_hist_gradient_boosting.csv", "TARGET")]:
        p = pd.read_csv(name, usecols=[column])[column]
        changed = int(((p < exact_t) != (p < RECOMMENDED_THRESHOLD)).sum())
        assert changed == 0
        rounded_checks[str(name.relative_to(ROOT))] = changed
    metrics = pd.read_csv(TABLES / "oof_threshold_metrics.csv")
    recommended = metrics.loc[np.isclose(metrics.threshold, exact_t, rtol=0, atol=1e-12)].iloc[0]
    current = metrics.loc[metrics.threshold.eq(.5)].iloc[0]
    groups = pd.read_csv(TABLES / "selection_rates_all_thresholds.csv")
    groups = groups[np.isclose(groups.threshold, exact_t, rtol=0, atol=1e-12)]
    scenarios = pd.read_csv(TABLES / "economic_scenarios.csv")
    payoffs = pd.read_csv(TABLES / "profit_sensitivity.csv")
    payoff = payoffs[payoffs.scenario.eq("2025_margin_financing_recovery")]
    main_payoff = payoff.loc[np.isclose(payoff.threshold, exact_t, rtol=0, atol=1e-12)].iloc[0]
    best_grid = payoff.loc[payoff.oof_label_based_payoff_per_applicant.idxmax()]
    test_outcomes = pd.read_csv(TABLES / "test_threshold_outcomes.csv")
    test_result = test_outcomes.loc[np.isclose(test_outcomes.threshold, exact_t, rtol=0, atol=1e-12)].iloc[0]
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def figure(filename, caption):
        md(f"![{caption}](../reports/model_card/figures/{filename})\n\n*{caption}*")

    md("""# Home Credit default prediction model card

**Capstone recommendation · October 25, 2026 · Prepared by Dahyun Kim**

This card explains what the model predicts, how well it performs, and the trade-offs of using it to support lending decisions. Results come from the completed project analysis; no model was retrained to prepare this card.""")

    md(f"""## Executive Summary

**Recommend a proxy-based probability threshold of 0.162724 for this capstone.** In the simulation, applications with a predicted payment-difficulty probability below **16.2724%** are selected; applications at or above it are flagged as higher risk.

The threshold is based on **externally sourced consumer-lending assumptions** about earnings and recovery. It provides a transparent capstone decision rule. **Before deployment, Home Credit would need to recalibrate this threshold using its actual profit, recovery, and loss data**, aligned to the loan terms and outcome being predicted.

At this threshold, the validation results identify **{recommended.recall:.2%}** of applicants who later had payment difficulty. Of those flagged, **{recommended.precision:.2%}** had that outcome. The simulated selection rate is **{recommended.simulated_selection_rate:.2%}** in validation and **{test_result.simulated_selection_rate:.2%}** on the competition test population. Lowering the cutoff from 0.50 finds more difficult cases, while also flagging more applicants who repaid without the recorded difficulty.

The model supports an offline risk-ranking and policy discussion. Its present use of sensitive characteristics, incomplete explanation of external scores, and unvalidated financial assumptions prevent treating this capstone recommendation as a production lending policy.""")

    md("""## Model Details

The model estimates the chance of **payment difficulty**, the competition's specific definition of the outcome. This is not automatically equivalent to a loan charge-off or a complete loss of principal. The source defines difficulty as a payment more than an undisclosed number of days late among an undisclosed number of early installments.

""" + table(["Item", "Description"], [
        ["Data source", "Kaggle Home Credit Default Risk; application-level borrower and loan information only."],
        ["Training data", "307,511 labeled applications: 24,825 with payment difficulty and 282,686 without it."],
        ["Competition test data", "48,744 separate applications with no locally available outcome labels."],
        ["Model", "Unweighted histogram gradient boosting: a sequence of decision trees that learns nonlinear patterns."],
        ["Validation", "Three stratified folds, each holding out different applicants. A row's validation prediction comes from a model that did not train on that row."],
        ["Final model", "The selected configuration was fitted on all 307,511 labeled rows. This saved final model remains unchanged."],
        ["Data preparation", "Flags missing and unusual values, corrects known invalid codes, adds age/employment/affordability measures, fills missing values, and encodes categories. Learned preparation is fitted within each validation training fold."],
        ["Excluded information", "Application ID and the outcome label are excluded from predictors. Historical supplementary tables are not joined."],
        ["Configuration record", "200 boosting iterations; learning rate 0.05; at most 31 leaves per tree; minimum 50 rows per leaf; L2 regularization 1.0; random seed 42. Final refit has early stopping disabled."],
        ["Predictor representation", "404 encoded or engineered predictors; scikit-learn 1.7.2."],
    ]) + "\n\nSource: saved final-model configuration; original modeling notebook, Data preparation summary and Final model selection sections.")

    md("""## Intended Use

**Intended users:** the capstone audience and credit-risk stakeholders comparing model performance, financial trade-offs, and group outcomes. The score can support prioritization for human review and evaluation of potential lending policies on historical data.

**Capstone decision rule:** select an application in the simulation when its score is below 0.162724. A higher score means the application is flagged, not that a real loan has been denied. Actual approval would still require underwriting, affordability checks, and applicable policy controls.

This version is not intended for autonomous approvals or denials, pricing or credit-limit decisions, production adverse-action notices, or forecasting Home Credit's realized profits. Its results do not describe rejected applicants or establish the effect of changing a lending policy.""")

    md("""## Performance Metrics

**AUC measures ranking quality across all cutoffs:** how often a randomly chosen payment-difficulty case receives a higher risk score than a randomly chosen other case. AUC of 0.50 indicates chance-level ranking; 1.00 indicates perfect ranking.

""" + table(["Evaluation", "AUC", "Interpretation"], [
        ["Mean three-fold validation", "0.757301", "Local model-development estimate; fold standard deviation 0.002343."],
        ["Pooled out-of-fold validation", "0.757289", "Combines all held-out predictions across the three folds."],
        ["Kaggle public leaderboard", "0.74531", "Recorded score on a competition-held-out partition."],
        ["Kaggle private leaderboard", "0.74479", "Recorded score on the other competition-held-out partition."],
    ]) + "\n\nKaggle performance is lower than local validation. This limits confidence that local results will transfer unchanged to a new lending population. Kaggle scores are recorded project results; the hidden outcome labels cannot be independently checked here.")

    md(table(["Measure", "Recommended 0.162724", "Comparison 0.50"], [
        ["Precision: share of flagged applicants with difficulty", percent(recommended.precision), percent(current.precision)],
        ["Recall: share of all difficult cases flagged", percent(recommended.recall), percent(current.recall)],
        ["Predicted-default rate: share flagged", percent(recommended.predicted_default_rate), percent(current.predicted_default_rate)],
        ["Simulated selection rate: share below the cutoff", percent(recommended.simulated_selection_rate), percent(current.simulated_selection_rate)],
    ]) + "\n\nThese threshold metrics use all **307,511 out-of-fold validation predictions**, not the final model's predictions on its training rows. “Default” in these metrics refers to the documented payment-difficulty label.")

    md("### What the recommended cutoff gets right and misses\n\n" + table(
        ["Actual outcome", "Selected in simulation", "Flagged as higher risk"], [
            ["No payment difficulty", f"{int(recommended.TN):,} (correctly selected)", f"{int(recommended.FP):,} (false alarms)"],
            ["Payment difficulty", f"{int(recommended.FN):,} (missed difficult cases)", f"{int(recommended.TP):,} (correctly flagged)"],
        ]) + "\n\nThe recommended cutoff still misses many difficult cases. Its trade-off is to catch substantially more than the 0.50 rule without rejecting every applicant with some risk.")
    figure("confusion_matrix_comparison.png", "Validation outcome counts at the recommended sourced threshold and the 0.50 comparison.")

    md(f"""## Decision Threshold Analysis

**Recommendation for this capstone: 0.162724.** This follows the externally sourced profit/loss framework below, rather than assuming 0.50 is optimal.

If a selected loan earns net gain **G** when repaid and incurs net loss **L** when it defaults, its expected payoff is **(1 − probability) × G − probability × L**. With no incremental payoff assigned to rejection, the break-even cutoff is **G ÷ (G + L)**. Applicants below that cutoff have positive expected payoff within this simplified framework.

""" + table(["Input", "Value used", "Basis and qualification"], [
        ["Gain proxy G", f"{provenance['gain_proxy']:.6f} per normalized exposure unit", "Annual pre-credit-loss, pretax portfolio contribution margin derived from OneMain's 2025 consumer-finance disclosures; not measured Home Credit profit conditional on repayment."],
        ["Recovery proxy", "47%", "Mean recovery for financing debts in an Italian consumer-credit study; external benchmark, not Home Credit recovery."],
        ["Loss proxy L", "53%", "1 − 47% recovery. Gross unrecovered share; not a fully costed, discounted Home Credit loss given default."],
        ["Recommended capstone cutoff", "0.162724", "0.103004828 ÷ (0.103004828 + 0.53), rounded to six decimals."],
    ]) + f"""

The gain calculation is **(5,432 − 1,270 − 1,687) ÷ 24,028 = 10.300483%**: annual interest income less interest expense and operating expense, divided by average net receivables (all source amounts in USD millions). It excludes other revenue and credit-loss provisions. Source: [OneMain FY2025 earnings release, C&I segment and additional metrics tables]({ONEMAIN}). The recovery input comes from [Distaso, Roccazzella and Vrins, Table 1, Financing row, printed page 8]({RECOVERY}).

This combines quantities from different populations and horizons on equal normalized exposure. It does not establish actual expected profit in currency. **Home Credit must recalibrate the threshold using its own loan-level profit, recoveries, collection costs, losses and timing before deployment**, including the relationship between payment difficulty and economic default. The six-decimal cutoff produces the same decisions as the unrounded formula on both saved prediction populations.
""")

    sensitivity_rows = []
    for t in [.10, .125, .15, exact_t, .20, .30, .50]:
        row = metrics.loc[np.isclose(metrics.threshold, t, rtol=0, atol=1e-12)].iloc[0]
        label = "0.162724 — recommended" if t == exact_t else f"{t:.3f}"
        sensitivity_rows.append([label, percent(row.precision), percent(row.recall), percent(row.predicted_default_rate), percent(row.simulated_selection_rate)])
    md("### Sensitivity analysis\n\nLower cutoffs flag more applicants and find more difficult cases, but increase false alarms. Higher cutoffs select more applicants and miss more difficult cases.\n\n" + table(
        ["Cutoff", "Precision", "Recall", "Share flagged", "Simulated selection"], sensitivity_rows))
    figure("threshold_tradeoffs.png", "How validation performance and simulated selection change across probability cutoffs.")

    labels = {
        "2024_margin_financing_recovery": "2024 margin + financing recovery",
        "2024_margin_banking_recovery": "2024 margin + banking recovery",
        "2024_margin_italy_unsecured_2023_recovery": "2024 margin + 2023 broader unsecured recovery",
        "2024_margin_italy_unsecured_2024_recovery": "2024 margin + 2024 broader unsecured recovery",
        "2025_margin_financing_recovery": "2025 margin + financing recovery — recommended",
        "2025_margin_banking_recovery": "2025 margin + banking recovery",
        "2025_margin_italy_unsecured_2023_recovery": "2025 margin + 2023 broader unsecured recovery",
        "2025_margin_italy_unsecured_2024_recovery": "2025 margin + 2024 broader unsecured recovery",
    }
    economic_rows = [[labels[r.scenario], percent(r.gain_proxy), percent(r.recovery_rate), f"{r.threshold:.6f}"]
                     for r in scenarios.itertuples() if r.scenario in labels]
    md("### How the financial assumptions affect the cutoff\n\n" + table(
        ["Sourced scenario", "Annual gain proxy", "Recovery proxy", "Break-even cutoff"], economic_rows) + f"""

The sourced scenarios produce cutoffs from **0.121283 to 0.165336**. The broader unsecured benchmarks include firms and public guarantees and are used only to test sensitivity. Sources: [OneMain]({ONEMAIN}), [consumer-credit recovery study]({RECOVERY}), and [Bank of Italy, December 2025, page 2]({ITALY}).

The professor's **7:1 loss-to-profit** example implies a cutoff of **1 ÷ (1 + 7) = 0.125**. It is an illustrative comparison, not the basis for the capstone recommendation.

At the recommended cutoff, the label-assigned proxy payoff is **{main_payoff.oof_label_based_payoff_per_applicant:.6f}** normalized units per validation applicant. The highest payoff among the tested cutoffs occurs at **{best_grid.threshold:.2f}** (**{best_grid.oof_label_based_payoff_per_applicant:.6f}**). That retrospective difference does not establish a deployable optimum. The recommended threshold follows the sourced assumptions rather than a claim to maximize actual lending profit.
""")
    figure("profit_sensitivity.png", "Sensitivity of proxy payoff to the cutoff and external assumptions; these are not realized financial returns.")

    names = {
        "EXT_SOURCE_3": ("External score 3", "Higher score tends to reduce predicted risk."),
        "EXT_SOURCE_2": ("External score 2", "Higher score tends to reduce predicted risk."),
        "EXT_SOURCE_1": ("External score 1", "Higher score tends to reduce predicted risk."),
        "AMT_GOODS_PRICE": ("Price of the financed purchase", "Higher recorded price tends to reduce predicted risk in this model."),
        "AMT_CREDIT": ("Loan amount", "Higher amount tends to increase predicted risk."),
        "NAME_EDUCATION_TYPE_Higher education": ("Higher-education category", "Presence of this category tends to reduce predicted risk."),
        "CODE_GENDER_M": ("Male gender category", "Presence of this category tends to increase predicted risk; sensitive-feature concern."),
        "DAYS_EMPLOYED": ("Recorded employment duration", "Shorter recorded tenure tends to increase predicted risk."),
        "DAYS_ID_PUBLISH": ("Time since identity-document issue", "A more recent issue date tends to increase predicted risk."),
        "AMT_ANNUITY": ("Loan payment amount", "Higher amount tends to increase predicted risk."),
    }
    top = pd.read_csv(TABLES / "shap_top_features.csv").head(10)
    feature_rows = [[int(r.rank), names[r.feature][0], names[r.feature][1]] for r in top.itertuples()]
    md("""## Explainability/SHAP

SHAP describes how each piece of information pushes the model's score up or down. The analysis used the **unchanged final model** and a reproducible sample of **1,000 competition-test applications**. The following are the strongest average influences, ordered by their measured contribution size.

""" + table(["Rank", "Information used", "Typical direction in the sample"], feature_rows) + """

These are model associations, not causes or promises about an individual applicant. Related fields can share influence. The three external scores dominate, but their underlying ingredients are not documented here. Gender and education also affect predictions and require careful governance.

In the figure, points to the right increase modeled risk and points to the left reduce it. Red indicates a higher feature value and blue a lower value. The horizontal scale is a technical contribution scale (log-odds), not a percentage-point change in default probability. The explanation totals were checked against the model's probabilities.
""")
    figure("shap_summary.png", "SHAP explanation of the saved final model on 1,000 sampled test applications; field names correspond to the source data.")

    adverse_rows = [
        ["Loan amount: AMT_CREDIT", "Requested credit amount contributed to the elevated model score.", "Use only when this applicant's verified amount is an adverse contributor. Do not infer inadequate income from loan size alone."],
        ["Payment amount: AMT_ANNUITY", "Required loan payment amount contributed to the elevated model score.", "Confirm the recorded payment and the actual decision process; do not claim unaffordability without supporting income/cash-flow evidence."],
        ["Employment duration: DAYS_EMPLOYED and derived duration", "Length of recorded employment contributed to the elevated model score.", "Check dates and missing/sentinel flags; assess the applicant's combined local effect and avoid double-counting the derived field."],
        ["External scores: EXT_SOURCE_1/2/3", "No customer-facing reason approved from these opaque scores alone.", "Obtain the underlying, accurate factors and identify which actually drove the decision. Do not substitute a generic low-score explanation."],
        ["Gender, age and marital-status fields", "Do not translate these sensitive traits into customer-facing reasons.", "Escalate feature use for policy and applicable-law review. Removing a label from a notice does not remove its influence from the model."],
        ["Education, purchase price and ID-document recency", "Internal review signals only in this capstone.", "Validate meaning, individual contribution and permissible use; do not imply unverified misconduct, insufficient collateral or a causal risk explanation."],
    ]
    pd.DataFrame(adverse_rows, columns=["model_signal", "candidate_internal_mapping", "required_validation"]).to_csv(TABLES / "stakeholder_adverse_action_mapping.csv", index=False)
    md(f"""## Adverse Action Mapping

**This is a proposed internal review mapping, not a set of production denial notices.** Global SHAP rankings describe the model overall; they do not establish why a particular application was declined. Before issuing a notice, a reviewer would need the applicant's actual adverse contributors, verified underlying facts, the decision record, and approved jurisdiction-specific wording.

As a U.S. reference, [Regulation B §1002.9(b)(2) and its official interpretation]({REASONS}) require specific principal reasons for adverse action; score factors alone do not substitute for the actual reasons. [§1002.6]({EVALUATION}) governs the use of prohibited bases and has specific rules for age and other information. The competition's deployment jurisdiction is not established, so this is a governance reference rather than a compliance certification.

""" + table(["Model signal", "Candidate internal mapping", "Required validation"], adverse_rows) + """

No applicant notices are generated by this notebook. A deployment process should consolidate overlapping reasons, exclude inaccurate or unsupported descriptions, document reviewer decisions, and provide a route for applicants to correct erroneous source information.""")

    md("""## Fairness Analysis

The tables show **simulated selection rates, not observed approvals**, using the recommended cutoff of **0.162724**. A selected application has a model probability below that cutoff. No different cutoffs are applied to different groups.

The primary result uses the saved final model on **48,744 competition-test applications**. Validation rates are shown separately for the **307,511 labeled applications** using held-out fold predictions. These populations and models are different and are not pooled. Raw missing and unknown categories are retained; a group with no records has no defined rate.""")
    for attr, title in [("CODE_GENDER", "Gender"), ("NAME_EDUCATION_TYPE", "Education")]:
        a = groups[groups.attribute.eq(attr)]
        test = a[a.population.eq("competition_test_saved_final_model")].set_index("group")
        oof = a[a.population.eq("training_OOF_fold_models")].set_index("group")
        rows = []
        for group, row in test.iterrows():
            val = oof.loc[group]
            label = {"F": "Female (F)", "M": "Male (M)", "XNA": "Unknown (XNA)", "<MISSING>": "Missing"}.get(group, group)
            rows.append([label, f"{int(row.n):,}", percent(row.simulated_selection_rate),
                         f"{int(val.n):,}", percent(val.simulated_selection_rate)])
        md(f"### {title}\n\n" + table(["Group", "Test applicants", "Test selection", "Validation applicants", "Validation selection"], rows))
    gender = groups[groups.attribute.eq("CODE_GENDER") & groups.population.eq("competition_test_saved_final_model")].set_index("group")
    education = groups[groups.attribute.eq("NAME_EDUCATION_TYPE") & groups.population.eq("competition_test_saved_final_model")].set_index("group")
    gender_gap = 100 * (gender.loc["F", "simulated_selection_rate"] - gender.loc["M", "simulated_selection_rate"])
    education_gap = 100 * (education.loc["Higher education", "simulated_selection_rate"] - education.loc["Secondary / secondary special", "simulated_selection_rate"])
    md(f"""The test selection rate is **{gender_gap:.2f} percentage points higher for women than men**. It is **{education_gap:.2f} points higher for higher education than secondary/secondary-special education**. These are meaningful descriptive differences for review; they do not by themselves establish a cause, fairness, or legal compliance.

Only **41** test applicants have an academic degree; their 100% selection rate should not be generalized. The validation unknown-gender group contains just **four** records. The model directly uses gender and other sensitive information, so deployment requires assessment of permissible inputs and validated alternatives. Merely concealing those fields in explanations would not resolve the issue.""")
    figure("simulated_selection_by_group.png", "Group selection under the recommended sourced threshold and the 0.50 comparison; neither represents observed lending approvals.")

    md("""## Limitations and Risks

| Limitation | Business implication and required response |
| --- | --- |
| Financial proxies come from other lenders, markets and recovery windows. | Keep the 0.162724 recommendation for the capstone; recalibrate with Home Credit's actual profit, recovery, loss, funding and collection data before deployment. |
| Payment difficulty is not the same event as economic default. | Establish how the label relates to cures, charge-offs and cash losses, with a consistent prediction horizon and exposure basis. |
| Model development and evaluation used the same labeled population. | The recovered folds reproduce development results, not a fresh untouched test. Obtain independent, preferably temporal validation. |
| The final model differs from the early-stopped fold models. | Validation metrics describe the model-development configuration; final-model deployment performance still needs independent evaluation. |
| Probabilities have not been recalibrated for deployment. | Check reliability on the intended portfolio and revisit the decision threshold as outcomes and economics change. |
| Gender and other sensitive/proxy features influence the model. | Resolve permissible use, review subgroup error and selection outcomes, and validate alternative model designs before operational decisions. |
| External score ingredients are not explained. | Obtain source lineage and usable individual explanations; do not issue opaque or generic adverse-action reasons. |
| The data contain issued loans rather than the full applicant pool. | Simulated selection cannot estimate actual approval access or outcomes for rejected applicants. |
| Some groups are small, and no uncertainty intervals are reported. | Avoid strong fairness conclusions; expand the evaluation and quantify uncertainty. |
| No operational monitoring or review workflow has been validated. | Assign ownership, monitor data/score drift and outcomes, review errors and complaints, and define escalation and stop rules. |

These limitations do not change the capstone recommendation. They define the work needed to turn it into an accountable lending process.""")

    md(f"""## Sources

**Project evidence.** The completed `03_model_card_analysis.ipynb` and its local exports provide the results reused here:

| Finding | Exact project record |
| --- | --- |
| Model, data and configuration | `models/final_hist_gradient_boosting_config.json`: `training_data`, `classifier_parameters`, `final_refit`; original `02_modeling.ipynb`, Data preparation summary and Final model selection. |
| CV and Kaggle AUC | Same configuration: `selection_evidence` and `kaggle_score`; `reports/model_card/tables/oof_provenance.json`. |
| Threshold metrics and confusion counts | `reports/model_card/tables/oof_threshold_metrics.csv`; `oof_confusion_matrices.csv`. |
| Economics and sensitivity | `sourced_margin_inputs.csv`, `sourced_recovery_inputs.csv`, `economic_scenarios.csv`, `profit_sensitivity.csv` in the same tables directory. |
| SHAP rank and direction | `shap_top_features.csv`, `shap_sample_ids.csv`; `reports/model_card/analysis_provenance.json`, `shap` section. |
| Gender and education rates | `selection_rates_all_thresholds.csv` and the two `selection_rates_CODE_GENDER` / `selection_rates_NAME_EDUCATION_TYPE` tables. |

**External evidence and methods.**

1. [OneMain Holdings, FY2025 earnings release, February 5, 2026]({ONEMAIN}). C&I segment, fiscal-year columns and additional metrics. Financial inputs for 2024 and 2025; margin is calculated by this project, not reported as Home Credit profit.
2. [Distaso, Roccazzella and Vrins, *Business cycle and realized losses in the consumer credit industry*, July 25, 2023 working paper]({RECOVERY}). Table 1, printed page 8, Financing and Banking rows; Section 3 describes the collection sample. Italian financing recovery: 47% across 353,891 records; banking recovery: 48% across 240,936.
3. [Bank of Italy, *Bad loan recovery rates in 2024*, December 2025]({ITALY}). Printed page 2, unsecured positions: 28% in 2023 and 37% in 2024; mixed populations and public guarantees limit comparability.
4. [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html). Stored training-path background and additive explanation method; project used SHAP 0.50.0 with seed 42.
5. [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Interpretation and checking of predicted probabilities.
6. [CFPB Regulation B §1002.9]({REASONS}) and [§1002.6]({EVALUATION}), consulted October 25, 2026. U.S. references for specific adverse-action reasons and evaluation of applications; deployment jurisdiction requires its own review.

**Integrity and scope.** This card reuses existing predictions and explanations; it performs no training, scoring or SHAP recomputation. The final-model and configuration hashes match the completed analysis. The earlier three temporary CV fits were explicitly authorized to recover OOF predictions. The professor's 7:1 example comes from the capstone discussion and is illustrative only.
""")

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({"title": "Home Credit stakeholder model card", "language_info": {"name": "python"},
        "model_card": {"date": "2026-10-25", "recommended_capstone_threshold": RECOMMENDED_THRESHOLD,
                       "status": "Recommended proxy-based threshold for capstone; recalibration required before deployment",
                       "training_performed": False, "source": "Completed model-card analysis"}})
    source_path = ROOT / "notebooks/04_model_card.ipynb"
    nbformat.write(notebook, source_path)

    publication = copy.deepcopy(notebook)
    image_pattern = re.compile(r"!\[([^]]*)\]\(\.\./reports/model_card/figures/([^)]*)\)")
    for cell in publication.cells:
        matches = image_pattern.findall(cell.source)
        if matches:
            cell.attachments = {}
            for caption, filename in matches:
                raw = (OUT / "figures" / filename).read_bytes()
                cell.attachments[filename] = {"image/png": base64.b64encode(raw).decode("ascii")}
                cell.source = cell.source.replace(f"../reports/model_card/figures/{filename}", f"attachment:{filename}")
    local_path = OUT / "04_model_card.ipynb"
    nbformat.write(publication, local_path)
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(publication, resources={"metadata": {"name": "Home Credit stakeholder model card"}})
    style = """<style>
    body { background: #fff !important; color: #20262d; }
    main { max-width: 1120px; margin: 0 auto; }
    .jp-Notebook { padding: 32px 40px !important; }
    .jp-RenderedHTMLCommon { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 16px; line-height: 1.65; }
    .jp-RenderedHTMLCommon h1 { font-size: 32px; color: #173d49; }
    .jp-RenderedHTMLCommon h2 { margin-top: 32px; color: #173d49; }
    .jp-RenderedHTMLCommon table { font-size: 14px; width: 100%; margin: 18px 0; }
    .jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td { text-align: left !important; white-space: normal; vertical-align: top; padding: 9px 12px !important; border: 1px solid #d8e0e4; }
    .jp-RenderedHTMLCommon th { background: #eaf1f4; }
    .jp-RenderedHTMLCommon img { display: block; max-width: 100%; height: auto; margin: 16px auto; }
    @media print { .jp-Notebook { padding: 0 !important; } h2,h3 { break-after: avoid; } tr,img { break-inside: avoid; } }
    </style>"""
    body = body.replace("</head>", style + "\n</head>")
    (OUT / "04_model_card.html").write_text(body)
    markdown = "\n\n".join(c.source for c in notebook.cells).replace("../reports/model_card/figures/", "figures/")
    (OUT / "stakeholder_model_card.md").write_text(markdown + "\n")
    assert "[NEEDS INPUT]" not in markdown
    assert all(digest(ROOT / name) == expected for name, expected in protected.items())
    metadata = {"recommended_capstone_threshold": RECOMMENDED_THRESHOLD,
        "unrounded_source_threshold": exact_t, "rounding_changed_decisions": rounded_checks,
        "recommendation_basis": "Externally sourced consumer-lending gain/recovery proxies; user selected for capstone",
        "deployment_requirement": "Recalibrate with Home Credit actual profit, recovery and loss data before deployment",
        "model_retrained": False, "new_cv_fits": 0, "source_analysis_sha256": digest(OUT / "analysis_provenance.json"),
        "protected_model_hashes_verified": protected,
        "source_notebook": str(source_path.relative_to(ROOT)),
        "publication_notebook": str(local_path.relative_to(ROOT)),
        "html_code_inputs_hidden": True,
        "input_table_hashes": {p.name: digest(p) for p in TABLES.glob("*.csv") if not p.name.startswith("stakeholder_")}}
    (OUT / "stakeholder_card_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Created source and publication notebooks with {len(cells)} narrative cells.")
    print("HTML embeds all figures; code inputs are excluded; protected model hashes match.")


if __name__ == "__main__":
    main()
