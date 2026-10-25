# Home-Credit-project

Assignment_1

Dahyun Kim

## Project Overview

Many people with limited or no credit history struggle to obtain loans and may be vulnerable to untrustworthy lenders. Home Credit aims to promote financial inclusion by using alternative data, such as telecommunications and transaction records, to assess applicants' repayment ability. This project applies statistical and machine learning methods to improve those predictions, helping qualified borrowers gain access to loans with suitable terms and repayment plans.

## Project contents

- [`notebooks/`](notebooks/) contains the exploratory analysis notebooks.
- [`reports/`](reports/) contains submitted reports and rendered EDA output.
- [`reports/analysis/`](reports/analysis/) contains the data map, EDA plan, and feature-feasibility assessment in Markdown.
- [`src/modeling.py`](src/modeling.py) contains reusable, leakage-safe model pipelines and cross-validation utilities.
- [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb) documents the modeling experiments and results.

Raw Home Credit data are intentionally excluded from version control. Place local
copies in `data/raw/` to reproduce the analysis.

## Data preparation

[`src/data_preparation.py`](src/data_preparation.py) turns the reversible
application-level decisions in `notebooks/01_eda.ipynb` into reusable
functions. It never edits raw CSV files and preserves `TARGET` unchanged in
the prepared training data.

| EDA decision | Preparation action |
|---|---|
| Missingness review (cells 5--6 and 20--21) | Adds indicators for every source feature that is missing in training, without imputation. |
| Sentinel/data-quality remediation (cell 87) | Replaces `DAYS_EMPLOYED == 365243` and `CODE_GENDER == "XNA"` with missing values while retaining flags; flags and nulls positive relative-day values. |
| Extreme-value remediation (cells 87 and 89) | Learns training-only 99.9th-percentile thresholds and adds flags, without capping or deleting source values. |
| Stable numeric views (cell 87) | Adds `log1p` versions of the documented monetary fields. |
| Applicant and affordability analysis (cells 36--41 and 46--49) | Adds age and valid employment duration in years, an EDA-defined age-band feature, plus safe credit-to-income and annuity-to-income ratios. |

The command-line entry point reads raw application CSVs and writes derived
train/test CSVs plus a JSON file containing the training-fitted parameters:

Run the complete preparation step with local raw data:

```bash
python src/data_preparation.py \
  --train data/raw/application_train.csv \
  --test data/raw/application_test.csv \
  --output-dir data/processed
```

This writes prepared train/test CSVs and the training-fitted 99.9th-percentile
thresholds to `data/processed/`, which is intentionally ignored by Git. The
script fits both the thresholds and the set of missingness indicators on
training data only, then applies the same parameters to test data.

The generated files are:

- `data/processed/application_train_prepared.csv`: 307,511 rows and 210 columns, including `TARGET`.
- `data/processed/application_test_prepared.csv`: 48,744 rows and 209 columns.
- `data/processed/application_preparation_parameters.json`: the training-fitted extreme-value thresholds and missing-indicator choices.

Supplementary-table features are not implemented. They were optional, and the
EDA did not establish sufficiently reliable pre-application timing for every
candidate historical feature. Other EDA items that are descriptive, fairness,
or governance reviews are not feature transformations; unverified extremes are
flagged rather than capped or deleted, as the EDA requires.

`validate_train_test()` is run before and after transformation. It verifies a
non-missing, unique `SK_ID_CURR` for every row and identical feature names and
order in train and test, allowing only the training-only `TARGET` column. With
the local supplied data, validation found 307,511 unique training IDs and
48,744 unique test IDs; the prepared splits have 209 shared features and no
schema differences.

## Modeling

The executed [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb)
compares the majority-class and L2-regularized logistic baselines with bounded
random-forest and gradient-boosting candidates. Generated HTML checkpoints are
kept locally at `reports/02_modeling_baselines.html`,
`reports/02_modeling_candidates.html`, `reports/02_modeling_tuning.html`, and
`reports/02_modeling_final.html`. They are reproducible from the notebook and
intentionally excluded from Git.

All learned preparation and model preprocessing is inside the scikit-learn
pipeline evaluated by three-fold stratified cross-validation. Each fold learns
its own extreme-value thresholds, missing-indicator schema, imputations,
one-hot categories, and numeric scaling. An index-level audit verifies that
these steps see only the training rows and no validation rows during fitting.

| Model | Mean CV AUC | Fold SD | OOF AUC | Mean accuracy | Total CV seconds |
|---|---:|---:|---:|---:|---:|
| Histogram gradient boosting, unweighted | **0.757301** | 0.002343 | **0.757289** | 0.919483 | 67.94 |
| Histogram gradient boosting, balanced | 0.757196 | 0.002134 | 0.757166 | 0.710329 | 67.75 |
| Logistic regression, balanced | 0.749993 | 0.003327 | 0.749975 | 0.688076 | 227.91 |
| Logistic regression, unweighted | 0.749777 | 0.002926 | 0.749762 | 0.919265 | 41.69 |
| Random forest, balanced subsample | 0.731238 | 0.001951 | 0.731228 | 0.812771 | 197.29 |
| Majority-class dummy | 0.500000 | 0.000000 | 0.500000 | 0.919271 | 0.01 |

The dummy result illustrates why accuracy is not the selection metric for this
imbalanced target. The two logistic models are treated as essentially tied:
their fold-level AUC difference changes sign and their mean difference is only
0.000216. The boosting weight comparison is also effectively tied at this
stage; weighting changes mean AUC by -0.000105 and materially changes the
default-threshold classifications.

LightGBM is not installed in the project environment. The comparison therefore
uses scikit-learn's `HistGradientBoostingClassifier`, an available
histogram-binned gradient booster with class weights and early stopping. Its
internal early-stopping sample is drawn only from each outer training fold.

### Histogram-boosting tuning

`RandomizedSearchCV` evaluated 20 parameter combinations with ROC-AUC scoring
and three-fold stratified CV on a reproducible 5,000-row sample (404 positive
rows). The search covered bounded learning rates, iteration caps, mutually
exclusive leaf/depth constraints, minimum leaf sizes, and L2 regularization.
It ran serially so multiple dense fold matrices could not compete for memory.

The best sample score was 0.720072. This is a noisy parameter-selection score,
not a full-data performance estimate. The selected parameters were:

| Parameter | Selected value |
|---|---:|
| `learning_rate` | 0.075 |
| `max_iter` | 200 |
| `max_leaf_nodes` | 15 |
| `max_depth` | `None` |
| `min_samples_leaf` | 60 |
| `l2_regularization` | 10.0 |

The selected configuration was then evaluated on all 307,511 rows using the
exact original folds:

| Configuration | Mean CV AUC | Fold SD | OOF AUC | Mean accuracy | Total CV seconds |
|---|---:|---:|---:|---:|---:|
| Default unweighted boosting | 0.757301 | 0.002343 | 0.757289 | 0.919483 | 67.94 |
| Sample-tuned unweighted boosting | 0.757940 | 0.002260 | 0.757936 | 0.919483 | 55.24 |

The tuned-minus-default fold differences were `+0.001031`, `+0.000272`, and
`+0.000613`. Although tuned boosting improved every fold, its mean gain
(`+0.000638`) and OOF gain (`+0.000646`) did not reach the predeclared
`+0.001` thresholds. The project therefore retains the default unweighted
histogram-boosting configuration.

The 5,000 tuning rows come from the same labeled population used by the later
full-fold confirmation, so this is a model-development comparison rather than
a fully nested, untouched estimate. Twenty trials are not exhaustive, and
three paired folds do not support a strong significance claim.

### Final model and submission

The final model is the retained default, unweighted histogram gradient booster.
Iteration handling uses two training-only fits. The first fit uses the original
10% internal validation fraction, ROC-AUC scoring, 15-round patience, and a
200-iteration cap. It reached the cap at `n_iter_=200`, rather than stopping
early; the highest internal validation AUC was `0.767595` at iteration 199.
That internal score selects training duration only and is not a performance
estimate. A fresh pipeline then disables early stopping, fixes `max_iter=200`,
and refits its preparation, encoded-feature preprocessing, and classifier on
all 307,511 labeled rows. Neither the competition test rows nor their feature
distribution affects fitting or iteration selection. The calibration and
final fits took 31.34 and 32.66 seconds, respectively, in the recorded run.

`TARGET` is removed before pipeline fitting, and `SK_ID_CURR` is retained only
for submission alignment and removed before the predictor matrix. The fitted
pipeline has 404 encoded predictors. It is saved locally at
`models/final_hist_gradient_boosting_pipeline.joblib`; the reproducibility
configuration is at `models/final_hist_gradient_boosting_config.json`.

The generated submission is
`submissions/home_credit_hist_gradient_boosting.csv`. Validation after writing
and rereading the CSV produced:

| Check | Result |
|---|---:|
| Exact columns and order (`SK_ID_CURR`, `TARGET`) | Passed |
| Rows | 48,744 |
| Unique, non-missing IDs | 48,744 |
| IDs match sample submission in exact row order | Passed |
| IDs match competition test in exact row order | Passed |
| All predictions finite and in `[0, 1]` | Passed |
| Prediction minimum / maximum / mean | 0.006439 / 0.700170 / 0.079486 |
| Reloaded pipeline matches CSV probabilities | Passed within float64 round-trip tolerance |

### Kaggle results

The submission of `home_credit_hist_gradient_boosting.csv` succeeded. Kaggle's
held-out leaderboard scores are kept distinct from the local development
estimate:

| Evaluation source | ROC-AUC | Difference from mean CV AUC |
|---|---:|---:|
| Local mean 3-fold stratified CV | **0.757301** | — |
| Kaggle public leaderboard | **0.74531** | -0.011991 |
| Kaggle private leaderboard | **0.74479** | -0.012511 |

The leaderboard AUCs are about 0.012 below local CV, so the local estimate was
modestly optimistic for the competition test data. Sampling variation,
model-development selection, and train/test distribution differences are all
plausible contributors because the evaluation populations and protocols are
not identical. The public and private scores differ by only 0.00052, indicating
similar ranking performance across Kaggle's two held-out partitions.

The local artifact rules in [`.gitignore`](.gitignore) exclude `models/*` and
`submissions/*`, while retaining only each directory's `.gitkeep`. The fitted
pipeline, configuration JSON, and submission CSV were all confirmed ignored.
Only application-table features are included; the tuning comparison is not
fully nested. Probability calibration, blending, and bootstrap intervals remain
deferred. Subsequent SHAP and simulated subgroup selection analyses are documented
in the separate model-card notebook below.

For documentation-only changes such as recording Kaggle scores, regenerate the
HTML without rerunning model training:

```bash
jupyter nbconvert --to html notebooks/02_modeling.ipynb \
  --output 02_modeling_final.html --output-dir reports
```

Re-execute the original modeling notebook only when the data, preparation, or
modeling code changes:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb \
  --output 02_modeling.ipynb --output-dir notebooks \
  --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to html notebooks/02_modeling.ipynb \
  --output 02_modeling_final.html --output-dir reports
```

## Stakeholder model card

[`notebooks/04_model_card.ipynb`](notebooks/04_model_card.ipynb) presents the final
capstone model card for business stakeholders: performance, the recommended
proxy-based threshold **0.162724**, sensitivity, SHAP explanations, adverse-action
mapping, fairness, limitations, and sources. Before deployment, the threshold
must be recalibrated using Home Credit's actual profit, recovery, and loss data.

The local [HTML card](reports/model_card/04_model_card.html) shows tables and
figures with code hidden; the [self-contained notebook](reports/model_card/04_model_card.ipynb)
embeds the same figures. Rebuild these from the completed analysis with
`python src/build_stakeholder_model_card.py`; this performs no training or scoring.
The [technical analysis notebook](notebooks/03_model_card_analysis.ipynb)
documents the calculations. Notebook sources contain no execution outputs;
model artifacts and all generated reports, predictions, and figures remain
excluded from Git by [`.gitignore`](.gitignore).
