# Home-Credit-project

Assignment_1

Dahyun Kim

## Project Overview

Many people with limited or no credit history struggle to obtain loans and may be vulnerable to untrustworthy lenders. Home Credit aims to promote financial inclusion by using alternative data, such as telecommunications and transaction records, to assess applicants' repayment ability. This project applies statistical and machine learning methods to improve those predictions, helping qualified borrowers gain access to loans with suitable terms and repayment plans.

## Project contents

- [`notebooks/`](notebooks/) contains the exploratory analysis notebooks.
- [`reports/`](reports/) contains submitted reports and rendered EDA output.
- [`reports/analysis/`](reports/analysis/) contains the data map, EDA plan, and feature-feasibility assessment in Markdown.

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
