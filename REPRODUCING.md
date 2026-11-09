# Reproducing the Home Credit portfolio

## Read or rerun the portfolio walkthrough

The committed notebooks contain narrative and saved results for browsing on GitHub. The stakeholder model card embeds its figures. The individual walkthrough reads only the committed aggregate `reports/portfolio/metrics.json`; it does not need raw data or a saved model.

Use Python 3.13.9 to match the recorded project environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `notebooks/00_individual_portfolio.ipynb` and run all cells. To run it without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute notebooks/00_individual_portfolio.ipynb \
  --output 00_individual_portfolio.ipynb --output-dir notebooks
```

Core data and model libraries are pinned to their recorded versions; supporting plotting and notebook dependencies use compatible ranges. This is not a fully locked operating-system image.

## Reproduce the original modeling work

1. Download the [Home Credit Default Risk data](https://www.kaggle.com/competitions/home-credit-default-risk/data) using your own account and accept the applicable terms.
2. Put the extracted CSVs in `data/raw/`. Modeling needs `application_train.csv`, `application_test.csv`, and `sample_submission.csv`. The working EDA also inspects supplementary tables and the data dictionary; download the full dataset to rerun it.
3. Run `notebooks/01_eda.ipynb` for the working investigation. `01_eda_submission.ipynb` is the earlier application-table report.
4. Run `notebooks/02_modeling.ipynb` from beginning to end. It fits baselines, candidates, sample-based tuning, and the retained final model. This takes substantially longer than the aggregate walkthrough and writes local model and submission artifacts.

The modeling notebook reads raw data and fits preparation inside each fold. The optional standalone preparation command below produces derived files for inspection; those full-training prepared files are not substituted for raw fold inputs in CV.

```bash
python src/data_preparation.py \
  --train data/raw/application_train.csv \
  --test data/raw/application_test.csv \
  --output-dir data/processed
```

The final fit writes `models/final_hist_gradient_boosting_pipeline.joblib`, its configuration JSON, and `submissions/home_credit_hist_gradient_boosting.csv`. These remain ignored. Recorded Kaggle scores are historical results, not locally calculated test scores or a guarantee that reruns will obtain identical leaderboard results.

## Reproduce the model card

This analysis needs the fitted model, configuration, submission, and original development-fold predictions. After modeling finishes, install the explanation dependency and recover those predictions:

```bash
python -m pip install -r requirements-model-card.txt
python src/recover_model_card_oof.py
```

The recovery utility fits three temporary cross-validation models using the original folds. It does not retrain the saved final model and refuses to overwrite an existing OOF prediction file. These are development-fold predictions, not a new independent evaluation.

Next run `notebooks/03_model_card_analysis.ipynb`, then rebuild the presentation:

```bash
python src/build_stakeholder_model_card.py
cp reports/model_card/04_model_card.ipynb notebooks/04_model_card.ipynb
```

The copy step publishes the self-contained version with embedded figures. Local HTML is optional. Generated artifacts under `reports/model_card/` remain ignored; aggregate portfolio evidence under `reports/portfolio/` is committed separately. The card's external financial assumptions describe a capstone scenario rather than actual Home Credit economics or an operational cutoff.

## Evidence and interpretation

`reports/portfolio/metrics.json` stores comparison results extracted from the saved model configuration, recorded Kaggle scores, and aggregate counts calculated from saved out-of-fold predictions. Source-file hashes provide provenance. Individual IDs, row-level outcomes and predictions, and the estimator are not included.

Running the walkthrough reproduces arithmetic and visualizations from these aggregates. Independently regenerating the underlying model evidence requires the raw-data workflow. Results may vary by software, hardware, and development choices. Current or untouched evaluation, probability calibration, and current-policy comparisons remain future work.
