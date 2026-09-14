# 🩺 GlucoGuard AI — Next-Level Diabetes Risk Screening

A polished Streamlit portfolio project for binary diabetes-risk prediction.

## Architecture

```text
CSV dataset
   │
   ├── validation
   │
   ├── clinical zero → missing handling
   │
   ├── stratified train/test split
   │
   ├── 5-fold CV
   │      ├── Logistic Regression
   │      ├── Random Forest
   │      └── Gradient Boosting
   │
   ├── automatic model selection by CV ROC-AUC
   │
   ├── untouched holdout evaluation
   │
   ├── permutation importance
   │
   └── Streamlit prediction UI
             │
             └── SQLite prediction history
```

## Included

- Premium dashboard UI
- Risk assessment page
- Probability output
- Automatic model selection
- Cross-validation
- Holdout metrics
- Confusion matrix
- Classification report
- Permutation feature importance
- SQLite history
- CSV export
- Dataset explorer
- Optional PDF dependency for future report generation
- Streamlit-ready project structure

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Put the project in a GitHub repository and deploy:

```text
app.py
```

The dataset must remain beside `app.py`:

```text
diabetes (2).csv
```

## SQLite warning

SQLite requires no server and is excellent for a portfolio/demo. However,
local filesystem/database persistence should not be treated as permanent
storage on Streamlit Community Cloud.

For a real multi-user application, use a hosted database such as PostgreSQL.

## ML note

The model selection criterion is 5-fold ROC-AUC on the training split.
The final metrics are calculated on an untouched 20% holdout split.

No performance number is hard-coded.

## Medical disclaimer

This software is for educational/research purposes. It is not a medical
device and is not a substitute for clinical assessment or professional
medical advice.
