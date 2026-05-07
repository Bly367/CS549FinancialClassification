# CS549 — Financial Transaction Category Classification

Spring 2026 team project. Combines two Kaggle financial datasets and runs a
reproducible preprocessing pipeline so each team member can plug in their
model on top of identical features.

**Team:** Brian Ly · Bryce Rambach · Kien Tu

## Repository layout

```
ProjectCS549/
├── data/
│   ├── raw/                              # original Kaggle CSVs (do not edit)
│   │   ├── Personal_Finance_Dataset.csv
│   │   └── aug_personal_transactions_with_UserId.csv
│   └── processed/                        # outputs from the pipeline (gitignored)
├── docs/                                 # proposal + course project spec
├── src/                                  # importable Python package
│   ├── config.py                         # paths, seeds, taxonomy, hyperparameters
│   ├── data_loader.py                    # raw CSV loading
│   ├── cleaning.py                       # per-source standardization & dedupe
│   ├── denoise.py                        # merchant-rule label correction
│   ├── merging.py                        # combine cleaned frames
│   ├── features.py                       # split + TF-IDF + scaling + SMOTE
│   └── evaluation.py                     # metrics, runtime, confusion matrix helpers
├── scripts/
│   ├── build_dataset.py                  # stage 1: produce merged_clean.csv + train/test split
│   ├── prepare_features.py               # stage 2: fit transformers, save feature arrays
│   └── train_random_forest.py            # stage 3a: Random Forest (Brian)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── tests/
│   └── test_cleaning.py                  # quick sanity checks
├── models/                               # fitted preprocessor (gitignored)
├── requirements.txt
└── README.md
```

## Setup

Tested on Python 3.11. macOS / Linux / WSL.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the pipeline

The pipeline is split into two stages so model code can re-load the cleaned
CSV without paying the feature-engineering cost every time, and so the
feature-engineering step can be re-run with different hyperparameters
without re-cleaning the raw data.

### Stage 1 — combine + clean + split

```bash
python -m scripts.build_dataset
```

Outputs in `data/processed/`:

- `merged_clean.csv` — both datasets combined into one canonical schema
- `train.csv` / `test.csv` — stratified 80/20 split (seed = 42)
- `category_summary.csv` — per-source class counts

### Stage 2 — feature engineering

```bash
python -m scripts.prepare_features
```

Outputs:

- `data/processed/X_train.npz`, `X_test.npz` — sparse feature matrices
- `data/processed/y_train.npy`, `y_test.npy` — encoded labels
- `models/preprocessor.joblib` — fitted `ColumnTransformer` + `LabelEncoder`

Optional flags:

```bash
python -m scripts.prepare_features --no-smote
python -m scripts.prepare_features --tfidf-max-features 10000
```

### Stage 3 — train a model

```bash
python -m scripts.train_random_forest                 # full grid search (~45s)
python -m scripts.train_random_forest --quick         # smaller grid (~10s)
python -m scripts.train_random_forest --no-grid-search # fit a single RF with defaults
```

Outputs to `models/random_forest/`:

- `rf_model.joblib` — fitted `Pipeline` (preprocessing + classifier)
- `rf_label_encoder.joblib` — fitted `LabelEncoder`
- `rf_metrics.json` — accuracy, per-class P/R/F1, runtime, best hyperparameters
- `rf_classification_report.csv`
- `rf_confusion_matrix.csv`

### Tests

```bash
python -m pytest tests
```

## Label noise reduction

Exploratory analysis of `aug_personal_transactions_with_UserId.csv` showed
that the original labels are essentially random with respect to the
description: rows with `Description = "Mortgage Payment"` appear under
`"Coffee Shops"`, `"Auto Insurance"`, and many other unrelated labels. The
augmented dataset only contains 65 distinct merchant strings, and almost
all of them are unambiguous (e.g. `Starbucks`, `Netflix`, `Biweekly
Paycheck`), so a curated substring rule table gives a much more reliable
label than the original column.

`src/denoise.py` defines that rule table. For each row in
`personal_transactions`, the first matching rule overrides the original
category; rules are ordered most-specific-first so `"Amazon Video"` is
mapped to `Entertainment` before the generic `"amazon"` rule maps anything
remaining to `Shopping`. The Personal Finance dataset is left untouched
(its descriptions are placeholder text that carries no merchant signal).

In our run the denoiser matched **100% of the 10,806 augmented rows** and
**overrode 75% of the labels** (8,103 rows). The Random Forest ablation
quantifies the impact:

| Metric          | Without denoise | With denoise | Δ        |
|-----------------|-----------------|--------------|----------|
| Accuracy        | 0.281           | 0.898        | **+0.62** |
| Precision (m)   | 0.256           | 0.836        | +0.58    |
| Recall (m)      | 0.284           | 0.795        | +0.51    |
| F1 (macro)      | 0.227           | 0.775        | +0.55    |
| F1 (weighted)   | 0.254           | 0.927        | +0.67    |
| Composite score | 0.254           | 0.807        | +0.55    |

To reproduce the ablation:

```bash
python -m scripts.build_dataset --no-denoise && python -m scripts.train_random_forest --quick
python -m scripts.build_dataset             && python -m scripts.train_random_forest
```

## What the pipeline does (and why)

The two source datasets disagree on schema, label vocabulary, and even on
how the transaction direction is encoded (`Expense`/`Income` vs
`debit`/`credit`). The pipeline reconciles all of that **before** any model
training.

| Step | Implementation | Why it matters |
|---|---|---|
| Column standardization | `cleaning.clean_personal_finance` / `clean_personal_transactions` | Each source is renamed to the canonical schema (`date, description, category, amount, type, user_id, account, source`). Missing columns are added as `unknown` so a single concat works. |
| Date parsing | `pd.to_datetime(..., errors="coerce")` | The two datasets use different date formats; coercing surfaces bad rows as `NaT` instead of crashing. |
| Type unification | `TYPE_TAXONOMY` in `config.py` | `debit→expense`, `credit→income`, so models see one binary feature instead of two vocabularies. |
| Category unification | `CATEGORY_TAXONOMY` in `config.py` | 32 raw labels collapse into 12 unified classes (`Food & Dining`, `Housing`, `Utilities`, `Transportation`, `Shopping`, `Entertainment`, `Travel`, `Health & Personal`, `Income`, `Investment`, `Transfers`, `Other`). The mapping is explicit so it is reviewable. |
| Label denoise | `MERCHANT_RULES` in `src/denoise.py` | Substring rules override the noisy label on `personal_transactions` rows; see *Label noise reduction* above. |
| Text normalization | `cleaning.normalize_text` | Lowercases, strips punctuation, collapses whitespace. Keeps `&` because category names use it. |
| Amount sanitation | `pd.to_numeric(...).abs()` | Sign is captured by the `type` column, so amounts are stored as positive magnitudes. |
| Missing-value handling | `cleaning.drop_invalid_rows` | Drops rows missing any required field or with non-positive amounts. Categorical missing values become `"unknown"` instead of being dropped. |
| Duplicate removal | `cleaning.drop_duplicates` | Business key = `(date, description, amount, type, user_id, account)`. Recurring transactions on different dates are preserved. |
| Stratified split | `features.stratified_split` | 80/20 split that preserves per-class proportions. Falls back to a random split if any class has < 2 samples. |
| TF-IDF (description) | `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` | Captures merchant/keyword signal; bigrams help phrases like `grocery store`. |
| One-hot (`type`, `account`, `source`) | `OneHotEncoder(handle_unknown="ignore")` | Lets unseen account names at inference time degrade gracefully. |
| Numeric scaling | `StandardScaler` on `amount` + calendar features | Required by SVM and MLP, harmless for Random Forest. |
| Calendar features | `features.add_date_features` | `day_of_week`, `day_of_month`, `month`, `is_weekend`. |
| Class imbalance | `imblearn.over_sampling.SMOTE` (training set only) | Up-samples minority classes; never touches the test set, so the test metrics stay honest. |

### Leakage safety

Every transformer (`TfidfVectorizer`, `OneHotEncoder`, `StandardScaler`,
`LabelEncoder`, `SMOTE`) is fit on **the training split only**, then applied
to the test split. The fitted preprocessor is persisted to
`models/preprocessor.joblib` so model code can re-use the exact same
transformation at inference time.

## Loading the processed data in your model code

```python
import joblib
import numpy as np
from scipy import sparse

from src import config

X_train = sparse.load_npz(config.PROCESSED_DIR / "X_train.npz")
X_test  = sparse.load_npz(config.PROCESSED_DIR / "X_test.npz")
y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
y_test  = np.load(config.PROCESSED_DIR / "y_test.npy")

preprocessor = joblib.load(config.PREPROCESSOR_JOBLIB)
class_names = preprocessor.class_names    # for confusion matrices, etc.
```

## Models

### Random Forest (Brian)

Built in `scripts/train_random_forest.py`. The script wraps preprocessing
(TF-IDF + one-hot + StandardScaler) and the `RandomForestClassifier` in a
single `sklearn.pipeline.Pipeline`, then runs `GridSearchCV` over the
hyperparameters listed in the proposal:

```python
DEFAULT_GRID = {
    "classifier__n_estimators":     [200, 400, 600],
    "classifier__max_depth":        [None, 20, 40],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_features":     ["sqrt", "log2"],
}
```

CV is 3-fold stratified, scoring on `f1_macro`. Wrapping the preprocessor
inside the Pipeline means TF-IDF and the scaler are re-fit on each training
fold, so CV scores are leakage-free. RF uses `class_weight="balanced"`
instead of the SMOTE training arrays, which keeps CV honest and is a
better fit for tree models.

**Held-out test results (12-class, 2,462 rows):**

| Metric            | Value  |
|-------------------|--------|
| Accuracy          | 0.8981 |
| Precision (macro) | 0.8359 |
| Recall (macro)    | 0.7949 |
| F1 (macro)        | 0.7746 |
| F1 (weighted)     | 0.9272 |
| Composite score   | 0.8074 |
| Train time        | ~41s   |
| Predict time      | <0.05s |

Best hyperparameters found: `n_estimators=200`, `max_depth=40`,
`min_samples_leaf=1`, `max_features="log2"`.

**Per-class observations:**

- Perfect or near-perfect F1 on all classes derived from the (now
  denoised) `personal_transactions` dataset: `Transfers` (1.00),
  `Transportation` (1.00), `Food & Dining` (0.97), `Utilities` (0.95),
  `Shopping` (0.91), `Income` (0.91), `Entertainment` (0.90).
- Lower F1 on classes that exist *only* in the Personal Finance dataset
  (`Travel`, `Investment`, `Other`) because that dataset's descriptions
  are random word strings with no merchant signal. The model learns to
  map those rows onto the available labels using only `amount` and
  `type`, which is a noisy signal.
- This is a property of the source data, not a defect of the pipeline; the
  final report should call this out as a limitation and an opportunity for
  the comparative analysis.

### SVM (Bryce) and MLP (Kien)

Stubs to come. Both models can either:
- Load `data/processed/{X_train,X_test}.npz` + `y_*.npy` produced by
  `scripts/prepare_features.py` (the SMOTE-balanced training arrays), or
- Build their own Pipeline like `scripts/train_random_forest.py` does so
  CV is leakage-free.

Whichever they choose, they should call into `src/evaluation.py` so the
metrics and confusion matrices are formatted identically across all three
models for the comparison table in the report.

## Reproducibility

- All randomness uses `config.RANDOM_SEED = 42`.
- The preprocessor is the only stateful artifact; deleting `data/processed/`
  and re-running both scripts reproduces every output exactly.

## Datasets

- **Personal Finance Data** — https://www.kaggle.com/datasets/ramyapintchy/personal-finance-data
- **Personal Transactions (with UserID)** — https://www.kaggle.com/datasets/shyakanobledavid/personal-transactions-userid-new-transactions
