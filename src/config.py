"""Project-wide configuration: paths, seeds, schema, and label taxonomy.

Keeping these values in one place lets every script in ``scripts/`` and every
module in ``src/`` agree on the same conventions (column names, target labels,
random seed, file locations) without duplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"

# Source files (raw, untouched downloads from Kaggle)
PERSONAL_FINANCE_CSV: Path = RAW_DIR / "Personal_Finance_Dataset.csv"
PERSONAL_TRANSACTIONS_CSV: Path = RAW_DIR / "aug_personal_transactions_with_UserId.csv"

# Pipeline outputs
MERGED_CLEAN_CSV: Path = PROCESSED_DIR / "merged_clean.csv"
TRAIN_CSV: Path = PROCESSED_DIR / "train.csv"
TEST_CSV: Path = PROCESSED_DIR / "test.csv"
FEATURES_NPZ: Path = PROCESSED_DIR / "features.npz"
PREPROCESSOR_JOBLIB: Path = MODELS_DIR / "preprocessor.joblib"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
# Every record in the merged dataset conforms to this schema regardless of
# which source it came from.  Columns missing in the source are filled with
# ``pd.NA`` and either imputed or treated as a separate category downstream.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "date",          # pandas datetime64[ns]
    "description",   # raw merchant / memo text
    "category",      # unified target label (see CATEGORY_TAXONOMY)
    "amount",        # float, always positive (sign carried by `type`)
    "type",          # 'expense' | 'income'
    "user_id",       # string id (or 'unknown')
    "account",       # account / card name (or 'unknown')
    "source",        # 'personal_finance' | 'personal_transactions'
)

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
# The two source datasets use different label vocabularies.  We unify them
# into the 12-class taxonomy below.  Keeping this explicit (rather than
# auto-derived) makes the mapping reviewable and stable across runs.
CATEGORY_TAXONOMY: dict[str, str] = {
    # ---- Food & Dining --------------------------------------------------
    "food & drink":         "Food & Dining",
    "food & dining":        "Food & Dining",
    "groceries":            "Food & Dining",
    "restaurants":          "Food & Dining",
    "coffee shops":         "Food & Dining",
    "alcohol & bars":       "Food & Dining",
    "fast food":            "Food & Dining",
    # ---- Shopping -------------------------------------------------------
    "shopping":             "Shopping",
    "electronics & software": "Shopping",
    # ---- Utilities ------------------------------------------------------
    "utilities":            "Utilities",
    "internet":             "Utilities",
    "mobile phone":         "Utilities",
    "television":           "Utilities",
    # ---- Housing --------------------------------------------------------
    "rent":                 "Housing",
    "mortgage & rent":      "Housing",
    "home improvement":     "Housing",
    # ---- Transportation -------------------------------------------------
    "gas & fuel":           "Transportation",
    "auto insurance":       "Transportation",
    # ---- Entertainment --------------------------------------------------
    "entertainment":        "Entertainment",
    "movies & dvds":        "Entertainment",
    "music":                "Entertainment",
    # ---- Health & Personal ---------------------------------------------
    "health & fitness":     "Health & Personal",
    "haircut":              "Health & Personal",
    # ---- Travel ---------------------------------------------------------
    "travel":               "Travel",
    # ---- Income ---------------------------------------------------------
    "salary":               "Income",
    "paycheck":             "Income",
    # ---- Investment -----------------------------------------------------
    "investment":           "Investment",
    # ---- Transfers ------------------------------------------------------
    "credit card payment":  "Transfers",
    # ---- Catch-all ------------------------------------------------------
    "other":                "Other",
}

UNIFIED_CATEGORIES: tuple[str, ...] = tuple(sorted(set(CATEGORY_TAXONOMY.values())))

# ---------------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------------
# Personal_Finance uses {Expense, Income}; Personal_Transactions uses
# {debit, credit}.  In personal banking, debits remove money (expense) and
# credits add money (income), so we collapse to a single binary scheme.
TYPE_TAXONOMY: dict[str, str] = {
    "expense": "expense",
    "income":  "income",
    "debit":   "expense",
    "credit":  "income",
}

# ---------------------------------------------------------------------------
# Feature engineering hyperparameters (used in src/features.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Hyperparameters for the feature pipeline.

    These values can be overridden at the call site without editing this file
    (e.g. ``FeatureConfig(tfidf_max_features=10_000)``).
    """

    tfidf_max_features: int = 5_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    apply_smote: bool = True
    smote_k_neighbors: int = 5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
