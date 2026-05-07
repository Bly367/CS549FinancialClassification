"""Feature engineering: leakage-safe train/test transformations.

The core principle here is that **every transformation is fit on the training
set only** and then applied to the test set.  This includes TF-IDF on the
description text, one-hot encoding on categorical columns, z-score scaling
on numeric columns, and SMOTE oversampling (which is only applied to the
training set, never the test set).

The fitted preprocessor is returned so it can be persisted via ``joblib`` and
re-applied to new transactions at inference time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date feature extraction
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: tuple[str, ...] = (
    "amount",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("type", "account", "source")
TEXT_FEATURE: str = "description"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical/calendar features extracted from the ``date`` column.

    Operates out-of-place; the input frame is not modified.
    """
    out = df.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    out["day_of_week"] = dt.dt.dayofweek.astype("float64")
    out["day_of_month"] = dt.dt.day.astype("float64")
    out["month"] = dt.dt.month.astype("float64")
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float64")
    return out


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split that preserves the per-class proportions.

    Classes that appear fewer than 2 times can't be stratified, so we fall
    back to an unstratified split with a warning if that happens.
    """
    counts = df["category"].value_counts()
    if (counts < 2).any():
        rare = counts[counts < 2].index.tolist()
        logger.warning("Cannot stratify rare classes %s; using random split.", rare)
        stratify = None
    else:
        stratify = df["category"]

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
@dataclass
class FittedPreprocessor:
    """Bundle of fitted transformers + helpers for inference re-use."""

    column_transformer: ColumnTransformer
    label_encoder: LabelEncoder
    feature_config: config.FeatureConfig

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        df = add_date_features(df)
        return self.column_transformer.transform(df)

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(y)

    def decode_labels(self, y: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y)

    @property
    def feature_names(self) -> list[str]:
        return list(self.column_transformer.get_feature_names_out())

    @property
    def class_names(self) -> list[str]:
        return list(self.label_encoder.classes_)


def _build_column_transformer(feature_config: config.FeatureConfig) -> ColumnTransformer:
    text_pipe = TfidfVectorizer(
        max_features=feature_config.tfidf_max_features,
        ngram_range=feature_config.tfidf_ngram_range,
        min_df=feature_config.tfidf_min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    numeric_pipe = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_FEATURE),
            ("categorical", categorical_pipe, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric_pipe, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_config: config.FeatureConfig | None = None,
) -> tuple[FittedPreprocessor, sparse.csr_matrix, np.ndarray]:
    """Fit the column transformer + label encoder on the training set.

    Returns the fitted preprocessor along with the transformed matrices for
    the training set so callers don't have to call ``transform`` immediately.
    """
    cfg = feature_config or config.FeatureConfig()
    train_df = add_date_features(train_df)

    ct = _build_column_transformer(cfg)
    X_train = ct.fit_transform(train_df)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["category"])

    logger.info(
        "Fitted preprocessor: %d features, %d classes",
        X_train.shape[1],
        len(le.classes_),
    )
    pre = FittedPreprocessor(ct, le, cfg)
    return pre, X_train, y_train


# ---------------------------------------------------------------------------
# Class imbalance handling (training set only!)
# ---------------------------------------------------------------------------
def apply_smote_if_configured(
    X: sparse.csr_matrix,
    y: np.ndarray,
    *,
    feature_config: config.FeatureConfig,
    random_state: int = config.RANDOM_SEED,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Apply SMOTE to the training set when configured to do so.

    The minimum class count must exceed ``smote_k_neighbors``; otherwise we
    skip SMOTE for that run and log a warning.  Test data must never be
    passed through this function.
    """
    if not feature_config.apply_smote:
        return X, y

    # Imported lazily so the project still installs without imbalanced-learn
    # if a user only wants to run the merge stage.
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:  # pragma: no cover
        logger.warning("imbalanced-learn not installed (%s); skipping SMOTE.", exc)
        return X, y

    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    k = min(feature_config.smote_k_neighbors, max(min_count - 1, 1))
    if min_count <= 1:
        logger.warning("Smallest class has %d samples; skipping SMOTE.", min_count)
        return X, y

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "SMOTE: %d -> %d rows (k_neighbors=%d)",
        X.shape[0],
        X_res.shape[0],
        k,
    )
    return X_res, y_res
