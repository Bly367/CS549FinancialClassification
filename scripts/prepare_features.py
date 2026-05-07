"""Stage 2 — Fit feature transformers on the training set and persist arrays.

Reads ``data/processed/train.csv`` and ``data/processed/test.csv`` produced
by :mod:`scripts.build_dataset`, then:

    * Fits TF-IDF, OneHot, and StandardScaler **on the training set only**
    * Transforms both splits into sparse feature matrices
    * Optionally applies SMOTE to the training set
    * Saves features as a single ``.npz`` file
    * Saves the fitted preprocessor as ``models/preprocessor.joblib``

Run with:
    python -m scripts.prepare_features
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, features  # noqa: E402

logger = logging.getLogger("prepare_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling on the training set.",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=config.FeatureConfig.tfidf_max_features,
        help="Maximum vocabulary size for TF-IDF (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    args = parse_args()

    if not config.TRAIN_CSV.exists() or not config.TEST_CSV.exists():
        raise SystemExit(
            "Train/test CSVs not found. Run `python -m scripts.build_dataset` first."
        )

    train_df = pd.read_csv(config.TRAIN_CSV, parse_dates=["date"])
    test_df = pd.read_csv(config.TEST_CSV, parse_dates=["date"])
    logger.info("Loaded train=%d test=%d", len(train_df), len(test_df))

    feature_cfg = config.FeatureConfig(
        tfidf_max_features=args.tfidf_max_features,
        apply_smote=not args.no_smote,
    )

    preprocessor, X_train, y_train = features.fit_preprocessor(train_df, feature_cfg)
    X_test = preprocessor.transform(test_df)
    y_test = preprocessor.encode_labels(test_df["category"])
    logger.info("X_train=%s X_test=%s", X_train.shape, X_test.shape)

    X_train_resampled, y_train_resampled = features.apply_smote_if_configured(
        X_train, y_train, feature_config=feature_cfg
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(config.PROCESSED_DIR / "X_train.npz", X_train_resampled.tocsr())
    sparse.save_npz(config.PROCESSED_DIR / "X_test.npz", X_test.tocsr())
    np.save(config.PROCESSED_DIR / "y_train.npy", y_train_resampled)
    np.save(config.PROCESSED_DIR / "y_test.npy", y_test)
    joblib.dump(preprocessor, config.PREPROCESSOR_JOBLIB)

    logger.info("Saved feature arrays to %s", config.PROCESSED_DIR)
    logger.info("Saved preprocessor to %s", config.PREPROCESSOR_JOBLIB)

    print()
    print(f"Train matrix:    {X_train_resampled.shape} (post-SMOTE)")
    print(f"Test matrix:     {X_test.shape}")
    print(f"Classes ({len(preprocessor.class_names)}): {preprocessor.class_names}")


if __name__ == "__main__":
    main()
