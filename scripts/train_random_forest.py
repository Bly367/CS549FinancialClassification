"""Train and evaluate the Random Forest baseline (Brian's model).

This script builds a self-contained sklearn :class:`~sklearn.pipeline.Pipeline`
that combines feature engineering (TF-IDF on description, one-hot encoding
of categorical columns, scaling of numeric columns) with the RF classifier.
Wrapping everything in a single Pipeline lets ``GridSearchCV`` re-fit the
feature transformers inside each CV fold, which is the standard
leakage-free protocol.

Why RF doesn't need SMOTE:
    Random Forest's split criterion can use ``class_weight='balanced'`` to
    weight gain by inverse class frequency, which addresses imbalance
    without inflating the training set.  Empirically this matches or
    outperforms SMOTE on tree models and keeps CV honest.

Outputs (in ``models/random_forest/``):
    rf_model.joblib                - the fitted Pipeline
    rf_metrics.json                - structured metrics + best hyperparameters
    rf_classification_report.csv   - per-class precision/recall/F1
    rf_confusion_matrix.csv        - labelled confusion matrix

Run:
    python -m scripts.train_random_forest
    python -m scripts.train_random_forest --quick           # smaller grid
    python -m scripts.train_random_forest --no-grid-search  # fit a single model
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, evaluation, features  # noqa: E402

logger = logging.getLogger("train_random_forest")

MODEL_DIR: Path = config.MODELS_DIR / "random_forest"


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline(*, random_state: int = config.RANDOM_SEED) -> Pipeline:
    """Construct the (preprocessing -> RF) pipeline.

    The same column transformer used in ``src.features`` is replicated here
    so the model script is self-contained and CV can re-fit transformers
    per fold.
    """
    text_pipe = TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    numeric_pipe = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipe, features.TEXT_FEATURE),
            ("categorical", categorical_pipe, list(features.CATEGORICAL_FEATURES)),
            ("numeric", numeric_pipe, list(features.NUMERIC_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
DEFAULT_GRID: dict[str, list] = {
    "classifier__n_estimators":      [200, 400, 600],
    "classifier__max_depth":         [None, 20, 40],
    "classifier__min_samples_leaf":  [1, 2, 5],
    "classifier__max_features":      ["sqrt", "log2"],
}

QUICK_GRID: dict[str, list] = {
    "classifier__n_estimators":     [200, 400],
    "classifier__max_depth":        [None, 30],
    "classifier__min_samples_leaf": [1, 2],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a smaller hyperparameter grid.")
    parser.add_argument("--no-grid-search", action="store_true", help="Skip GridSearchCV; fit defaults only.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds (default: 3).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for GridSearchCV.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Top-level routine
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    args = parse_args()

    if not config.TRAIN_CSV.exists() or not config.TEST_CSV.exists():
        raise SystemExit(
            "Train/test CSVs not found. Run `python -m scripts.build_dataset` first."
        )

    logger.info("Loading train/test from %s", config.PROCESSED_DIR)
    train_df = pd.read_csv(config.TRAIN_CSV, parse_dates=["date"])
    test_df = pd.read_csv(config.TEST_CSV, parse_dates=["date"])
    logger.info("train=%d rows, test=%d rows", len(train_df), len(test_df))

    train_df = features.add_date_features(train_df)
    test_df = features.add_date_features(test_df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["category"])
    y_test = label_encoder.transform(test_df["category"])
    class_names = list(label_encoder.classes_)
    logger.info("Classes (%d): %s", len(class_names), class_names)

    pipeline = build_pipeline()

    best_params: dict = {}
    cv_best_score: float | None = None

    if args.no_grid_search:
        logger.info("Skipping grid search (fitting pipeline with defaults).")
        with evaluation.stopwatch() as t:
            pipeline.fit(train_df, y_train)
        train_seconds = t[0]
    else:
        grid = QUICK_GRID if args.quick else DEFAULT_GRID
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=config.RANDOM_SEED)
        n_combinations = int(np.prod([len(v) for v in grid.values()]))
        logger.info(
            "Grid search: %d combinations x %d folds = %d fits",
            n_combinations, args.cv_folds, n_combinations * args.cv_folds,
        )

        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=args.n_jobs,
            verbose=1,
            refit=True,
        )

        with evaluation.stopwatch() as t:
            search.fit(train_df, y_train)
        train_seconds = t[0]

        pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = float(search.best_score_)
        logger.info("Best CV F1 (macro): %.4f", cv_best_score)
        logger.info("Best params: %s", best_params)

    with evaluation.stopwatch() as t:
        y_pred = pipeline.predict(test_df)
    predict_seconds = t[0]

    metrics = evaluation.evaluate(
        model_name="Random Forest",
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        best_params=best_params,
        cv_best_score=cv_best_score,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "rf_model.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "rf_label_encoder.joblib")
    metrics.save_json(MODEL_DIR / "rf_metrics.json")

    cm_df = evaluation.confusion_matrix_df(y_test, y_pred, class_names)
    cm_df.to_csv(MODEL_DIR / "rf_confusion_matrix.csv")
    logger.info("Saved confusion matrix to %s", MODEL_DIR / "rf_confusion_matrix.csv")

    report_df = pd.DataFrame(metrics.classification_report).T
    report_df.to_csv(MODEL_DIR / "rf_classification_report.csv")
    logger.info("Saved classification report to %s", MODEL_DIR / "rf_classification_report.csv")

    evaluation.print_summary(metrics)

    print("Per-class F1 scores:")
    for cls in class_names:
        row = metrics.classification_report.get(cls, {})
        print(f"  {cls:<20s}  precision={row.get('precision', 0):.3f}  recall={row.get('recall', 0):.3f}  f1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")

    print()
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
