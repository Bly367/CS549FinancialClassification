"""Train and evaluate the Support Vector Machine model (Bryce's model).

Mirrors ``scripts/train_random_forest.py`` so the comparison across the
three models in the final report stays apples-to-apples: the same column
transformer (TF-IDF + one-hot + StandardScaler), the same train/test
split, the same evaluation helpers, and the same leakage-free CV protocol
where the preprocessor is re-fit inside every fold.

Why SVM doesn't use the SMOTE-balanced training arrays:
    Same reason as RF -- using ``class_weight='balanced'`` lets the SVM
    margin objective up-weight minority classes without inflating the
    training set, which keeps GridSearchCV honest. SMOTE-resampled data
    would leak synthetic neighbours across CV folds.

Kernels searched:
    Per the proposal we test both linear and RBF kernels. ``decision_-
    function_shape='ovr'`` enforces the one-vs-rest strategy described in
    Section 3.2. We tune the regularization parameter ``C`` for both
    kernels and ``gamma`` for RBF.

Outputs (in ``models/svm/``):
    svm_model.joblib                - the fitted Pipeline
    svm_label_encoder.joblib        - LabelEncoder (so predictions decode)
    svm_metrics.json                - structured metrics + best hyperparameters
    svm_classification_report.csv   - per-class precision/recall/F1
    svm_confusion_matrix.csv        - labelled confusion matrix

Run:
    python -m scripts.train_svm
    python -m scripts.train_svm --quick           # smaller grid
    python -m scripts.train_svm --no-grid-search  # fit a single RBF model
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, evaluation, features  # noqa: E402

logger = logging.getLogger("train_svm")

MODEL_DIR: Path = config.MODELS_DIR / "svm"


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline(*, random_state: int = config.RANDOM_SEED) -> Pipeline:
    """Construct the (preprocessing -> SVM) pipeline.

    The column transformer is rebuilt here (rather than imported from
    ``src.features``) so each CV fold inside GridSearchCV gets a fresh
    fit on its own training subset.
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

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
        cache_size=500,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", svm)])


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
# A list of dicts lets GridSearchCV explore kernel-specific hyperparameters
# without wasting fits on (linear + gamma) combinations that ignore gamma.
DEFAULT_GRID: list[dict[str, list]] = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C":      [0.1, 1.0, 10.0],
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C":      [1.0, 10.0, 100.0],
        "classifier__gamma":  ["scale", 0.1, 0.01],
    },
]

QUICK_GRID: list[dict[str, list]] = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C":      [1.0, 10.0],
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C":      [1.0, 10.0],
        "classifier__gamma":  ["scale"],
    },
]


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
        n_combinations = sum(int(np.prod([len(v) for v in subgrid.values()])) for subgrid in grid)
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
        model_name="Support Vector Machine",
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        best_params=best_params,
        cv_best_score=cv_best_score,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "svm_model.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "svm_label_encoder.joblib")
    metrics.save_json(MODEL_DIR / "svm_metrics.json")

    cm_df = evaluation.confusion_matrix_df(y_test, y_pred, class_names)
    cm_df.to_csv(MODEL_DIR / "svm_confusion_matrix.csv")
    logger.info("Saved confusion matrix to %s", MODEL_DIR / "svm_confusion_matrix.csv")

    report_df = pd.DataFrame(metrics.classification_report).T
    report_df.to_csv(MODEL_DIR / "svm_classification_report.csv")
    logger.info("Saved classification report to %s", MODEL_DIR / "svm_classification_report.csv")

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
