"""Train and evaluate the MLP classifier model.

Mirrors ``scripts.train_random_forest`` and ``scripts.train_svm`` so the
final report comparison stays as apples-to-apples as possible: the same
train/test split, the same raw feature recipe (TF-IDF + one-hot +
StandardScaler), the same evaluation helpers, and the same leakage-free CV
protocol where preprocessing is re-fit inside every fold.

Why MLP adds TruncatedSVD:
    ``MLPClassifier`` trains on dense numeric arrays, while TF-IDF and
    one-hot encoding produce a high-dimensional sparse matrix. We therefore
    add ``TruncatedSVD`` after preprocessing to project the sparse matrix into
    a smaller dense latent space before fitting the neural network.

Class imbalance note:
    Unlike SVM and Random Forest, sklearn's ``MLPClassifier`` does not support
    ``class_weight='balanced'`` directly. This means the MLP comparison uses
    the same train/test data and feature pipeline, but does not up-weight
    minority classes in the same built-in way as SVM/RF.

Outputs (in ``models/mlp/``):
    mlp_model.joblib               - fitted sklearn Pipeline
    mlp_label_encoder.joblib       - label encoder for category ids
    mlp_metrics.json               - aggregate metrics + best hyperparameters
    mlp_classification_report.csv  - per-class precision/recall/F1
    mlp_confusion_matrix.csv       - labelled confusion matrix

Run:
    python -m scripts.train_mlp
    python -m scripts.train_mlp --quick
    python -m scripts.train_mlp --full-grid
    python -m scripts.train_mlp --no-grid-search
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, evaluation, features  # noqa: E402

logger = logging.getLogger("train_mlp")

MODEL_DIR: Path = config.MODELS_DIR / "mlp"


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline(*, random_state: int = config.RANDOM_SEED) -> Pipeline:
    """Construct the preprocessing -> TruncatedSVD -> MLP pipeline.

    The column transformer is rebuilt here instead of imported from
    ``src.features`` so every CV fold gets a fresh preprocessing fit on only
    that fold's training subset.
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

    svd = TruncatedSVD(
        n_components=400,
        random_state=random_state,
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svd", svd),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
# Each CV fold retrains TF-IDF, one-hot encoding, SVD, and the neural network,
# so the default grid is intentionally smaller than the RF/SVM grids.
FULL_GRID: dict[str, list] = {
    "svd__n_components": [100, 200, 300, 400, 500],
    "classifier__hidden_layer_sizes": [(256, 128), (256, 128, 64), (384, 192)],
    "classifier__alpha": [1e-5, 1e-4, 1e-3],
    "classifier__learning_rate_init": [5e-4, 1e-3, 2e-3],
}

DEFAULT_GRID: dict[str, list] = {
    "svd__n_components": [200, 300],
    "classifier__hidden_layer_sizes": [(256, 128), (256, 128, 64)],
    "classifier__alpha": [1e-4, 1e-3],
    "classifier__learning_rate_init": [1e-3],
}

QUICK_GRID: dict[str, list] = {
    "svd__n_components": [200],
    "classifier__hidden_layer_sizes": [(256, 128)],
    "classifier__alpha": [1e-4],
    "classifier__learning_rate_init": [1e-3],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use the smallest hyperparameter grid.")
    parser.add_argument("--full-grid", action="store_true", help="Use the exhaustive grid; much slower.")
    parser.add_argument("--no-grid-search", action="store_true", help="Skip GridSearchCV; fit defaults only.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for GridSearchCV. Increase only if your machine has enough RAM.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Top-level routine
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )
    args = parse_args()

    if args.quick and args.full_grid:
        raise SystemExit("Choose at most one of --quick and --full-grid.")

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
        logger.info("Skipping grid search. Fitting pipeline with defaults.")
        with evaluation.stopwatch() as t:
            pipeline.fit(train_df, y_train)
        train_seconds = t[0]

    else:
        if args.full_grid:
            grid = FULL_GRID
        elif args.quick:
            grid = QUICK_GRID
        else:
            grid = DEFAULT_GRID

        cv = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=config.RANDOM_SEED,
        )

        n_combinations = int(np.prod([len(v) for v in grid.values()]))
        logger.info(
            "Grid search: %d combinations x %d folds = %d fits",
            n_combinations,
            args.cv_folds,
            n_combinations * args.cv_folds,
        )

        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        with evaluation.stopwatch() as t:
            search.fit(train_df, y_train)
        train_seconds = t[0]

        pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = float(search.best_score_)

        logger.info("Best CV F1 macro: %.4f", cv_best_score)
        logger.info("Best params: %s", best_params)

    with evaluation.stopwatch() as t:
        y_pred = pipeline.predict(test_df)
    predict_seconds = t[0]

    metrics = evaluation.evaluate(
        model_name="MLP",
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        best_params=best_params,
        cv_best_score=cv_best_score,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_DIR / "mlp_model.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "mlp_label_encoder.joblib")
    metrics.save_json(MODEL_DIR / "mlp_metrics.json")

    cm_df = evaluation.confusion_matrix_df(y_test, y_pred, class_names)
    cm_df.to_csv(MODEL_DIR / "mlp_confusion_matrix.csv")
    logger.info("Saved confusion matrix to %s", MODEL_DIR / "mlp_confusion_matrix.csv")

    report_df = pd.DataFrame(metrics.classification_report).T
    report_df.to_csv(MODEL_DIR / "mlp_classification_report.csv")
    logger.info("Saved classification report to %s", MODEL_DIR / "mlp_classification_report.csv")

    evaluation.print_summary(metrics)

    print("Per-class F1 scores:")
    for cls in class_names:
        row = metrics.classification_report.get(cls, {})
        print(
            f"  {cls:<20s}  "
            f"precision={row.get('precision', 0):.3f}  "
            f"recall={row.get('recall', 0):.3f}  "
            f"f1={row.get('f1-score', 0):.3f}  "
            f"support={int(row.get('support', 0))}"
        )

    print()
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()