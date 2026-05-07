"""Shared evaluation utilities for all three models.

Each model script (Random Forest, SVM, MLP) calls into these helpers so the
metrics, runtime instrumentation, and report formatting are identical
across the comparison.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for a single model's evaluation results."""

    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    composite_score: float
    train_seconds: float
    predict_seconds: float
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_best_score: float | None = None
    classification_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Saved metrics to %s", path)


# ---------------------------------------------------------------------------
# Composite score from the proposal
# ---------------------------------------------------------------------------
def composite_score(
    *, f1: float, precision: float, recall: float, accuracy: float
) -> float:
    """Weighted score defined in the project proposal.

    ``0.4 * F1 + 0.25 * Precision + 0.25 * Recall + 0.1 * Accuracy``.
    """
    return 0.4 * f1 + 0.25 * precision + 0.25 * recall + 0.1 * accuracy


# ---------------------------------------------------------------------------
# Runtime instrumentation
# ---------------------------------------------------------------------------
@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """Context manager that records elapsed wall time in seconds.

    Usage::

        with stopwatch() as t:
            ...
        elapsed = t[0]
    """
    holder: list[float] = []
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder.append(time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    *,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    train_seconds: float,
    predict_seconds: float,
    best_params: dict[str, Any] | None = None,
    cv_best_score: float | None = None,
) -> ModelMetrics:
    """Compute the full metric bundle from predictions."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    composite = composite_score(
        f1=f1_macro,
        precision=precision_macro,
        recall=recall_macro,
        accuracy=accuracy,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return ModelMetrics(
        model_name=model_name,
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        composite_score=composite,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        best_params=best_params or {},
        cv_best_score=cv_best_score,
        classification_report=report,
    )


def confusion_matrix_df(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame for easy CSV/MD export."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm, index=class_names, columns=class_names)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_summary(metrics: ModelMetrics) -> None:
    print()
    print("=" * 72)
    print(f"  {metrics.model_name}")
    print("=" * 72)
    if metrics.best_params:
        print("Best hyperparameters:")
        for k, v in sorted(metrics.best_params.items()):
            print(f"  {k}: {v}")
    if metrics.cv_best_score is not None:
        print(f"CV best F1 (macro): {metrics.cv_best_score:.4f}")
    print()
    print(f"Accuracy:           {metrics.accuracy:.4f}")
    print(f"Precision (macro):  {metrics.precision_macro:.4f}")
    print(f"Recall    (macro):  {metrics.recall_macro:.4f}")
    print(f"F1        (macro):  {metrics.f1_macro:.4f}")
    print(f"F1     (weighted):  {metrics.f1_weighted:.4f}")
    print(f"Composite score:    {metrics.composite_score:.4f}")
    print(f"Train time:         {metrics.train_seconds:.2f}s")
    print(f"Predict time:       {metrics.predict_seconds:.4f}s")
    print()
