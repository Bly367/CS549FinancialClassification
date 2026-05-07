"""CS549 Spring 2026 — Financial Transaction Category Classification.

Source package containing the data preprocessing pipeline:
    - config:      paths, constants, taxonomy
    - data_loader: CSV ingestion
    - cleaning:    schema standardization, dedupe, missing-value handling
    - merging:     unification of the two source datasets
    - features:    leakage-safe feature engineering for ML models
"""

__all__ = [
    "config",
    "data_loader",
    "cleaning",
    "denoise",
    "merging",
    "features",
    "evaluation",
]
