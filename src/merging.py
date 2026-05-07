"""Combine the two cleaned datasets into a single canonical frame."""
from __future__ import annotations

import logging

import pandas as pd

from src import cleaning, config, data_loader, denoise

logger = logging.getLogger(__name__)


def build_merged_dataset(*, apply_denoise: bool = True) -> pd.DataFrame:
    """End-to-end: load -> clean -> denoise -> concatenate -> de-duplicate.

    Parameters
    ----------
    apply_denoise:
        When ``True`` (default), apply the merchant-rule label corrections
        from :mod:`src.denoise` to ``personal_transactions`` rows before
        merging.  Set to ``False`` to keep the original noisy labels (useful
        for ablation studies / proving the denoise step actually helps).

    Returns a DataFrame with the columns listed in
    :data:`src.config.CANONICAL_COLUMNS`, sorted by date for readability.
    """
    pf_clean = cleaning.clean_personal_finance(data_loader.load_personal_finance())
    pt_clean = cleaning.clean_personal_transactions(data_loader.load_personal_transactions())

    logger.info("personal_finance after clean:      %d rows", len(pf_clean))
    logger.info("personal_transactions after clean: %d rows", len(pt_clean))

    if apply_denoise:
        pt_clean = denoise.denoise_labels(pt_clean)

    merged = pd.concat([pf_clean, pt_clean], ignore_index=True, copy=False)
    logger.info("concatenated:                       %d rows", len(merged))

    merged = cleaning.drop_invalid_rows(merged)
    merged = cleaning.drop_duplicates(merged)

    merged = merged.sort_values(["date", "source"], kind="mergesort").reset_index(drop=True)
    return merged


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary frame useful for the EDA notebook / report."""
    return (
        df.groupby(["source", "category"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["source", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
