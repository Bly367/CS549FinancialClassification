"""Per-source cleaning: column standardization, type unification, dedupe.

Each ``clean_*`` function accepts the raw DataFrame produced by
:mod:`src.data_loader` and returns a DataFrame conforming to
:data:`src.config.CANONICAL_COLUMNS`.  Cleaning is intentionally
deterministic and side-effect free so the merge in :mod:`src.merging` can
simply concatenate the outputs.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9 &]")


def normalize_text(value: object) -> str:
    """Lowercase, trim, and collapse whitespace.

    Returned value is always a str (NaN/None becomes ``""``).  Special
    characters are stripped except ``&`` since it carries category meaning
    (e.g. ``Food & Drink``).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = _NON_ALPHANUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _normalize_category(value: object) -> str | float:
    """Map a raw category label to the unified taxonomy."""
    key = normalize_text(value)
    if not key:
        return np.nan
    mapped = config.CATEGORY_TAXONOMY.get(key)
    if mapped is None:
        logger.debug("Unmapped category encountered: %r -> falling back to 'Other'", value)
        return "Other"
    return mapped


def _normalize_type(value: object) -> str | float:
    """Map raw transaction type strings to ``expense`` / ``income``."""
    key = normalize_text(value)
    if not key:
        return np.nan
    return config.TYPE_TAXONOMY.get(key, np.nan)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Guarantee every canonical column exists, filling missing with NA."""
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(columns)]


# ---------------------------------------------------------------------------
# Per-source cleaners
# ---------------------------------------------------------------------------
def clean_personal_finance(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize the Personal Finance dataset to the canonical schema."""
    df = raw.rename(
        columns={
            "Date": "date",
            "Transaction Description": "description",
            "Category": "category",
            "Amount": "amount",
            "Type": "type",
        }
    ).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["description"] = df["description"].map(normalize_text)
    df["category"] = df["category"].map(_normalize_category)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df["type"] = df["type"].map(_normalize_type)
    df["user_id"] = "unknown"
    df["account"] = "unknown"
    df["source"] = "personal_finance"

    return _ensure_columns(df, config.CANONICAL_COLUMNS)


def clean_personal_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize the Personal Transactions dataset to the canonical schema."""
    df = raw.rename(
        columns={
            "User ID": "user_id",
            "Date": "date",
            "Description": "description",
            "Amount": "amount",
            "Transaction Type": "type",
            "Category": "category",
            "Account Name": "account",
        }
    ).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["description"] = df["description"].map(normalize_text)
    df["category"] = df["category"].map(_normalize_category)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df["type"] = df["type"].map(_normalize_type)
    df["account"] = df["account"].map(normalize_text).replace("", "unknown")

    user_ids = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64").astype("string")
    df["user_id"] = ("u" + user_ids).fillna("unknown")
    df["source"] = "personal_transactions"

    return _ensure_columns(df, config.CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Post-merge hygiene
# ---------------------------------------------------------------------------
def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose required fields are still missing after cleaning."""
    required = ["date", "description", "category", "amount", "type"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    df = df[df["description"].str.len() > 0].reset_index(drop=True)
    df = df[df["amount"] > 0].reset_index(drop=True)
    after = len(df)
    logger.info("drop_invalid_rows: %d -> %d (%d removed)", before, after, before - after)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose business-key fields are exactly duplicated.

    The business key is (date, description, amount, type, user_id, account).
    Re-occurring legitimate transactions (e.g. a recurring subscription) will
    have different dates, so they survive.
    """
    keys = ["date", "description", "amount", "type", "user_id", "account"]
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    after = len(df)
    logger.info("drop_duplicates: %d -> %d (%d removed)", before, after, before - after)
    return df
