"""Raw dataset loading.

Each loader returns a :class:`pandas.DataFrame` with a ``source`` column
already attached, but otherwise leaves the raw schema untouched.  All
column-name standardization, date parsing, and label unification happens in
:mod:`src.cleaning` and :mod:`src.merging` so this layer stays a pure I/O
boundary.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_personal_finance(path: Path = config.PERSONAL_FINANCE_CSV) -> pd.DataFrame:
    """Load the Kaggle 'Personal Finance Data' CSV.

    Source columns: ``Date, Transaction Description, Category, Amount, Type``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Personal Finance dataset not found at {path}")

    df = pd.read_csv(path, dtype={"Amount": "float64"})
    df["source"] = "personal_finance"
    logger.info("Loaded personal_finance: %d rows, %d cols", len(df), df.shape[1])
    return df


def load_personal_transactions(path: Path = config.PERSONAL_TRANSACTIONS_CSV) -> pd.DataFrame:
    """Load the Kaggle 'Personal Transactions (with UserID)' CSV.

    Source columns: ``User ID, Date, Description, Amount, Transaction Type,
    Category, Account Name``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Personal Transactions dataset not found at {path}")

    df = pd.read_csv(path, dtype={"Amount": "float64"})
    df["source"] = "personal_transactions"
    logger.info("Loaded personal_transactions: %d rows, %d cols", len(df), df.shape[1])
    return df
