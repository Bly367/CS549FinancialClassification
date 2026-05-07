"""Sanity checks for the cleaning + merging pipeline.

Run with:
    python -m pytest tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cleaning, config


def test_normalize_text_handles_missing():
    assert cleaning.normalize_text(None) == ""
    assert cleaning.normalize_text(np.nan) == ""
    assert cleaning.normalize_text("  Hello,   World! ") == "hello world"
    assert cleaning.normalize_text("Food & Drink") == "food & drink"


def test_clean_personal_finance_conforms_to_canonical_schema():
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Transaction Description": ["Rent payment", "Score each."],
            "Category": ["Rent", "Food & Drink"],
            "Amount": [1200.0, -42.5],
            "Type": ["Expense", "Expense"],
            "source": ["personal_finance", "personal_finance"],
        }
    )
    cleaned = cleaning.clean_personal_finance(raw)
    assert list(cleaned.columns) == list(config.CANONICAL_COLUMNS)
    assert cleaned["category"].tolist() == ["Housing", "Food & Dining"]
    assert (cleaned["amount"] >= 0).all()
    assert cleaned["type"].tolist() == ["expense", "expense"]


def test_clean_personal_transactions_maps_credit_debit():
    raw = pd.DataFrame(
        {
            "User ID": [1.0, 2.0, np.nan],
            "Date": ["1/2/2018", "1/3/2018", "1/4/2018"],
            "Description": ["Amazon", "Mortgage Payment", "Paycheck"],
            "Amount": [11.11, 1247.44, 2000.0],
            "Transaction Type": ["debit", "debit", "credit"],
            "Category": ["Shopping", "Mortgage & Rent", "Paycheck"],
            "Account Name": ["Platinum Card", "Checking", "Checking"],
            "source": ["personal_transactions"] * 3,
        }
    )
    cleaned = cleaning.clean_personal_transactions(raw)
    assert cleaned["type"].tolist() == ["expense", "expense", "income"]
    assert cleaned["category"].tolist() == ["Shopping", "Housing", "Income"]
    assert cleaned["user_id"].tolist() == ["u1", "u2", "unknown"]


def test_drop_duplicates_removes_exact_repeats():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "description": ["amazon"] * 3,
            "category": ["Shopping"] * 3,
            "amount": [10.0, 10.0, 20.0],
            "type": ["expense"] * 3,
            "user_id": ["u1"] * 3,
            "account": ["platinum card"] * 3,
            "source": ["personal_transactions"] * 3,
        }
    )
    deduped = cleaning.drop_duplicates(df)
    assert len(deduped) == 2


def test_drop_invalid_rows_filters_zero_and_missing():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-01-02")],
            "description": ["amazon", "", "mortgage payment"],
            "category": ["Shopping", "Housing", None],
            "amount": [10.0, 5.0, 200.0],
            "type": ["expense", "expense", "expense"],
            "user_id": ["u1", "u2", "u3"],
            "account": ["a", "b", "c"],
            "source": ["personal_transactions"] * 3,
        }
    )
    cleaned = cleaning.drop_invalid_rows(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["description"] == "amazon"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Groceries", "Food & Dining"),
        ("Coffee Shops", "Food & Dining"),
        ("Mortgage & Rent", "Housing"),
        ("Salary", "Income"),
        ("Paycheck", "Income"),
        ("Credit Card Payment", "Transfers"),
    ],
)
def test_category_taxonomy_coverage(raw: str, expected: str):
    assert config.CATEGORY_TAXONOMY[raw.lower()] == expected
