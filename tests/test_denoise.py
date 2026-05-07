"""Tests for the merchant-rule label denoiser."""
from __future__ import annotations

import pandas as pd
import pytest

from src import denoise


@pytest.mark.parametrize(
    "description,expected",
    [
        ("starbucks", "Food & Dining"),
        ("grocery store", "Food & Dining"),
        ("thai restaurant", "Food & Dining"),
        ("brewing company", "Food & Dining"),
        ("liquor store", "Food & Dining"),
        ("mortgage payment", "Housing"),
        ("hardware store", "Housing"),
        ("biweekly paycheck", "Income"),
        ("credit card payment", "Transfers"),
        ("netflix", "Entertainment"),
        ("amazon video", "Entertainment"),
        ("amazon", "Shopping"),
        ("best buy", "Shopping"),
        ("internet service provider", "Utilities"),
        ("gas company", "Utilities"),
        ("city water charges", "Utilities"),
        ("shell", "Transportation"),
        ("state farm", "Transportation"),
        ("barbershop", "Health & Personal"),
    ],
)
def test_resolve_category(description: str, expected: str):
    assert denoise._resolve_category(description) == expected


def test_amazon_video_beats_amazon_rule():
    assert denoise._resolve_category("amazon video") == "Entertainment"
    assert denoise._resolve_category("amazon") == "Shopping"


def test_gas_company_is_utility_not_transport():
    assert denoise._resolve_category("gas company") == "Utilities"
    assert denoise._resolve_category("gas station") == "Transportation"


def test_unknown_description_returns_none():
    assert denoise._resolve_category("zzz unknown merchant") is None
    assert denoise._resolve_category("") is None
    assert denoise._resolve_category(None) is None


def test_denoise_labels_only_overrides_personal_transactions():
    df = pd.DataFrame(
        {
            "description": ["starbucks", "starbucks", "score each"],
            "category":    ["Other",     "Other",     "Other"],
            "source": [
                "personal_transactions",
                "personal_finance",
                "personal_finance",
            ],
        }
    )
    out = denoise.denoise_labels(df)
    assert out["category"].tolist() == ["Food & Dining", "Other", "Other"]


def test_denoise_labels_keeps_unmatched_rows_as_is():
    df = pd.DataFrame(
        {
            "description": ["starbucks", "mystery merchant xyz"],
            "category":    ["Other",     "Other"],
            "source":      ["personal_transactions", "personal_transactions"],
        }
    )
    out = denoise.denoise_labels(df)
    assert out["category"].tolist() == ["Food & Dining", "Other"]


def test_rule_coverage_report_columns():
    df = pd.DataFrame(
        {
            "description": ["starbucks", "starbucks", "amazon", "mystery"],
            "category":    ["Other"] * 4,
            "source":      ["personal_transactions"] * 4,
        }
    )
    report = denoise.rule_coverage_report(df)
    assert list(report.columns) == ["description", "rule_category", "count"]
    assert set(report["description"]) == {"starbucks", "amazon", "mystery"}
    starbucks = report.loc[report["description"] == "starbucks"].iloc[0]
    assert starbucks["rule_category"] == "Food & Dining"
    assert starbucks["count"] == 2
