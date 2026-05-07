"""Label-noise reduction for the augmented personal transactions dataset.

The ``aug_personal_transactions_with_UserId.csv`` file has heavy label noise:
the same merchant description (e.g. ``"Starbucks"``, ``"Mortgage Payment"``,
``"Biweekly Paycheck"``) appears under many unrelated category labels.  EDA
on the raw data shows, for example, that ``"Mortgage Payment"`` rows are
labelled with categories ranging from ``"Coffee Shops"`` to ``"Auto
Insurance"``.

Because there are only ~65 distinct merchant strings in this dataset and
nearly all of them are unambiguous, a small curated rule table gives a
much more reliable label than the noisy original column.  This module
contains those rules and the override logic.

Design notes:
    * Rules are conservative: each rule maps a merchant string to a single
      unified category from :data:`src.config.UNIFIED_CATEGORIES`.
    * Matching is substring-based on the lowercased description, which makes
      the rules robust to minor formatting differences.  The first rule
      whose pattern is a substring of the description wins, so rules are
      ordered from most specific to least specific.
    * Only labels are changed.  Descriptions, dates, amounts, and types are
      left untouched.
    * The Personal Finance dataset is **not** denoised here — its
      descriptions are placeholder text and carry no merchant signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MerchantRule:
    """A single substring -> category rule.

    ``pattern`` is matched as a lowercase substring against the lowercased
    description.  ``priority`` is used purely for tiebreaking when more than
    one rule matches; lower numbers win.
    """

    pattern: str
    category: str
    priority: int = 100


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------
# Rules are ordered from most-specific to least-specific.  When two rules
# both match a description, the one declared first wins (so we list things
# like ``"amazon video"`` before ``"amazon"``).
MERCHANT_RULES: tuple[MerchantRule, ...] = (
    # ----- Entertainment (must come BEFORE generic 'amazon' match) ------
    MerchantRule("amazon video", "Entertainment", priority=10),
    MerchantRule("netflix",      "Entertainment", priority=20),
    MerchantRule("spotify",      "Entertainment", priority=20),
    MerchantRule("movie theater", "Entertainment", priority=20),

    # ----- Income -------------------------------------------------------
    MerchantRule("biweekly paycheck", "Income", priority=20),
    MerchantRule("paycheck",          "Income", priority=30),

    # ----- Transfers ----------------------------------------------------
    MerchantRule("credit card payment", "Transfers", priority=20),

    # ----- Housing ------------------------------------------------------
    MerchantRule("mortgage payment",        "Housing", priority=20),
    MerchantRule("mike s construction co",  "Housing", priority=20),
    MerchantRule("hardware store",          "Housing", priority=30),

    # ----- Utilities (utility 'gas company' must precede vehicle gas) ---
    MerchantRule("internet service provider", "Utilities", priority=20),
    MerchantRule("phone company",             "Utilities", priority=20),
    MerchantRule("power company",             "Utilities", priority=20),
    MerchantRule("city water charges",        "Utilities", priority=20),
    MerchantRule("gas company",               "Utilities", priority=20),

    # ----- Transportation ----------------------------------------------
    MerchantRule("state farm",   "Transportation", priority=30),  # auto insurance
    MerchantRule("gas station",  "Transportation", priority=40),
    MerchantRule("quiktrip",     "Transportation", priority=40),
    MerchantRule("circle k",     "Transportation", priority=40),
    MerchantRule("sheetz",       "Transportation", priority=40),
    MerchantRule("go mart",      "Transportation", priority=40),
    MerchantRule("chevron",      "Transportation", priority=40),
    MerchantRule("conoco",       "Transportation", priority=40),
    MerchantRule("exxon",        "Transportation", priority=40),
    MerchantRule("valero",       "Transportation", priority=40),
    MerchantRule("shell",        "Transportation", priority=40),
    MerchantRule("bp",           "Transportation", priority=40),

    # ----- Food & Dining (groceries / restaurants / coffee / alcohol) --
    MerchantRule("grocery store",  "Food & Dining", priority=30),
    MerchantRule("blue sky market", "Food & Dining", priority=30),
    MerchantRule("starbucks",      "Food & Dining", priority=30),
    MerchantRule("liquor store",   "Food & Dining", priority=30),
    MerchantRule("brewing company", "Food & Dining", priority=30),
    MerchantRule("american tavern", "Food & Dining", priority=30),
    MerchantRule("irish pub",      "Food & Dining", priority=30),
    MerchantRule("food truck",     "Food & Dining", priority=30),
    MerchantRule("bakery place",   "Food & Dining", priority=30),
    MerchantRule("new york deli",  "Food & Dining", priority=30),
    MerchantRule("tiny deli",      "Food & Dining", priority=30),
    MerchantRule("roadside diner", "Food & Dining", priority=30),
    MerchantRule("hawaiian grill", "Food & Dining", priority=30),
    MerchantRule("steakhouse",     "Food & Dining", priority=30),
    MerchantRule("pizza place",    "Food & Dining", priority=30),
    MerchantRule("chick fil a",    "Food & Dining", priority=30),
    MerchantRule("bojangles",      "Food & Dining", priority=30),
    MerchantRule("wendy s",        "Food & Dining", priority=30),
    MerchantRule("chili s",        "Food & Dining", priority=30),
    # Generic 'restaurant' catches everything else: thai, italian, sushi,
    # mexican, japanese, greek, mediterranean, belgian, vietnamese, latin,
    # german, bbq, brunch, fancy, seafood, irish, sushi, ...
    MerchantRule("restaurant",     "Food & Dining", priority=50),

    # ----- Health & Personal -------------------------------------------
    MerchantRule("barbershop", "Health & Personal", priority=30),

    # ----- Shopping (least specific, listed last) ----------------------
    MerchantRule("best buy", "Shopping", priority=40),
    MerchantRule("target",   "Shopping", priority=40),
    MerchantRule("amazon",   "Shopping", priority=60),
)


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------
def _resolve_category(description: str) -> str | None:
    """Return the rule-derived category for a description, or None."""
    if not isinstance(description, str) or not description:
        return None
    for rule in MERCHANT_RULES:
        if rule.pattern in description:
            return rule.category
    return None


def denoise_labels(df: pd.DataFrame, *, only_source: str = "personal_transactions") -> pd.DataFrame:
    """Override noisy labels using the merchant rule table.

    Parameters
    ----------
    df:
        Canonical-schema DataFrame (post :mod:`src.cleaning`).  Must contain
        ``description``, ``category``, and ``source`` columns.
    only_source:
        Source name to apply denoising to.  Defaults to
        ``"personal_transactions"`` because the Personal Finance dataset
        descriptions are not merchant strings.

    Returns
    -------
    A new DataFrame.  Two diagnostic columns are *not* added; we just log
    aggregate statistics so downstream training code stays clean.
    """
    out = df.copy()

    mask = (out["source"] == only_source) & out["description"].notna()
    if not mask.any():
        return out

    rule_categories = out.loc[mask, "description"].map(_resolve_category)
    matched = rule_categories.notna()

    rows_evaluated = int(mask.sum())
    rows_matched = int(matched.sum())

    target_idx = out.index[mask][matched]
    new_labels = rule_categories[matched]
    original = out.loc[target_idx, "category"]
    differs = (new_labels.values != original.values)
    rows_changed = int(differs.sum())

    out.loc[target_idx, "category"] = new_labels.values

    coverage = rows_matched / rows_evaluated if rows_evaluated else 0.0
    change_rate = rows_changed / rows_matched if rows_matched else 0.0
    logger.info(
        "denoise_labels(%s): %d/%d rows matched a rule (%.1f%% coverage); "
        "%d labels overridden (%.1f%% of matched rows).",
        only_source, rows_matched, rows_evaluated, coverage * 100,
        rows_changed, change_rate * 100,
    )
    return out


def rule_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-description summary of rule matches.

    Useful for the EDA notebook / appendix of the report when justifying
    the denoising step.  Columns: description, rule_category, count.
    """
    descs = df["description"].dropna().astype(str).unique()
    rows = []
    for d in sorted(descs):
        rule_cat = _resolve_category(d)
        count = int((df["description"] == d).sum())
        rows.append({"description": d, "rule_category": rule_cat, "count": count})
    out = pd.DataFrame(rows)
    return out.sort_values("count", ascending=False).reset_index(drop=True)
