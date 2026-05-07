"""Stage 1 — Combine both Kaggle datasets into a clean, unified CSV.

This script is intentionally deterministic and contains no model-fitting
steps.  It reads the raw CSVs in ``data/raw/`` and writes:

    data/processed/merged_clean.csv   # full unified dataset
    data/processed/train.csv          # 80% stratified split
    data/processed/test.csv           # 20% stratified split
    data/processed/category_summary.csv

Run with:
    python -m scripts.build_dataset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow ``python scripts/build_dataset.py`` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, features, merging  # noqa: E402

logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.TEST_SIZE,
        help="Fraction of rows held out for the test set (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for the train/test split (default: %(default)s).",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help=(
            "Disable merchant-rule label correction on the personal_transactions "
            "dataset.  Use this for ablation runs that quantify how much "
            "denoising helps downstream model accuracy."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    args = parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    apply_denoise = not args.no_denoise
    logger.info("Building merged dataset (denoise=%s)", apply_denoise)
    merged = merging.build_merged_dataset(apply_denoise=apply_denoise)
    merged.to_csv(config.MERGED_CLEAN_CSV, index=False)
    logger.info("Wrote %s (%d rows)", config.MERGED_CLEAN_CSV, len(merged))

    summary = merging.summarize(merged)
    summary_path = config.PROCESSED_DIR / "category_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)

    logger.info("Performing stratified 80/20 split (seed=%d)", args.seed)
    train_df, test_df = features.stratified_split(
        merged, test_size=args.test_size, random_state=args.seed
    )
    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    logger.info("Wrote %s (%d rows)", config.TRAIN_CSV, len(train_df))
    logger.info("Wrote %s (%d rows)", config.TEST_CSV, len(test_df))

    logger.info("Done.")
    print()
    print("Merged dataset summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
