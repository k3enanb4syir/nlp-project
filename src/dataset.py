# src/dataset.py
"""
Task 2 -- Strict chronological train / validation / test splitting.

WHY NOT A RANDOM SPLIT
----------------------
A random or k-fold split of a time series is invalid here for two compounding
reasons:

  1. Direct leakage. Exchange-rate features are built from rolling windows, so a
     randomly-held-out day's neighbours -- which encode overlapping information
     -- would sit in the training set.
  2. Regime leakage. Even with non-overlapping features, training on 2026 and
     testing on 2022 lets the model learn the level, volatility regime and
     policy environment of a period it is then scored on. Reported accuracy
     would be optimistic and would not survive deployment.

THE SPLIT
---------
Three contiguous blocks in strict time order, with no shuffling and no overlap:

    train      STUDY_START .. TRAIN_END
    validation TRAIN_END+1 .. VAL_END
    test       VAL_END+1   .. STUDY_END

Validation sits entirely after training and test entirely after validation, so
every tuning decision is made on data that follows the training period, and the
test block is touched only once, at the end.

Boundaries are defined in src/config.py so they exist in exactly one place.

WARM-UP ROWS
------------
The longest rolling window is 21 trading days, so the first rows of the series
have undefined rate features. Those rows are dropped from the TRAINING block
only -- they are genuine NaNs, not missing data to impute. Validation and test
blocks start well past the warm-up and are unaffected.

Usage:
    python src/dataset.py
"""

from __future__ import annotations

import pandas as pd

try:
    from config import (
        DAILY_DATASET_CSV,
        ROLLING_WINDOWS,
        SPLIT_DIR,
        TEST_CSV,
        TRAIN_CSV,
        TRAIN_END,
        VAL_CSV,
        VAL_END,
    )
except ImportError:  # pragma: no cover
    from src.config import (
        DAILY_DATASET_CSV,
        ROLLING_WINDOWS,
        SPLIT_DIR,
        TEST_CSV,
        TRAIN_CSV,
        TRAIN_END,
        VAL_CSV,
        VAL_END,
    )

# Columns that are identifiers, targets or raw text rather than model inputs.
NON_FEATURE_COLS = {
    "date",
    "rate",           # the level itself; models use returns, not the level
    "prev_rate",
    "log_return",     # the target's continuous form
    "direction",      # the target
    "titles_concat",  # raw text, vectorised separately inside the pipeline
}


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(DAILY_DATASET_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    # A session with no qualifying news has an empty headline string, which a
    # CSV round-trip turns back into NaN. TfidfVectorizer rejects NaN, so the
    # empty-document representation has to be restored explicitly.
    if "titles_concat" in df.columns:
        df["titles_concat"] = df["titles_concat"].fillna("").astype(str)
    return df.sort_values("date").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model inputs, in stable order."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def split_chronologically(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cut the table into the three contiguous time blocks."""
    d = df["date"].dt.date

    train = df[d <= TRAIN_END].copy()
    val = df[(d > TRAIN_END) & (d <= VAL_END)].copy()
    test = df[d > VAL_END].copy()

    # The target is undefined on the first row (no previous fixing to compare
    # against) and rate features are undefined through the warm-up window.
    warmup = max(ROLLING_WINDOWS)
    before = len(train)
    train = train.iloc[warmup:].copy()
    train = train.dropna(subset=["direction"])
    if before - len(train):
        print(f"  dropped {before - len(train)} warm-up rows from train "
              f"(longest rolling window = {warmup} trading days)")

    for name, block in (("validation", val), ("test", test)):
        n_before = len(block)
        block.dropna(subset=["direction"], inplace=True)
        if n_before - len(block):
            print(f"  dropped {n_before - len(block)} undefined-target rows from {name}")

    return {"train": train, "validation": val, "test": test}


def describe(splits: dict[str, pd.DataFrame]) -> None:
    print("\nChronological split")
    print(f"{'block':<12}{'rows':>7}{'start':>14}{'end':>14}"
          f"{'UP rate':>10}{'news cov':>10}")
    for name, block in splits.items():
        if block.empty:
            print(f"{name:<12}{0:>7}")
            continue
        print(
            f"{name:<12}{len(block):>7}"
            f"{str(block['date'].min().date()):>14}"
            f"{str(block['date'].max().date()):>14}"
            f"{block['direction'].mean():>10.3f}"
            f"{block['has_news'].mean():>10.2f}"
        )

    # Explicit guard: the blocks must not overlap in time.
    order = ["train", "validation", "test"]
    for a, b in zip(order, order[1:]):
        if splits[a].empty or splits[b].empty:
            continue
        assert splits[a]["date"].max() < splits[b]["date"].min(), (
            f"{a} overlaps {b} -- chronological split violated"
        )
    print("  no temporal overlap between blocks: OK")


def main() -> None:
    df = load_daily()
    print(f"Loaded {len(df):,} trading days from {DAILY_DATASET_CSV.name}")

    splits = split_chronologically(df)
    describe(splits)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in (("train", TRAIN_CSV), ("validation", VAL_CSV), ("test", TEST_CSV)):
        splits[name].to_csv(path, index=False, encoding="utf-8")
        print(f"  wrote {path}")

    feats = feature_columns(df)
    print(f"\n{len(feats)} numeric feature columns available:")
    print("  " + ", ".join(feats))


if __name__ == "__main__":
    main()
