# src/evaluate.py
"""
Task 2 -- Evaluation metrics for the binary direction task.

METRIC CHOICE (requirement 3b)
------------------------------
The target is the direction of the next JISDOR fixing (UP vs DOWN), so the
metric set is classification-oriented. Four metrics are reported, each covering
a failure mode the others miss:

  * Directional Accuracy -- the headline metric, and the one that maps onto the
    project hypothesis ("does news help predict the direction of the dollar?").
    Reported alongside the majority-class rate, because on a near-balanced but
    slightly skewed target a model can look competent while predicting one class
    throughout.

  * Macro F1 -- the guard against exactly that degenerate behaviour. It averages
    F1 over the UP and DOWN classes with equal weight, so a constant predictor
    is heavily penalised however the base rate is skewed.

  * Matthews Correlation Coefficient -- a single balanced summary that is 0 for
    any constant or random predictor regardless of class balance. For financial
    direction problems, where genuine edge is small, MCC is the most honest
    single number: small positive values are meaningful, and it cannot be gamed
    by exploiting the base rate.

  * ROC-AUC -- ranking quality independent of the 0.5 threshold, which matters
    because a model may order days correctly while being poorly calibrated.

Accuracy alone would be misleading here; MCC and macro-F1 are what distinguish a
model with real signal from one that has learned the base rate.

Usage:
    python src/evaluate.py          # re-print the saved results table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from config import RESULTS_DIR
except ImportError:  # pragma: no cover
    from src.config import RESULTS_DIR

METRICS_CSV = RESULTS_DIR / "metrics.csv"


def directional_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Compute the full metric set for one model on one split."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    out: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_up": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_up": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_up": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
    }

    # ROC-AUC is undefined if the split contains a single class.
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            out["roc_auc"] = roc_auc_score(y_true, np.asarray(y_prob, dtype=float))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")

    # Context: what a constant predictor would score on this split.
    base = max(y_true.mean(), 1 - y_true.mean())
    out["majority_baseline"] = base
    out["lift_over_majority"] = out["accuracy"] - base
    # Share of days the model calls UP -- exposes degenerate constant predictors.
    out["pred_up_rate"] = float(y_pred.mean())
    return out


def confusion(y_true, y_pred) -> str:
    """Render a 2x2 confusion matrix as text."""
    cm = confusion_matrix(np.asarray(y_true).astype(int),
                          np.asarray(y_pred).astype(int), labels=[0, 1])
    return (
        "              pred DOWN  pred UP\n"
        f"  true DOWN   {cm[0,0]:>9}  {cm[0,1]:>7}\n"
        f"  true UP     {cm[1,0]:>9}  {cm[1,1]:>7}"
    )


DISPLAY_ORDER = [
    "accuracy", "majority_baseline", "lift_over_majority",
    "f1_macro", "mcc", "roc_auc", "pred_up_rate",
]


def summarise(results: pd.DataFrame, split: str | None = None) -> None:
    """Print a compact leaderboard, best directional accuracy first."""
    df = results if split is None else results[results["split"] == split]
    if df.empty:
        print("(no results)")
        return

    for split_name, block in df.groupby("split", sort=False):
        print(f"\n=== {split_name.upper()} ===")
        cols = ["model"] + [c for c in DISPLAY_ORDER if c in block.columns]
        view = block[cols].sort_values("accuracy", ascending=False)
        header = f"{'model':<26}" + "".join(f"{c[:13]:>15}" for c in cols[1:])
        print(header)
        print("-" * len(header))
        for _, row in view.iterrows():
            line = f"{row['model']:<26}"
            for c in cols[1:]:
                val = row[c]
                line += f"{val:>15.4f}" if pd.notna(val) else f"{'-':>15}"
            print(line)


def save(results: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_CSV, index=False)
    print(f"\nWrote {METRICS_CSV}")


def main() -> None:
    if not METRICS_CSV.exists():
        raise SystemExit(f"No results at {METRICS_CSV}; run src/models.py first.")
    summarise(pd.read_csv(METRICS_CSV))


if __name__ == "__main__":
    main()
