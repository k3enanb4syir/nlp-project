# src/ablation.py
"""
Task 3, step 4 -- the full ablation matrix, both targets, block-bootstrap CIs.

Each row adds or removes exactly one kind of information, so a change in the
score can be put down to that one change. Every row is fitted the same way:
logistic regression and XGBoost, each with its Task 2 grid crossed with the
decay half-life grid; the single best VALIDATION macro-F1 wins (family
included). The test block is scored once per row.

THE ROWS (per target)
---------------------
A        rates only                                   time-series baseline
B        rates + Task 2 lexicons                      does basic NLP help?
C1-C4    rates + channel axes (ax_*), one encoder     static vs contextual,
         each: glove, fasttext, finbert, sbert         domain vs general
C-fbs    rates + FinBERT's own sentiment              "why not just FinBERT
                                                       sentiment?"
C-raw    rates + PCA of session-mean raw embeddings   do axes beat dumping
         (best encoder; k components chosen on val)   vectors?
C5       best C + novelty and concentration           does news surprise help?
C6       gated late fusion (best encoder, C5 text)    fusion strategy
C6-stack the same without the gate                    does the gate itself help?
drop-*   C5 minus one channel's _dir and _attn        which channel matters

"Best encoder" = the C1-C4 row with the highest validation macro-F1 for that
target. It is chosen on validation, never on test.

BOOTSTRAP
---------
For every row, MCC(row) - MCC(A) on the test block, with a 95% percentile CI
from a circular block bootstrap (BOOTSTRAP_BLOCK-session blocks,
BOOTSTRAP_REPS resamples). Both models are scored on the SAME resampled days,
so the CI is for the difference itself. `p_gt0` is the share of resamples in
which the row beats A.

C-RAW FEATURES
--------------
`python src/ablation.py --build-raw` reads the gitignored embedding cache
(data/embeddings/*.npy), averages headline vectors per session, fits PCA on
TRAINING sessions only, and writes data/cleaned/text_features/raw_pca_<enc>.csv
(committed, so the ablation itself needs no embedding cache). If those files
are missing, the C-raw row is skipped with a message.

Usage:
    python src/ablation.py --build-raw     # once, needs data/embeddings/
    python src/ablation.py                 # the matrix, ~2-3 min on CPU
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

try:
    from anchors import CHANNELS
    from config import (
        ABLATION_BOOTSTRAP_CSV,
        ABLATION_METRICS_CSV,
        ABLATION_PREDICTIONS_CSV,
        BOOTSTRAP_BLOCK,
        BOOTSTRAP_REPS,
        DECAY_HALFLIVES,
        FUSION_TARGETS,
        RANDOM_SEED,
        RAW_PCA_COMPONENTS,
        RAW_PCA_K_GRID,
        RESULTS_DIR,
        TEXT_FEATURES_DIR,
        TRAIN_END,
    )
    from embeddings import ENCODERS
    from evaluate import directional_metrics
    from fusion import (
        NOVELTY_COLS,
        Fitted,
        FusionData,
        _meta_inputs,
        _X,
        fit_late,
        select_and_fit,
    )
    from models import news_features, rate_features
except ImportError:  # pragma: no cover
    from src.anchors import CHANNELS
    from src.config import (
        ABLATION_BOOTSTRAP_CSV,
        ABLATION_METRICS_CSV,
        ABLATION_PREDICTIONS_CSV,
        BOOTSTRAP_BLOCK,
        BOOTSTRAP_REPS,
        DECAY_HALFLIVES,
        FUSION_TARGETS,
        RANDOM_SEED,
        RAW_PCA_COMPONENTS,
        RAW_PCA_K_GRID,
        RESULTS_DIR,
        TEXT_FEATURES_DIR,
        TRAIN_END,
    )
    from src.embeddings import ENCODERS
    from src.evaluate import directional_metrics
    from src.fusion import (
        NOVELTY_COLS,
        Fitted,
        FusionData,
        _meta_inputs,
        _X,
        fit_late,
        select_and_fit,
    )
    from src.models import news_features, rate_features

C_ROWS = {"glove": "C1", "fasttext": "C2", "finbert": "C3", "sbert": "C4"}


# --------------------------------------------------------------------------
# C-raw features
# --------------------------------------------------------------------------

def raw_pca_path(encoder: str):
    return TEXT_FEATURES_DIR / f"raw_pca_{encoder}.csv"


def build_raw_pca(encoder: str) -> pd.DataFrame:
    """Session-mean raw embeddings -> PCA fitted on training sessions only."""
    from sklearn.decomposition import PCA

    try:
        from embeddings import load_articles, load_embeddings
    except ImportError:  # pragma: no cover
        from src.embeddings import load_articles, load_embeddings

    articles = load_articles()
    X = load_embeddings(encoder, articles)
    means = pd.DataFrame(X).groupby(articles["trading_day"].to_numpy()).mean()
    train = means.index <= pd.Timestamp(TRAIN_END)
    pca = PCA(n_components=RAW_PCA_COMPONENTS, random_state=RANDOM_SEED)
    pca.fit(means[train].to_numpy())
    Z = pca.transform(means.to_numpy())
    out = pd.DataFrame(Z, columns=[f"raw_pc{i + 1}" for i in range(Z.shape[1])])
    out.insert(0, "date", means.index)
    out.to_csv(raw_pca_path(encoder), index=False, float_format="%.6f")
    ev = np.cumsum(pca.explained_variance_ratio_)
    print(f"[{encoder}] dim {X.shape[1]} -> {Z.shape[1]} PCs; train variance explained "
          + ", ".join(f"k={k}: {ev[k - 1]:.2f}" for k in RAW_PCA_K_GRID))
    return out


def load_data(encoder: str) -> FusionData:
    """FusionData with the C-raw columns merged in, when they exist."""
    data = FusionData(encoder)
    path = raw_pca_path(encoder)
    if path.exists():
        raw = pd.read_csv(path, parse_dates=["date"])
        data.base = data.base.merge(raw, on="date", how="left")
    return data


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def best_of(name: str, data: FusionData, cols: list[str], target: str,
            halflives=DECAY_HALFLIVES) -> Fitted:
    """Logistic and XGBoost grids; the family with the better validation macro-F1."""
    fits = [select_and_fit(f"{name} [{fam}]", data, cols, target, fam, halflives)
            for fam in ("logreg", "xgb")]
    best = max(fits, key=lambda f: f.val_f1)
    best.name = name
    return best


class Matrix:
    """Collects metrics and test predictions for one target."""

    def __init__(self, target: str):
        self.target = target
        self.metrics: list[dict] = []
        self.preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.test_dates: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    def _row(self, row, desc, encoder, family, halflife, params, split, y, pred, prob,
             in_sample=False, n_features=None):
        self.metrics.append({
            "target": self.target, "row": row, "description": desc, "encoder": encoder,
            "family": family, "decay_halflife": halflife, "params": params,
            "n_features": n_features, "split": split, "val_in_sample": in_sample,
            **directional_metrics(y, pred, prob)})

    def _keep_test(self, row, block, pred, prob):
        dates = block["date"].to_numpy()
        if self.test_dates is None:
            self.test_dates, self.y_test = dates, block[self.target].astype(int).to_numpy()
        assert (dates == self.test_dates).all(), f"{row}: test rows differ"
        self.preds[row] = (np.asarray(pred, dtype=int), np.asarray(prob, dtype=float))

    def fitted(self, row: str, desc: str, encoder: str, f: Fitted, extra: dict | None = None):
        params = json.dumps({**f.params, **(extra or {})})
        for split, block in f.splits.items():
            X = _X(block, f.cols)
            pred, prob = f.pipe.predict(X), f.pipe.predict_proba(X)[:, 1]
            self._row(row, desc, encoder, f.family, f.halflife, params, split,
                      block[self.target], pred, prob, n_features=len(f.cols))
            if split == "test":
                self._keep_test(row, block, pred, prob)

    def late(self, row: str, desc: str, encoder: str, lf, rates: Fitted, text: Fitted):
        params = json.dumps({k: round(v, 4) for k, v in lf.coefs.items()})
        for split in ("validation", "test"):
            block = text.splits[split]
            M = _meta_inputs(rates, text, split, lf.gated)
            pred, prob = lf.meta.predict(M), lf.meta.predict_proba(M)[:, 1]
            self._row(row, desc, encoder, "late", text.halflife, params, split,
                      block[self.target], pred, prob, in_sample=(split == "validation"),
                      n_features=len(rates.cols) + len(text.cols))
            if split == "test":
                self._keep_test(row, block, pred, prob)


def run_target(target: str, datas: dict[str, FusionData]) -> Matrix:
    print(f"\n=== target: {target} ===")
    mx = Matrix(target)
    ref = datas[ENCODERS[0]]
    rates = rate_features(ref.base)
    lex = news_features(ref.base)
    axes = [c for c in ref.base.columns if c.startswith("ax_")]

    mx.fitted("A", "rates only", "-", best_of("A", ref, rates, target, (0,)))
    mx.fitted("B", "rates + Task 2 lexicons", "-",
              best_of("B", ref, rates + lex, target, (0,)))

    c_fits: dict[str, Fitted] = {}
    for enc in ENCODERS:
        row = C_ROWS[enc]
        f = best_of(row, datas[enc], rates + axes, target)
        mx.fitted(row, f"rates + {enc} axes", enc, f)
        c_fits[enc] = f

    fin = datas["finbert"]
    fbs = [c for c in fin.base.columns if c.startswith("fbs_")]
    mx.fitted("C-fbs", "rates + FinBERT sentiment", "finbert",
              best_of("C-fbs", fin, rates + fbs, target))

    best = max(c_fits, key=lambda e: c_fits[e].val_f1)
    data = datas[best]
    print(f"  best encoder on validation: {best} ({C_ROWS[best]}, "
          f"macro-F1 {c_fits[best].val_f1:.4f})")

    raw_cols = [c for c in data.base.columns if c.startswith("raw_pc")]
    if raw_cols:
        raw_best, raw_k = None, None
        for k in RAW_PCA_K_GRID:
            f = best_of(f"C-raw k={k}", data, rates + raw_cols[:k], target, (0,))
            if raw_best is None or f.val_f1 > raw_best.val_f1:
                raw_best, raw_k = f, k
        raw_best.name = "C-raw"
        mx.fitted("C-raw", f"rates + PCA({raw_k}) of raw {best} embeddings", best,
                  raw_best, {"k": raw_k})
    else:
        print(f"  C-raw skipped: {raw_pca_path(best).name} missing "
              "(run: python src/ablation.py --build-raw)")

    text5 = axes + list(NOVELTY_COLS)
    mx.fitted("C5", f"rates + {best} axes + novelty", best,
              best_of("C5", data, rates + text5, target))

    # Late fusion needs logits, so both base models are logistic.
    a_lr = select_and_fit("C6 rates model", data, rates, target, "logreg", (0,))
    text_lr = select_and_fit("C6 text model", data, text5, target, "logreg", DECAY_HALFLIVES)
    for gated, row, desc in ((True, "C6", "gated late fusion"),
                             (False, "C6-stack", "late fusion, no gate")):
        lf = fit_late(row, a_lr, text_lr, gated)
        mx.late(row, f"{desc} ({best})", best, lf, a_lr, text_lr)

    for ch in CHANNELS:
        drop = {f"ax_{ch}_dir", f"ax_{ch}_attn"}
        cols = rates + [c for c in text5 if c not in drop]
        mx.fitted(f"drop-{ch}", f"C5 without the {ch} channel", best,
                  best_of(f"drop-{ch}", data, cols, target))
    return mx


# --------------------------------------------------------------------------
# Block bootstrap
# --------------------------------------------------------------------------

def block_bootstrap_indices(n: int, block: int, reps: int, seed: int) -> np.ndarray:
    """reps x n indices from a circular block bootstrap."""
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(reps, -1)[:, :n]


def mcc_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """MCC of each row of two (reps x n) 0/1 arrays; 0 when undefined (as sklearn)."""
    y, p = y.astype(float), p.astype(float)
    tp = (y * p).sum(1)
    tn = ((1 - y) * (1 - p)).sum(1)
    fp = ((1 - y) * p).sum(1)
    fn = (y * (1 - p)).sum(1)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)


def bootstrap(mx: Matrix) -> pd.DataFrame:
    y = mx.y_test
    idx = block_bootstrap_indices(len(y), BOOTSTRAP_BLOCK, BOOTSTRAP_REPS, RANDOM_SEED)
    yb = y[idx]
    base_pred = mx.preds["A"][0]
    base_full = mcc_rows(y[None], base_pred[None])[0]
    base_b = mcc_rows(yb, base_pred[idx])
    rows = []
    for row, (pred, _) in mx.preds.items():
        full = mcc_rows(y[None], pred[None])[0]
        diff = mcc_rows(yb, pred[idx]) - base_b
        rows.append({"target": mx.target, "row": row, "mcc_test": full,
                     "mcc_A": base_full, "d_mcc": full - base_full,
                     "ci_low": float(np.percentile(diff, 2.5)),
                     "ci_high": float(np.percentile(diff, 97.5)),
                     "p_gt0": float((diff > 0).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(targets=FUSION_TARGETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    datas = {enc: load_data(enc) for enc in ENCODERS}
    metrics, boots, preds = [], [], []
    for target in targets:
        mx = run_target(target, datas)
        metrics.append(pd.DataFrame(mx.metrics))
        boots.append(bootstrap(mx))
        for row, (pred, prob) in mx.preds.items():
            preds.append(pd.DataFrame({"date": mx.test_dates, "target": target, "row": row,
                                       "actual": mx.y_test, "pred": pred, "prob": prob}))

    metrics = pd.concat(metrics, ignore_index=True)
    boots = pd.concat(boots, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ABLATION_METRICS_CSV, index=False)
    boots.to_csv(ABLATION_BOOTSTRAP_CSV, index=False)
    pd.concat(preds, ignore_index=True).to_csv(ABLATION_PREDICTIONS_CSV, index=False)

    pd.set_option("display.width", 160)
    test = metrics[metrics["split"] == "test"]
    for target in targets:
        t = (test[test["target"] == target]
             .merge(boots[boots["target"] == target], on=["target", "row"])
             .set_index("row"))
        print(f"\n{target} -- test block, MCC difference vs A with 95% block-bootstrap CI")
        print(t[["encoder", "family", "decay_halflife", "accuracy", "f1_macro", "mcc",
                 "roc_auc", "d_mcc", "ci_low", "ci_high", "p_gt0"]].round(3).to_string())
    for p in (ABLATION_METRICS_CSV, ABLATION_BOOTSTRAP_CSV, ABLATION_PREDICTIONS_CSV):
        print(f"Wrote {p}")
    return metrics, boots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-raw", action="store_true",
                        help="build C-raw PCA features from data/embeddings/ and exit")
    parser.add_argument("--targets", nargs="+", choices=FUSION_TARGETS,
                        default=list(FUSION_TARGETS))
    args = parser.parse_args()
    if args.build_raw:
        TEXT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        for enc in ENCODERS:
            build_raw_pca(enc)
        return
    run(tuple(args.targets))


if __name__ == "__main__":
    main()
