# src/horizon_experiment.py
"""
Experiment -- does news predict the rupiah several sessions later?

The main pipeline asks whether the news aligned to session t predicts the
fixing at t (one session ahead). Markets may take time to act on news, so this
experiment moves the target further out, keeping features exactly as they are.

TARGETS (h = 1, 3, 5, 7 sessions)
---------------------------------
cum<h>  cumulative change from the previous fixing to h sessions later:
        log(rate_{t+h-1} / rate_{t-1}). Asks: where is the rate h days on?
del<h>  the single-session change h sessions later:
        log(rate_{t+h-1} / rate_{t+h-2}). Asks: is the reaction DELAYED?
h = 1 is the main pipeline's target (cum1 == del1). Each return becomes two
binary targets, as in Task 3: direction (> 0) and large move (|return| above
the 75th percentile of TRAINING returns at that horizon).

LEAKAGE
-------
A target h sessions ahead for the last h-1 days of the training block uses
fixings from the validation block (and likewise validation -> test). Those
rows are PURGED from train and validation, so no block's targets reach into
the next block. Features, selection and fitting are otherwise unchanged
(ablation.best_of: family x hyperparameters x decay chosen on validation,
test scored once). Overlapping cumulative targets make neighbouring test days
correlated; the block bootstrap (10-session blocks) accounts for that.

ROWS per horizon x target
-------------------------
A          rates only
C-<enc>    rates + 14 channel-axis features of one encoder (as C1-C4)
best-C     the C row with the best validation macro-F1 (the honest pick)

MODEL-FREE CHECK
----------------
Spearman correlation of every channel direction score with the single-session
return k sessions later (k = 1..10), on the training block and the test block
separately. A delayed effect would show as correlations that grow with k and
agree in sign between train and test.

Usage:
    python src/horizon_experiment.py      # ~2-4 min on CPU
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ablation import best_of, block_bootstrap_indices, mcc_rows
    from anchors import CHANNELS
    from config import (
        BOOTSTRAP_BLOCK,
        BOOTSTRAP_REPS,
        DECAY_HALFLIVES,
        LARGE_MOVE_QUANTILE,
        RANDOM_SEED,
        RESULTS_DIR,
    )
    from embeddings import ENCODERS
    from evaluate import directional_metrics
    from fusion import FusionData, _X
    from models import rate_features
except ImportError:  # pragma: no cover
    from src.ablation import best_of, block_bootstrap_indices, mcc_rows
    from src.anchors import CHANNELS
    from src.config import (
        BOOTSTRAP_BLOCK,
        BOOTSTRAP_REPS,
        DECAY_HALFLIVES,
        LARGE_MOVE_QUANTILE,
        RANDOM_SEED,
        RESULTS_DIR,
    )
    from src.embeddings import ENCODERS
    from src.evaluate import directional_metrics
    from src.fusion import FusionData, _X
    from src.models import rate_features

HORIZONS = (1, 3, 5, 7)
IC_LAGS = range(1, 11)
TARGETS = ("direction", "large_move")


def add_horizon_returns(base: pd.DataFrame) -> pd.DataFrame:
    """cum<h> and del<h> return columns on the trading-day table (one row per session)."""
    base = base.sort_values("date").reset_index(drop=True)
    for h in HORIZONS:
        base[f"cum{h}"] = np.log(base["rate"].shift(-(h - 1)) / base["prev_rate"])
        base[f"del{h}"] = base["log_return"].shift(-(h - 1))
    for k in IC_LAGS:
        base[f"ret_k{k}"] = base["log_return"].shift(-(k - 1))
    return base


class HorizonData:
    """Duck-types FusionData.splits() with a horizon target and purged boundaries."""

    def __init__(self, data: FusionData, col: str, h: int):
        self.data, self.col, self.h = data, col, h
        self._cache: dict[float, dict[str, pd.DataFrame]] = {}

    def splits(self, halflife: float = 0) -> dict[str, pd.DataFrame]:
        if halflife not in self._cache:
            sp = {k: b.copy() for k, b in self.data.splits(halflife).items()}
            for name in ("train", "validation"):          # purge boundary overlap
                if self.h > 1:
                    sp[name] = sp[name].iloc[: -(self.h - 1)]
            for name in sp:
                sp[name] = sp[name].dropna(subset=[self.col])
            thr = sp["train"][self.col].abs().quantile(LARGE_MOVE_QUANTILE)
            for b in sp.values():
                b["direction"] = (b[self.col] > 0).astype(int)
                b["large_move"] = (b[self.col].abs() > thr).astype(int)
            self._cache[halflife] = sp
        return self._cache[halflife]


def run_models(datas: dict[str, FusionData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates = rate_features(datas[ENCODERS[0]].base)
    axes = [c for c in datas[ENCODERS[0]].base.columns if c.startswith("ax_")]
    metrics, boots = [], []
    configs = [("cum", 1)] + [(kind, h) for h in HORIZONS[1:] for kind in ("cum", "del")]
    for kind, h in configs:
        col = f"{kind}{h}"
        label = "next session" if h == 1 else f"{kind}{h}"
        for target in TARGETS:
            print(f"\n=== {label} / {target} ===")
            fits = {"A": best_of("A", HorizonData(datas[ENCODERS[0]], col, h), rates, target, (0,))}
            for enc in ENCODERS:
                fits[f"C-{enc}"] = best_of(f"C-{enc}", HorizonData(datas[enc], col, h),
                                           rates + axes, target, DECAY_HALFLIVES)
            best = max((k for k in fits if k != "A"), key=lambda k: fits[k].val_f1)

            preds, y, dates = {}, None, None
            for row, f in fits.items():
                for split in ("validation", "test"):
                    b = f.splits[split]
                    X = _X(b, f.cols)
                    pred, prob = f.pipe.predict(X), f.pipe.predict_proba(X)[:, 1]
                    metrics.append({"kind": kind, "h": h, "target": target, "row": row,
                                    "family": f.family, "decay_halflife": f.halflife,
                                    "split": split, "best_on_val": row == best,
                                    **directional_metrics(b[target], pred, prob)})
                    if split == "test":
                        if y is None:
                            y, dates = b[target].to_numpy(), b["date"].to_numpy()
                        assert (b["date"].to_numpy() == dates).all()
                        preds[row] = pred

            idx = block_bootstrap_indices(len(y), BOOTSTRAP_BLOCK, BOOTSTRAP_REPS, RANDOM_SEED)
            base_b = mcc_rows(y[idx], preds["A"][idx])
            base_full = mcc_rows(y[None], preds["A"][None])[0]
            for row, p in preds.items():
                full = mcc_rows(y[None], p[None])[0]
                diff = mcc_rows(y[idx], p[idx]) - base_b
                boots.append({"kind": kind, "h": h, "target": target, "row": row,
                              "best_on_val": row == best, "n_test": len(y),
                              "mcc_test": full, "d_mcc": full - base_full,
                              "ci_low": np.percentile(diff, 2.5),
                              "ci_high": np.percentile(diff, 97.5),
                              "p_gt0": float((diff > 0).mean())})
    return pd.DataFrame(metrics), pd.DataFrame(boots)


def information_coefficients(datas: dict[str, FusionData]) -> pd.DataFrame:
    """Spearman corr of each channel direction score with the return k sessions later."""
    rows = []
    for enc, data in datas.items():
        sp = data.splits(0)
        for split in ("train", "test"):
            b = sp[split]
            for ch in CHANNELS:
                for k in IC_LAGS:
                    pair = b[[f"ax_{ch}_dir", f"ret_k{k}"]].dropna()
                    rows.append({"encoder": enc, "split": split, "channel": ch, "k": k,
                                 "n": len(pair),
                                 "ic": pair.corr(method="spearman").iloc[0, 1]})
    return pd.DataFrame(rows)


def main() -> None:
    datas = {}
    for enc in ENCODERS:
        d = FusionData(enc)
        d.base = add_horizon_returns(d.base)
        datas[enc] = d

    ic = information_coefficients(datas)
    metrics, boots = run_models(datas)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ic.to_csv(RESULTS_DIR / "horizon_ic.csv", index=False, float_format="%.5f")
    metrics.to_csv(RESULTS_DIR / "horizon_metrics.csv", index=False)
    boots.to_csv(RESULTS_DIR / "horizon_bootstrap.csv", index=False)

    pd.set_option("display.width", 170)
    b = boots.copy()
    b["config"] = np.where(b.h == 1, "next", b.kind + b.h.astype(str))
    for target in TARGETS:
        t = b[b.target == target]
        print(f"\n{target}: test MCC of A, and dMCC vs A [95% CI] per encoder (* = picked on validation)")
        for cfg, g in t.groupby("config", sort=False):
            a = g[g.row == "A"].mcc_test.iloc[0]
            cells = []
            for _, r in g[g.row != "A"].iterrows():
                star = "*" if r.best_on_val else " "
                cells.append(f"{r.row[2:]:>8}{star}{r.d_mcc:+.3f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]")
            print(f"  {cfg:<6} A={a:+.3f}  " + "  ".join(cells))

    v = metrics.pivot_table(index=["kind", "h", "target", "row"], columns="split", values="mcc").reset_index()
    v = v[v.row != "A"]
    print("\nValidation-vs-test agreement of C rows (Pearson r of MCC across encoders and configs):")
    print(v.groupby("target")[["validation", "test"]].apply(lambda g: g.corr().iloc[0, 1]).round(2).to_string())

    print("\nModel-free: mean |Spearman IC| of the 7 channel direction scores, by lag k "
          "(train n~800: |IC| > 0.07 is p<0.05; test n~176: > 0.15)")
    m = ic.assign(abs_ic=ic.ic.abs()).pivot_table(index=["encoder", "split"], columns="k", values="abs_ic")
    print(m.round(3).to_string())
    agree = ic.pivot_table(index=["encoder", "channel", "k"], columns="split", values="ic").reset_index()
    agree["same_sign"] = np.sign(agree.train) == np.sign(agree.test)
    print("\nShare of channel x lag pairs where train and test IC have the same sign (0.5 = chance):")
    print(agree.groupby(["encoder", "k"]).same_sign.mean().unstack().round(2).to_string())
    print("\nWrote results/horizon_ic.csv, horizon_metrics.csv, horizon_bootstrap.csv")


if __name__ == "__main__":
    main()
