# src/fusion.py
"""
Task 3, step 3 -- early fusion and gated late fusion of rate and text features.

Combines the Task 2 rate features (X_time) with the per-encoder session text
features from step 2 (X_text: channel axes, novelty, concentration and, for
FinBERT, its own sentiment) in two ways, for each of the four encoders and
for both prediction targets.

TARGETS
-------
direction   1 if the next JISDOR fixing is above the previous one (Task 2).
large_move  1 if |log return| exceeds the 75th percentile of TRAINING-block
            absolute returns. News is expected to predict magnitude better
            than sign. ~25% positives by construction, so models for this
            target use balanced class weights.

EARLY FUSION
------------
Concatenate [X_time ; X_text] and fit logistic regression and XGBoost (the
Task 3 plan named LightGBM; XGBoost is used because it is already the Task 2
tree model and is installed). The grid crosses each model's own
hyperparameters with the decay half-life of the channel scores
(DECAY_HALFLIVES); the pair with the best VALIDATION macro-F1 wins.

GATED LATE FUSION
-----------------
Fit a rates-only logistic model and a text-only logistic model separately on
the training block (each tuned on validation). Then fit a small meta-model on
their VALIDATION logits:

    p = sigmoid(w0 + a * z_rates + b * z_text + c * g * z_text)

where g is the session's semantic novelty (GATE_FEATURE), standardised with
training-block mean and standard deviation. c > 0 means the text model is
trusted more on days whose news is new. An ungated version (c fixed at 0) is
fitted alongside so the gate's own contribution can be read off.

Because the meta-model is fitted on validation, its validation metrics are
in-sample and optimistic. Those rows carry val_in_sample = True. Only its
test metrics are a fair comparison.

REFERENCE ROWS
--------------
For each target the script also fits majority, Model A (rates only) and
Model B (rates + Task 2 lexicon features), so that the large-move target has
its own baselines. Task 2's results/metrics.csv is left untouched.

LEAKAGE DISCIPLINE
------------------
  * Imputers, scalers, the large-move threshold and the gate's standardisation
    are all fitted on the training block.
  * The decay is a causal exponential moving average (past and present
    sessions only), so it can be applied to the whole series before splitting.
  * Hyperparameters and the decay half-life are chosen on validation; the
    meta-model is fitted on validation; the test block is scored once.

OUTPUTS (results/)
------------------
fusion_metrics.csv           one row per target x model x split
fusion_selected.csv          chosen hyperparameters and decay per model
fusion_gate.csv              meta-model coefficients (a, b, c) per encoder
fusion_test_predictions.csv  long format: date, target, model, actual, pred, prob

Usage:
    python src/fusion.py                              # all encoders, both targets
    python src/fusion.py --encoders sbert finbert
    python src/fusion.py --targets direction
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from config import (
        DECAY_HALFLIVES,
        FUSION_GATE_CSV,
        FUSION_METRICS_CSV,
        FUSION_PREDICTIONS_CSV,
        FUSION_SELECTED_CSV,
        FUSION_TARGETS,
        GATE_FEATURE,
        LARGE_MOVE_QUANTILE,
        META_C,
        RANDOM_SEED,
        RESULTS_DIR,
    )
    from dataset import load_daily, split_chronologically
    from embeddings import ENCODERS
    from evaluate import directional_metrics
    from models import news_features, numeric_pipeline, rate_features
    from semantic_axes import apply_decay, load_text_features
except ImportError:  # pragma: no cover
    from src.config import (
        DECAY_HALFLIVES,
        FUSION_GATE_CSV,
        FUSION_METRICS_CSV,
        FUSION_PREDICTIONS_CSV,
        FUSION_SELECTED_CSV,
        FUSION_TARGETS,
        GATE_FEATURE,
        LARGE_MOVE_QUANTILE,
        META_C,
        RANDOM_SEED,
        RESULTS_DIR,
    )
    from src.dataset import load_daily, split_chronologically
    from src.embeddings import ENCODERS
    from src.evaluate import directional_metrics
    from src.models import news_features, numeric_pipeline, rate_features
    from src.semantic_axes import apply_decay, load_text_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Same grids as Task 2 (src/models.py), so differences come from the features.
LOGREG_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
XGB_GRID = [
    {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03},
    {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.1},
]

NOVELTY_COLS = ("nov_headline", "nov_centroid", "concentration")
# Session scores that are smoothed over time by the decay half-life. Novelty
# already carries its own memory, so it is left as is.
DECAY_PREFIXES = ("ax_", "fbs_")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def text_columns(df: pd.DataFrame, novelty: bool = True,
                 sentiment: bool = True) -> list[str]:
    """Step 2 text features present in `df`, in stable order.

    novelty=False drops novelty and concentration (ablation C1-C4 vs C5);
    sentiment=False drops FinBERT's own sentiment columns.
    """
    cols = [c for c in df.columns if c.startswith("ax_")]
    if novelty:
        cols += [c for c in NOVELTY_COLS if c in df.columns]
    if sentiment:
        cols += [c for c in df.columns if c.startswith("fbs_")]
    return cols


class FusionData:
    """Daily table + one encoder's text features, split chronologically.

    `splits(halflife)` returns the train / validation / test blocks with the
    channel scores decayed at that half-life (cached), plus both targets.
    Rows are identical across half-lives; only text feature values change.
    """

    def __init__(self, encoder: str):
        self.encoder = encoder
        daily = load_daily()
        text = load_text_features(encoder)
        # LEFT join on the trading calendar: a session with no news keeps its
        # row, with NaN text features that the train-fitted imputer fills.
        self.base = daily.merge(text, on="date", how="left")
        self.decay_cols = [c for c in text.columns if c.startswith(DECAY_PREFIXES)]
        self._cache: dict[float, dict[str, pd.DataFrame]] = {}
        self.large_move_threshold: float | None = None

    def splits(self, halflife: float = 0) -> dict[str, pd.DataFrame]:
        if halflife not in self._cache:
            df = apply_decay(self.base, self.decay_cols, halflife).reset_index(drop=True)
            with contextlib.redirect_stdout(io.StringIO()):   # silence warm-up notice
                sp = split_chronologically(df)
            self.large_move_threshold = add_large_move(sp)
            self._cache[halflife] = sp
        return self._cache[halflife]


def add_large_move(splits: dict[str, pd.DataFrame]) -> float:
    """Add the `large_move` target; the threshold comes from training returns."""
    thr = float(splits["train"]["log_return"].abs().quantile(LARGE_MOVE_QUANTILE))
    for block in splits.values():
        block["large_move"] = (block["log_return"].abs() > thr).astype(int)
    return thr


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------

def build(family: str, params: dict, y_train: pd.Series, balanced: bool):
    """A train-fitted-preprocessing pipeline for one model family."""
    if family == "logreg":
        return numeric_pipeline(LogisticRegression(
            C=params["C"], max_iter=3000, random_state=RANDOM_SEED,
            class_weight="balanced" if balanced else None))
    if family == "xgb":
        from xgboost import XGBClassifier
        pos = float(y_train.mean())
        return numeric_pipeline(
            XGBClassifier(
                **params,
                scale_pos_weight=(1 - pos) / pos if balanced else 1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=4,
            ),
            scale=False,   # trees are scale-invariant
        )
    raise ValueError(family)


def _X(block: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return block[cols].to_numpy(dtype=float)


@dataclass
class Fitted:
    """A model fitted on train, with the choice that won on validation."""
    name: str
    target: str
    family: str
    cols: list[str]
    halflife: float
    params: dict
    val_f1: float
    pipe: object
    data: FusionData = field(repr=False)

    @property
    def splits(self) -> dict[str, pd.DataFrame]:
        return self.data.splits(self.halflife)


def select_and_fit(name: str, data: FusionData, cols: list[str], target: str,
                   family: str, halflives=(0,)) -> Fitted:
    """Grid over (half-life x hyperparameters); keep the best validation macro-F1."""
    grid = LOGREG_GRID if family == "logreg" else XGB_GRID
    balanced = target == "large_move"
    best: Fitted | None = None
    for hl in halflives:
        sp = data.splits(hl)
        tr, va = sp["train"], sp["validation"]
        for params in grid:
            pipe = build(family, params, tr[target], balanced)
            pipe.fit(_X(tr, cols), tr[target].astype(int))
            f1 = directional_metrics(va[target], pipe.predict(_X(va, cols)))["f1_macro"]
            if best is None or f1 > best.val_f1:
                best = Fitted(name, target, family, cols, hl, params, f1, pipe, data)
    print(f"  {name:<28} half-life {best.halflife:<3} {json.dumps(best.params):<58} "
          f"val macro-F1 {best.val_f1:.4f}")
    return best


# --------------------------------------------------------------------------
# Late fusion
# --------------------------------------------------------------------------

@dataclass
class LateFusion:
    name: str
    target: str
    gated: bool
    meta: LogisticRegression
    coefs: dict


def _gate(text: Fitted) -> dict[str, np.ndarray]:
    """Standardised novelty per split, using training mean and std. Missing -> 0."""
    sp = text.splits
    tr = sp["train"][GATE_FEATURE]
    mu, sd = tr.mean(), tr.std()
    return {k: ((b[GATE_FEATURE] - mu) / sd).fillna(0.0).to_numpy() for k, b in sp.items()}


def _meta_inputs(rates: Fitted, text: Fitted, split: str, gated: bool) -> np.ndarray:
    rb, tb = rates.splits[split], text.splits[split]
    assert (rb["date"].to_numpy() == tb["date"].to_numpy()).all(), "row mismatch"
    z_r = rates.pipe.decision_function(_X(rb, rates.cols))
    z_t = text.pipe.decision_function(_X(tb, text.cols))
    cols = [z_r, z_t]
    if gated:
        cols.append(_gate(text)[split] * z_t)
    return np.column_stack(cols)


def fit_late(name: str, rates: Fitted, text: Fitted, gated: bool) -> LateFusion:
    """Meta-model on VALIDATION logits of the two separately trained models."""
    target = rates.target
    y_val = text.splits["validation"][target].astype(int)
    meta = LogisticRegression(C=META_C, max_iter=1000,
                              class_weight="balanced" if target == "large_move" else None)
    meta.fit(_meta_inputs(rates, text, "validation", gated), y_val)
    w = meta.coef_[0]
    coefs = {"intercept": float(meta.intercept_[0]), "a_rates": float(w[0]),
             "b_text": float(w[1]), "c_gate": float(w[2]) if gated else 0.0}
    print(f"  {name:<28} a={coefs['a_rates']:+.3f}  b={coefs['b_text']:+.3f}  "
          f"c={coefs['c_gate']:+.3f}")
    return LateFusion(name, target, gated, meta, coefs)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.metrics: list[dict] = []
        self.selected: list[dict] = []
        self.gates: list[dict] = []
        self.preds: list[pd.DataFrame] = []

    def _add(self, target, model, family, encoder, split, y, pred, prob,
             in_sample=False):
        self.metrics.append({"target": target, "model": model, "family": family,
                             "encoder": encoder, "split": split,
                             "val_in_sample": in_sample,
                             **directional_metrics(y, pred, prob)})

    def _add_preds(self, target, model, block, pred, prob):
        self.preds.append(pd.DataFrame({
            "date": block["date"].to_numpy(), "target": target, "model": model,
            "actual": block[target].astype(int).to_numpy(),
            "pred": np.asarray(pred, dtype=int), "prob": np.asarray(prob, dtype=float)}))

    def fitted(self, f: Fitted, family: str, encoder: str):
        for split, block in f.splits.items():
            X = _X(block, f.cols)
            pred, prob = f.pipe.predict(X), f.pipe.predict_proba(X)[:, 1]
            self._add(f.target, f.name, family, encoder, split, block[f.target], pred, prob)
            if split == "test":
                self._add_preds(f.target, f.name, block, pred, prob)
        self.selected.append({"target": f.target, "model": f.name, "encoder": encoder,
                              "n_features": len(f.cols), "decay_halflife": f.halflife,
                              "params": json.dumps(f.params), "val_f1_macro": f.val_f1})

    def late(self, lf: LateFusion, rates: Fitted, text: Fitted, encoder: str):
        for split in ("validation", "test"):
            block = text.splits[split]
            M = _meta_inputs(rates, text, split, lf.gated)
            pred, prob = lf.meta.predict(M), lf.meta.predict_proba(M)[:, 1]
            self._add(lf.target, lf.name, "late", encoder, split, block[lf.target],
                      pred, prob, in_sample=(split == "validation"))
            if split == "test":
                self._add_preds(lf.target, lf.name, block, pred, prob)
        self.gates.append({"target": lf.target, "model": lf.name, "encoder": encoder,
                           "gated": lf.gated, "text_decay_halflife": text.halflife,
                           "text_params": json.dumps(text.params), **lf.coefs})

    def majority(self, data: FusionData, target: str):
        sp = data.splits(0)
        cls = int(sp["train"][target].mode().iloc[0])
        for split, block in sp.items():
            pred = np.full(len(block), cls)
            self._add(target, "majority", "baseline", "-", split, block[target], pred, None)
            if split == "test":
                self._add_preds(target, "majority", block, pred, pred.astype(float))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(encoders=ENCODERS, targets=FUSION_TARGETS) -> pd.DataFrame:
    rec = Recorder()
    datas = {enc: FusionData(enc) for enc in encoders}
    ref = datas[encoders[0]]          # rate / lexicon columns are encoder-independent
    rate_cols = rate_features(ref.base)
    lex_cols = news_features(ref.base)
    sp0 = ref.splits(0)
    print(f"Split sizes: train {len(sp0['train'])}, val {len(sp0['validation'])}, "
          f"test {len(sp0['test'])}")
    print(f"Large-move threshold |log return| > {ref.large_move_threshold:.5f} "
          f"(training {LARGE_MOVE_QUANTILE:.0%} quantile)")
    print(f"Feature groups: {len(rate_cols)} rate, {len(lex_cols)} lexicon")

    for target in targets:
        print(f"\n=== target: {target} ===")
        rec.majority(ref, target)

        print("[reference] Model A (rates only) and Model B (rates + lexicons)")
        a_lr = select_and_fit("A_rates_logreg", ref, rate_cols, target, "logreg")
        rec.fitted(a_lr, "baseline", "-")
        rec.fitted(select_and_fit("A_rates_xgb", ref, rate_cols, target, "xgb"), "baseline", "-")
        rec.fitted(select_and_fit("B_lexicon_logreg", ref, rate_cols + lex_cols, target,
                                  "logreg"), "baseline", "-")
        rec.fitted(select_and_fit("B_lexicon_xgb", ref, rate_cols + lex_cols, target,
                                  "xgb"), "baseline", "-")

        for enc in encoders:
            data = datas[enc]
            txt = text_columns(data.base)
            print(f"[{enc}] {len(txt)} text features")

            # Early fusion: one model over [X_time ; X_text].
            for fam in ("logreg", "xgb"):
                f = select_and_fit(f"early_{fam}_{enc}", data, rate_cols + txt,
                                   target, fam, DECAY_HALFLIVES)
                rec.fitted(f, "early", enc)

            # Late fusion: rates model (= Model A logistic, fitted on the same
            # rows) + a text-only model, combined on validation logits.
            a_on_enc = Fitted(a_lr.name, target, "logreg", rate_cols, 0, a_lr.params,
                              a_lr.val_f1, a_lr.pipe, data)
            text_only = select_and_fit(f"text_only_logreg_{enc}", data, txt, target,
                                       "logreg", DECAY_HALFLIVES)
            rec.fitted(text_only, "text_only", enc)
            for gated in (False, True):
                name = f"late_{'gated' if gated else 'stack'}_{enc}"
                lf = fit_late(name, a_on_enc, text_only, gated)
                rec.late(lf, a_on_enc, text_only, enc)

    metrics = pd.DataFrame(rec.metrics)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(FUSION_METRICS_CSV, index=False)
    pd.DataFrame(rec.selected).to_csv(FUSION_SELECTED_CSV, index=False)
    pd.DataFrame(rec.gates).to_csv(FUSION_GATE_CSV, index=False)
    pd.concat(rec.preds, ignore_index=True).to_csv(FUSION_PREDICTIONS_CSV, index=False)

    show = ["accuracy", "f1_macro", "mcc", "roc_auc"]
    pd.set_option("display.width", 140)
    for target in targets:
        for split in ("validation", "test"):
            t = metrics[(metrics["target"] == target) & (metrics["split"] == split)]
            t = t.set_index("model")[show + ["val_in_sample"]]
            print(f"\n{target} -- {split}"
                  + (" (late rows: meta-model fitted here, in-sample)" if split == "validation" else ""))
            print(t.sort_values("mcc", ascending=False).round(3).to_string())

    for p in (FUSION_METRICS_CSV, FUSION_SELECTED_CSV, FUSION_GATE_CSV, FUSION_PREDICTIONS_CSV):
        print(f"Wrote {p}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    parser.add_argument("--targets", nargs="+", choices=FUSION_TARGETS,
                        default=list(FUSION_TARGETS))
    args = parser.parse_args()
    run(tuple(args.encoders), tuple(args.targets))


if __name__ == "__main__":
    main()
