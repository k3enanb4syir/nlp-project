# src/analysis.py
"""
Task 3, step 5 -- feature analysis for notebook/04_feature_analysis.ipynb.

Four analyses, each written to a small CSV in results/ so that the notebook
needs neither the embedding cache nor any model fitting.

1. EMBEDDING MAP
   Headlines (SBERT, centred as in semantic_axes.Space) projected onto the
   first two principal components of TRAINING headlines, coloured by whether
   their session was a large-move day. Each channel is drawn as an arrow that
   starts at its topic vector and points along its direction axis (towards the
   USD/IDR-up pole). The plan named UMAP; a linear projection is used instead
   because only a linear map sends an axis to an arrow, and umap-learn is not
   a dependency. Each arrow also records how much of the axis lies in the
   plane (in_plane_share): a short arrow means the axis is mostly orthogonal to
   the main topics in the corpus.

2. SHAP + HEADLINE RETRIEVAL
   The C5 model of each target is refitted exactly as src/ablation.py fits it
   (same encoder, family, grid and seed; the test MCC is checked against
   results/ablation_metrics.csv). Per-feature contributions on the test block:
   exact TreeSHAP from XGBoost (pred_contribs=True) for a tree model, or
   coef * (x - training mean) for a logistic one. For the test days where the
   text features moved the log-odds most, the responsible channel is the axis
   feature with the largest |SHAP|, and the retrieved headlines are that day's
   AXIS_TOP_K most relevant to the channel, i.e. the headlines the pooling
   actually used.

3. EVENT FACE-VALIDITY CHECK (no models, no returns)
   For each event in ANALYSIS_EVENTS and each encoder, every channel's
   attention (ax_<ch>_attn) and direction (ax_<ch>_dir) over the sessions
   after the event, as z-scores against the training distribution. The
   expected channel should show the largest attention spike.

4. ERROR TABLE
   Test days split by which of Model A and C5 (from the ablation predictions)
   was right, with each day's return size, news volume, novelty and its most
   FX-relevant headlines (highest relevance to any channel).

Usage:
    python src/analysis.py        # needs data/embeddings/ (gitignored), ~1 min
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ablation import best_of, load_data
    from anchors import CHANNELS
    from config import (
        ABLATION_METRICS_CSV,
        ABLATION_PREDICTIONS_CSV,
        ANALYSIS_ERROR_HEADLINES,
        ANALYSIS_ERRORS_CSV,
        ANALYSIS_EVENT_SESSIONS,
        ANALYSIS_EVENTS,
        ANALYSIS_EVENTS_CSV,
        ANALYSIS_MAP_AXES_CSV,
        ANALYSIS_MAP_CSV,
        ANALYSIS_MAP_ENCODER,
        ANALYSIS_MAP_SAMPLE,
        ANALYSIS_SHAP_CSV,
        ANALYSIS_SHIFT_HEADLINES_CSV,
        ANALYSIS_SHIFTS_CSV,
        ANALYSIS_TOP_SHIFTS,
        AXIS_TOP_K,
        FUSION_TARGETS,
        RANDOM_SEED,
        RESULTS_DIR,
        TRAIN_END,
    )
    from embeddings import ENCODERS, load_articles, train_mask
    from evaluate import directional_metrics
    from fusion import NOVELTY_COLS, FusionData, Fitted, _X
    from models import rate_features
    from semantic_axes import Space, headline_scores, load_text_features
except ImportError:  # pragma: no cover
    from src.ablation import best_of, load_data
    from src.anchors import CHANNELS
    from src.config import (
        ABLATION_METRICS_CSV,
        ABLATION_PREDICTIONS_CSV,
        ANALYSIS_ERROR_HEADLINES,
        ANALYSIS_ERRORS_CSV,
        ANALYSIS_EVENT_SESSIONS,
        ANALYSIS_EVENTS,
        ANALYSIS_EVENTS_CSV,
        ANALYSIS_MAP_AXES_CSV,
        ANALYSIS_MAP_CSV,
        ANALYSIS_MAP_ENCODER,
        ANALYSIS_MAP_SAMPLE,
        ANALYSIS_SHAP_CSV,
        ANALYSIS_SHIFT_HEADLINES_CSV,
        ANALYSIS_SHIFTS_CSV,
        ANALYSIS_TOP_SHIFTS,
        AXIS_TOP_K,
        FUSION_TARGETS,
        RANDOM_SEED,
        RESULTS_DIR,
        TRAIN_END,
    )
    from src.embeddings import ENCODERS, load_articles, train_mask
    from src.evaluate import directional_metrics
    from src.fusion import NOVELTY_COLS, FusionData, Fitted, _X
    from src.models import rate_features
    from src.semantic_axes import Space, headline_scores, load_text_features


class Spaces:
    """Lazily built embedding spaces with their per-headline channel scores."""

    def __init__(self, articles: pd.DataFrame):
        self.articles = articles
        self._spaces: dict[str, Space] = {}
        self._scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.by_day = articles.groupby("trading_day").indices

    def space(self, enc: str) -> Space:
        if enc not in self._spaces:
            print(f"  building {enc} space ...")
            self._spaces[enc] = Space(enc, self.articles)
        return self._spaces[enc]

    def scores(self, enc: str) -> tuple[np.ndarray, np.ndarray]:
        """(relevance, direction), each n_articles x n_channels."""
        if enc not in self._scores:
            self._scores[enc] = headline_scores(self.space(enc))
        return self._scores[enc]

    def day(self, date) -> np.ndarray:
        return np.asarray(self.by_day.get(pd.Timestamp(date), []), dtype=int)


def session_flags() -> pd.DataFrame:
    """date, split and large_move for every modelled session."""
    sp = FusionData(ENCODERS[0]).splits(0)
    return pd.concat([b[["date", "large_move"]].assign(split=k) for k, b in sp.items()])


# --------------------------------------------------------------------------
# 1. Embedding map
# --------------------------------------------------------------------------

def embedding_map(spaces: Spaces) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA

    print("[1/4] embedding map")
    arts = spaces.articles
    space = spaces.space(ANALYSIS_MAP_ENCODER)
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(space.U[train_mask(arts)])

    flags = session_flags()
    keep = np.flatnonzero(arts["trading_day"].isin(flags["date"]).to_numpy())
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.sort(rng.choice(keep, size=min(ANALYSIS_MAP_SAMPLE, len(keep)), replace=False))
    xy = pca.transform(space.U[idx])
    pts = pd.DataFrame({"date": arts["trading_day"].to_numpy()[idx],
                        "title": arts["title_clean"].to_numpy()[idx],
                        "x": xy[:, 0], "y": xy[:, 1]})
    pts = pts.merge(flags, on="date", how="left")

    rows = []
    for ch in CHANNELS:
        start = pca.transform(space.topic(ch)[None])[0]
        arrow = space.direction(ch) @ pca.components_.T
        rows.append({"channel": ch, "x0": start[0], "y0": start[1],
                     "dx": arrow[0], "dy": arrow[1],
                     "in_plane_share": float(np.linalg.norm(arrow))})
    axes = pd.DataFrame(rows)
    print(f"  {len(pts)} headlines; PC1+PC2 explain "
          f"{pca.explained_variance_ratio_.sum():.1%} of training variance")
    return pts, axes


# --------------------------------------------------------------------------
# 2. SHAP + retrieval
# --------------------------------------------------------------------------

def refit_c5(target: str) -> tuple[Fitted, str]:
    """The ablation's C5 model for `target`, refitted and checked."""
    m = pd.read_csv(ABLATION_METRICS_CSV)
    ref = m[(m.target == target) & (m.row == "C5") & (m.split == "test")].iloc[0]
    enc = ref["encoder"]
    data = load_data(enc)
    axes = [c for c in data.base.columns if c.startswith("ax_")]
    cols = rate_features(data.base) + axes + list(NOVELTY_COLS)
    f = best_of("C5", data, cols, target)
    test = f.splits["test"]
    mcc = directional_metrics(test[target], f.pipe.predict(_X(test, f.cols)))["mcc"]
    assert f.family == ref["family"] and abs(mcc - ref["mcc"]) < 1e-9, (
        f"refit of C5/{target} does not match the ablation "
        f"({f.family} MCC {mcc:.4f} vs {ref['family']} {ref['mcc']:.4f}); "
        "rerun src/ablation.py first")
    return f, enc


def contributions(f: Fitted, block: pd.DataFrame) -> np.ndarray:
    """Per-feature log-odds contributions (n x n_features), bias column dropped."""
    Z = f.pipe[:-1].transform(_X(block, f.cols))
    model = f.pipe[-1]
    if f.family == "xgb":
        import xgboost as xgb
        return model.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
    Ztr = f.pipe[:-1].transform(_X(f.splits["train"], f.cols))
    return (Z - Ztr.mean(axis=0)) * model.coef_[0]


def feature_group(col: str) -> str:
    if col.endswith("_dir") and col.startswith("ax_"):
        return "channel direction"
    if col.endswith("_attn") and col.startswith("ax_"):
        return "channel attention"
    if col in NOVELTY_COLS:
        return "novelty"
    return "rates"


def shap_analysis(target: str, spaces: Spaces):
    print(f"[2/4] SHAP + retrieval ({target})")
    f, enc = refit_c5(target)
    test = f.splits["test"].reset_index(drop=True)
    phi = contributions(f, test)
    cols = np.array(f.cols)

    imp = pd.DataFrame({"target": target, "encoder": enc, "family": f.family,
                        "feature": cols, "group": [feature_group(c) for c in cols],
                        "mean_abs_shap": np.abs(phi).mean(axis=0)})

    is_text = np.array([feature_group(c) != "rates" for c in cols])
    is_axis = np.array([c.startswith("ax_") for c in cols])
    text_shift = phi[:, is_text].sum(axis=1)
    prob = f.pipe.predict_proba(_X(test, f.cols))[:, 1]
    rel, drn = spaces.scores(enc)

    shifts, heads = [], []
    for i in np.argsort(-np.abs(text_shift))[:ANALYSIS_TOP_SHIFTS]:
        j = np.flatnonzero(is_axis)[np.argmax(np.abs(phi[i, is_axis]))]
        feat = cols[j]
        ch = feat.split("_")[1]
        date = test.loc[i, "date"]
        shifts.append({"target": target, "encoder": enc, "date": date,
                       "actual": int(test.loc[i, target]), "prob": prob[i],
                       "pred": int(prob[i] >= 0.5), "text_shift": text_shift[i],
                       "rate_shift": phi[i, ~is_text].sum(), "channel": ch,
                       "top_axis_feature": feat, "top_axis_shap": phi[i, j],
                       "feature_value": test.loc[i, feat]})
        day = spaces.day(date)
        c = CHANNELS.index(ch)
        for rank, a in enumerate(day[np.argsort(-rel[day, c])][:AXIS_TOP_K], start=1):
            heads.append({"target": target, "date": date, "channel": ch, "rank": rank,
                          "title": spaces.articles.at[a, "title_clean"],
                          "relevance": rel[a, c], "direction": drn[a, c]})
    return imp, pd.DataFrame(shifts), pd.DataFrame(heads)


# --------------------------------------------------------------------------
# 3. Event check
# --------------------------------------------------------------------------

def event_check() -> pd.DataFrame:
    print("[3/4] event face-validity check")
    rows = []
    for enc in ENCODERS:
        tf = load_text_features(enc).sort_values("date").reset_index(drop=True)
        tr = tf[tf["date"] <= pd.Timestamp(TRAIN_END)]
        for name, day, expected in ANALYSIS_EVENTS:
            win = tf[tf["date"] > pd.Timestamp(day)].head(ANALYSIS_EVENT_SESSIONS)
            for ch in CHANNELS:
                a, d = f"ax_{ch}_attn", f"ax_{ch}_dir"
                za = (win[a] - tr[a].mean()) / tr[a].std()
                zd = (win[d] - tr[d].mean()) / tr[d].std()
                rows.append({"event": name, "event_date": day, "expected": expected,
                             "encoder": enc, "channel": ch,
                             "window": f"{win['date'].min().date()}..{win['date'].max().date()}",
                             "attn_z_max": za.max(), "dir_z_mean": zd.mean()})
    ev = pd.DataFrame(rows)
    ev["attn_rank"] = (ev.groupby(["event", "encoder"])["attn_z_max"]
                       .rank(ascending=False, method="min").astype(int))
    return ev


# --------------------------------------------------------------------------
# 4. Error table
# --------------------------------------------------------------------------

def error_table(spaces: Spaces) -> pd.DataFrame:
    print("[4/4] error table")
    preds = pd.read_csv(ABLATION_PREDICTIONS_CSV, parse_dates=["date"])
    m = pd.read_csv(ABLATION_METRICS_CSV)
    daily = FusionData(ENCODERS[0]).base[["date", "log_return", "n_articles"]]
    out = []
    for target in FUSION_TARGETS:
        enc = m[(m.target == target) & (m.row == "C5")]["encoder"].iloc[0]
        p = preds[preds.target == target]
        a = p[p.row == "A"].set_index("date")
        c = p[p.row == "C5"].set_index("date")
        t = pd.DataFrame({"actual": a["actual"], "pred_A": a["pred"], "pred_C5": c["pred"]})
        ra, rc = t.pred_A == t.actual, t.pred_C5 == t.actual
        t["group"] = np.select([rc & ~ra, ra & ~rc, ra & rc],
                               ["C5 right, A wrong", "A right, C5 wrong", "both right"],
                               "both wrong")
        t = t.reset_index()
        tf = load_text_features(enc)[["date", "nov_centroid", "concentration"]]
        t = t.merge(daily, on="date", how="left").merge(tf, on="date", how="left")
        t["abs_return"] = t.pop("log_return").abs()

        rel, _ = spaces.scores(enc)
        best_rel = rel.max(axis=1)
        tops = []
        for d in t["date"]:
            day = spaces.day(d)
            pick = day[np.argsort(-best_rel[day])][:ANALYSIS_ERROR_HEADLINES]
            tops.append(" | ".join(spaces.articles["title_clean"].to_numpy()[pick]))
        t["top_headlines"] = tops
        t.insert(0, "encoder", enc)
        t.insert(0, "target", target)
        out.append(t)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    spaces = Spaces(load_articles())

    pts, axes = embedding_map(spaces)
    pts.to_csv(ANALYSIS_MAP_CSV, index=False, float_format="%.5f")
    axes.to_csv(ANALYSIS_MAP_AXES_CSV, index=False, float_format="%.5f")

    imps, shifts, heads = [], [], []
    for target in FUSION_TARGETS:
        i, s, h = shap_analysis(target, spaces)
        imps.append(i), shifts.append(s), heads.append(h)
    pd.concat(imps).to_csv(ANALYSIS_SHAP_CSV, index=False, float_format="%.6f")
    pd.concat(shifts).to_csv(ANALYSIS_SHIFTS_CSV, index=False, float_format="%.6f")
    pd.concat(heads).to_csv(ANALYSIS_SHIFT_HEADLINES_CSV, index=False, float_format="%.4f")

    ev = event_check()
    ev.to_csv(ANALYSIS_EVENTS_CSV, index=False, float_format="%.3f")
    errs = error_table(spaces)
    errs.to_csv(ANALYSIS_ERRORS_CSV, index=False, float_format="%.6f")

    pd.set_option("display.width", 160)
    print("\nEvent check: expected channel's attention spike (z vs training) and its rank among 7")
    exp = ev[ev.channel == ev.expected]
    print(exp.pivot(index="event", columns="encoder", values="attn_z_max").round(2).to_string())
    print(exp.pivot(index="event", columns="encoder", values="attn_rank").to_string())
    print("\nSHAP: share of mean |SHAP| by feature group (test block)")
    imp = pd.concat(imps)
    g = imp.groupby(["target", "group"])["mean_abs_shap"].sum()
    print((g / g.groupby(level=0).transform("sum")).round(3).unstack().to_string())
    print("\nError groups (test days)")
    print(errs.groupby(["target", "group"]).agg(
        days=("date", "size"), abs_return=("abs_return", "mean"),
        n_articles=("n_articles", "mean"), nov_centroid=("nov_centroid", "mean"))
        .round(4).to_string())
    for p in (ANALYSIS_MAP_CSV, ANALYSIS_MAP_AXES_CSV, ANALYSIS_SHAP_CSV, ANALYSIS_SHIFTS_CSV,
              ANALYSIS_SHIFT_HEADLINES_CSV, ANALYSIS_EVENTS_CSV, ANALYSIS_ERRORS_CSV):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
