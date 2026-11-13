# src/semantic_axes.py
"""
Task 3, step 2 -- FX channel axes, news novelty and daily pooling.

Turns the cached headline embeddings (src/embeddings.py) into a small set of
interpretable, per-session text features, computed identically for each of the
four encoders so that the encoder is the only thing that varies in the
ablation.

TOPIC AND DIRECTION
-------------------
The anchors (src/anchors.py) are written as minimal pairs: positive anchor i
and negative anchor i describe the same kind of event and differ only in
direction. That gives each channel two vectors for free:

    topic     = mean(all 20 anchors)                  WHAT the news is about
    direction = mean(pos anchors) - mean(neg anchors) WHICH WAY it points

Shared content survives in the sum and cancels in the difference. Using the
direction alone does not work: on its own it responds to ANY up/down wording.
In our first version, which averaged direction over all of a day's headlines,
the most dollar-bullish training headline was "UPS stock jumps after big
profit" and the most dollar-bearish was "Bitcoin falls to lowest this month",
and FinBERT's seven channels correlated at |r| ~ 0.9 with each other. The
topic vector decides which headlines are allowed to speak for a channel; the
direction vector reads what they say.

KNOWN FAILURE MODES (visible in results/axes_top_headlines.txt)
  * Currency-pair direction: "EUR/USD climbs" is a WEAKER dollar, but an
    embedding sees "dollar" + "climbs". Embeddings do not parse which currency
    is the grammatical subject.
  * Negation: "ceasefire talks end with no breakthrough" can score as
    de-escalation.

PIPELINE (per encoder)
----------------------
  1. Centre.   Subtract the TRAINING-PERIOD mean embedding from every headline
               and anchor, then L2-normalise. Contextual encoders are
               anisotropic (two unrelated FinBERT headlines have raw cosine
               ~0.50), which compresses every cosine into a narrow band.
               Removing the common mean direction is the standard fix and is
               applied uniformly to all encoders.
  2. Score.    For every headline and channel:
                   relevance = cos(headline, topic)
                   direction = cos(headline, direction axis)   (+ = USD/IDR up)
  3. Select.   Per session and channel, keep the AXIS_TOP_K (=5) headlines
               with the highest relevance: hard top-k attention over the
               day's headline stream.
  4. Pool.     ax_<ch>_dir   mean direction of the selected headlines
               ax_<ch>_attn  mean relevance of the selected headlines, i.e.
                             how much the day's news is about this channel
  5. Novelty.  Markets price in what they already know, so we also measure
               how NEW the session's news is (past sessions only):
                 nov_headline   mean over headlines of 1 - max cosine to any
                                headline from the previous 5 sessions
                 nov_centroid   1 - cos(session centroid, exponentially decayed
                                memory of previous session centroids)
                 concentration  mean pairwise cosine among the session's
                                headlines (one dominant story vs many)
  6. FinBERT sentiment (finbert file only): session means of FinBERT's own
     positive / negative / neutral probabilities and net = pos - neg. This is
     the "obvious" transformer feature, kept as a comparison point.

7 channels x 2 + 3 novelty = 17 features per encoder (+4 FinBERT sentiment).
Temporal decay across sessions is NOT applied here: its half-life is a
hyperparameter, chosen on validation in the modelling step via apply_decay().

DIAGNOSTICS (no returns are used anywhere in this module)
--------------------------------------------------------
  * Leave-one-pair-out anchor accuracy: drop anchor pair i from BOTH poles,
    rebuild the axis, and check that pos i and neg i land on their own sides.
    (Dropping only one anchor is biased against minimal pairs: its twin stays
    in the opposite pole and pulls it across.)
  * Split-half reliability: build topic + direction from 5 random pairs and
    again from the other 5, run the full select-and-pool pipeline with each,
    and correlate the two series of session scores. High correlation means the
    feature reflects the channel, not the wording of particular anchors.
  * Inter-channel correlation of session direction scores. Reliability alone
    is not validity: an encoder can score every channel consistently while
    measuring one shared thing (e.g. sentiment) on all of them.
  * Top / bottom direction scores among the most channel-relevant training
    headlines, for a face-validity read.
Results go to results/axes_diagnostics.csv and results/axes_top_headlines.txt.

Usage:
    python src/semantic_axes.py                     # all encoders
    python src/semantic_axes.py --encoders sbert
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from anchors import ANCHORS, CHANNELS, anchors_hash
    from config import (
        AXIS_TOP_K,
        EMBED_DIR,
        NOVELTY_LOOKBACK_SESSIONS,
        NOVELTY_MEMORY_HALFLIFE,
        RANDOM_SEED,
        RESULTS_DIR,
        SPLIT_HALF_REPEATS,
        TEXT_FEATURES_DIR,
    )
    from embeddings import (
        ENCODERS,
        get_encoder,
        load_articles,
        load_embeddings,
        load_finbert_sentiment,
        train_mask,
    )
except ImportError:  # pragma: no cover
    from src.anchors import ANCHORS, CHANNELS, anchors_hash
    from src.config import (
        AXIS_TOP_K,
        EMBED_DIR,
        NOVELTY_LOOKBACK_SESSIONS,
        NOVELTY_MEMORY_HALFLIFE,
        RANDOM_SEED,
        RESULTS_DIR,
        SPLIT_HALF_REPEATS,
        TEXT_FEATURES_DIR,
    )
    from src.embeddings import (
        ENCODERS,
        get_encoder,
        load_articles,
        load_embeddings,
        load_finbert_sentiment,
        train_mask,
    )


# --------------------------------------------------------------------------
# Embedding space
# --------------------------------------------------------------------------

def _unit(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.clip(norms, 1e-9, None)


class Space:
    """Centred, unit-normalised embedding space for one encoder.

    Holds the headline vectors U (n x d) and the centred anchor vectors, with
    the centring mean estimated on training-period headlines only.
    """

    def __init__(self, name: str, articles: pd.DataFrame):
        self.name = name
        X = load_embeddings(name, articles)
        self.mu = X[train_mask(articles)].mean(axis=0)
        self.U = _unit(X - self.mu).astype(np.float32)
        raw = _anchor_embeddings(name, articles)
        self.anchors = {ch: {pole: _unit(v - self.mu) for pole, v in poles.items()}
                        for ch, poles in raw.items()}

    def direction(self, ch: str, pos_idx=None, neg_idx=None) -> np.ndarray:
        """Unit direction axis: mean(pos) - mean(neg), optionally from a subset."""
        pos, neg = self.anchors[ch]["pos"], self.anchors[ch]["neg"]
        pos = pos if pos_idx is None else pos[pos_idx]
        neg = neg if neg_idx is None else neg[neg_idx]
        return _unit(pos.mean(axis=0) - neg.mean(axis=0))

    def topic(self, ch: str, idx=None) -> np.ndarray:
        """Unit topic vector: mean of both poles' anchors, optionally a subset of pairs."""
        pos, neg = self.anchors[ch]["pos"], self.anchors[ch]["neg"]
        if idx is not None:
            pos, neg = pos[idx], neg[idx]
        return _unit(np.vstack([pos, neg]).mean(axis=0))

    def matrices(self, idx=None) -> tuple[np.ndarray, np.ndarray]:
        """(topic, direction) matrices, each d x n_channels."""
        T = np.stack([self.topic(ch, idx) for ch in CHANNELS], axis=1)
        D = np.stack([self.direction(ch, idx, idx) for ch in CHANNELS], axis=1)
        return T, D


def _anchor_embeddings(name: str, articles: pd.DataFrame) -> dict:
    """Raw anchor embeddings for one encoder, cached by anchor-text hash.

    Anchors go through exactly the same encoder object as the headlines, so
    for GloVe / FastText they get the same train-fitted SIF weights and
    common-component removal.
    """
    path = EMBED_DIR / f"anchors_{name}_{anchors_hash()}.npz"
    if path.exists():
        z = np.load(path)
        return {ch: {p: z[f"{ch}__{p}"] for p in ("pos", "neg")} for ch in CHANNELS}

    n = sum(len(v) for poles in ANCHORS.values() for v in poles.values())
    print(f"  [{name}] embedding {n} anchors ...")
    encoder = get_encoder(name, articles)
    out = {ch: {p: encoder.encode(ANCHORS[ch][p]).astype(np.float32)
                for p in ("pos", "neg")} for ch in CHANNELS}
    np.savez(path, **{f"{ch}__{p}": v for ch, poles in out.items() for p, v in poles.items()})
    return out


# --------------------------------------------------------------------------
# Headline scores and session pooling
# --------------------------------------------------------------------------

def _session_groups(articles: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[np.ndarray]]:
    groups = articles.groupby("trading_day").indices
    days = pd.DatetimeIndex(sorted(groups))
    return days, [np.asarray(groups[d]) for d in days]


def headline_scores(space: Space, idx=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-headline (relevance, direction), each n_articles x n_channels."""
    T, D = space.matrices(idx)
    return space.U @ T, space.U @ D


def pool_top_k(relevance: np.ndarray, direction: np.ndarray,
               groups: list[np.ndarray], k: int = AXIS_TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Per session and channel: mean direction and mean relevance of the top-k
    most relevant headlines. Returns two (n_sessions x n_channels) arrays."""
    n_ch = relevance.shape[1]
    dir_out = np.empty((len(groups), n_ch), dtype=np.float32)
    att_out = np.empty_like(dir_out)
    for s, idx in enumerate(groups):
        R, Dv = relevance[idx], direction[idx]
        kk = min(k, len(idx))
        top = np.argpartition(-R, kk - 1, axis=0)[:kk]          # kk x n_ch
        cols = np.arange(n_ch)
        dir_out[s] = Dv[top, cols].mean(axis=0)
        att_out[s] = R[top, cols].mean(axis=0)
    return dir_out, att_out


def axis_features(space: Space, articles: pd.DataFrame) -> pd.DataFrame:
    """ax_<ch>_dir and ax_<ch>_attn for every session."""
    days, groups = _session_groups(articles)
    rel, drn = headline_scores(space)
    d, a = pool_top_k(rel, drn, groups)
    cols = {}
    for c, ch in enumerate(CHANNELS):
        cols[f"ax_{ch}_dir"] = d[:, c]
        cols[f"ax_{ch}_attn"] = a[:, c]
    out = pd.DataFrame(cols, index=days)
    out.index.name = "date"
    return out


def novelty_features(space: Space, articles: pd.DataFrame) -> pd.DataFrame:
    """Novelty and concentration per session, from past sessions only."""
    days, groups = _session_groups(articles)
    U = space.U
    decay = 0.5 ** (1.0 / NOVELTY_MEMORY_HALFLIFE)
    memory = None
    rows = []

    for i, idx in enumerate(groups):
        Ut = U[idx]
        n = len(idx)

        # Headline novelty: distance to the closest headline in the lookback.
        lo = max(0, i - NOVELTY_LOOKBACK_SESSIONS)
        if i > 0:
            prev = np.concatenate(groups[lo:i])
            nov_headline = float(np.mean(1.0 - (Ut @ U[prev].T).max(axis=1)))
        else:
            nov_headline = np.nan

        # Centroid novelty against an exponentially decayed memory of the past.
        centroid = _unit(Ut.mean(axis=0))
        nov_centroid = np.nan if memory is None else float(1.0 - centroid @ _unit(memory))
        memory = centroid if memory is None else decay * memory + (1 - decay) * centroid

        # Concentration: mean pairwise cosine, via |sum u|^2 = n + sum_{i!=j} u_i.u_j
        if n >= 2:
            s = Ut.sum(axis=0)
            concentration = float((s @ s - n) / (n * (n - 1)))
        else:
            concentration = np.nan

        rows.append((nov_headline, nov_centroid, concentration))

    out = pd.DataFrame(rows, index=days, columns=["nov_headline", "nov_centroid", "concentration"])
    out.index.name = "date"
    return out


def finbert_sentiment_daily(articles: pd.DataFrame) -> pd.DataFrame:
    """Session means of FinBERT's class probabilities, plus net tone."""
    probs = load_finbert_sentiment(articles)
    probs.columns = [c.replace("finbert_", "fbs_") for c in probs.columns]
    probs["fbs_net"] = probs["fbs_positive"] - probs["fbs_negative"]
    out = probs.groupby(articles["trading_day"].to_numpy()).mean()
    out.columns = [f"{c}_mean" for c in out.columns]
    out.index.name = "date"
    return out


def build_text_features(name: str, articles: pd.DataFrame | None = None,
                        space: Space | None = None) -> pd.DataFrame:
    """All session-level text features for one encoder."""
    if articles is None:
        articles = load_articles()
    if space is None:
        space = Space(name, articles)
    parts = [axis_features(space, articles), novelty_features(space, articles)]
    if name == "finbert":
        parts.append(finbert_sentiment_daily(articles))
    return pd.concat(parts, axis=1).reset_index()


def load_text_features(name: str) -> pd.DataFrame:
    """Saved session-level features for one encoder (date as datetime)."""
    df = pd.read_csv(TEXT_FEATURES_DIR / f"{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def apply_decay(df: pd.DataFrame, cols: list[str], halflife: float) -> pd.DataFrame:
    """Exponentially decayed version of `cols` over sessions (past and present only).

    halflife is in sessions; 0 returns the columns unchanged. Missing sessions
    are skipped rather than treated as zero. Rows must be one per session.
    """
    out = df.sort_values("date").copy()
    if halflife and halflife > 0:
        out[cols] = out[cols].ewm(halflife=halflife, ignore_na=True).mean()
    return out


# --------------------------------------------------------------------------
# Diagnostics (no returns used)
# --------------------------------------------------------------------------

def loo_pair_accuracy(space: Space) -> dict[str, float]:
    """Share of held-out anchors on their own side when their whole pair is dropped."""
    acc = {}
    for ch in CHANNELS:
        k = len(space.anchors[ch]["pos"])
        correct = 0
        for j in range(k):
            keep = np.array([i for i in range(k) if i != j])
            axis = space.direction(ch, keep, keep)
            correct += space.anchors[ch]["pos"][j] @ axis > 0
            correct += space.anchors[ch]["neg"][j] @ axis < 0
        acc[ch] = correct / (2 * k)
    return acc


def split_half_reliability(space: Space, articles: pd.DataFrame) -> dict[str, float]:
    """Mean correlation of top-k session direction scores from disjoint anchor halves.

    Pairs stay together, so each half is itself a set of minimal pairs and
    builds its own topic and direction. Training-period sessions only.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    sub = articles[train_mask(articles)].reset_index(drop=True)
    U_train = space.U[train_mask(articles)]
    _, groups = _session_groups(sub)
    k = len(next(iter(ANCHORS.values()))["pos"])
    rs = {ch: [] for ch in CHANNELS}
    for _ in range(SPLIT_HALF_REPEATS):
        perm = rng.permutation(k)
        halves = []
        for idx in (perm[: k // 2], perm[k // 2:]):
            T, D = space.matrices(idx)
            d, _ = pool_top_k(U_train @ T, U_train @ D, groups)
            halves.append(d)
        for c, ch in enumerate(CHANNELS):
            rs[ch].append(np.corrcoef(halves[0][:, c], halves[1][:, c])[0, 1])
    return {ch: float(np.mean(v)) for ch, v in rs.items()}


def top_headlines(space: Space, articles: pd.DataFrame, k: int = 4,
                  pool: int = 300) -> str:
    """Among the `pool` most channel-relevant training headlines, the k most
    USD-bullish and k most USD-bearish by direction score."""
    tm = np.flatnonzero(train_mask(articles))
    rel, drn = headline_scores(space)
    rel, drn = rel[tm], drn[tm]
    titles = articles["title_clean"].to_numpy()[tm]
    lines = [f"===== {space.name} ====="]
    for c, ch in enumerate(CHANNELS):
        relevant = np.argsort(-rel[:, c])[:pool]
        order = relevant[np.argsort(drn[relevant, c])]
        lines.append(f"\n[{ch}]  most relevant: {titles[relevant[0]][:90]}")
        lines.append(f"[{ch}]  USD/IDR-up pole (among top-{pool} relevant):")
        lines += [f"   {drn[i, c]:+.3f}  {titles[i][:95]}" for i in order[::-1][:k]]
        lines.append(f"[{ch}]  USD/IDR-down pole:")
        lines += [f"   {drn[i, c]:+.3f}  {titles[i][:95]}" for i in order[:k]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    args = parser.parse_args()

    articles = load_articles()
    TEXT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"anchor set {anchors_hash()}: {len(CHANNELS)} channels x 2 poles x 10 anchors, "
          f"top-{AXIS_TOP_K} selection\n")

    diag_rows, tops, corr_tables = [], [], {}
    for name in args.encoders:
        print(f"[{name}] building space ...")
        space = Space(name, articles)

        feats = build_text_features(name, articles, space)
        path = TEXT_FEATURES_DIR / f"{name}.csv"
        feats.to_csv(path, index=False, float_format="%.6f")
        print(f"  wrote {path.name}: {feats.shape[0]} sessions x {feats.shape[1] - 1} features")

        loo = loo_pair_accuracy(space)
        rel = split_half_reliability(space, articles)
        dir_cols = [f"ax_{ch}_dir" for ch in CHANNELS]
        corr = feats[dir_cols].corr()
        corr.index = corr.columns = list(CHANNELS)
        corr_tables[name] = corr.round(2)
        off = corr.to_numpy()[~np.eye(len(CHANNELS), dtype=bool)]
        for ch in CHANNELS:
            diag_rows.append({"encoder": name, "channel": ch,
                              "loo_pair_acc": loo[ch], "split_half_r": rel[ch],
                              "mean_abs_corr_other_channels":
                                  float(corr.loc[ch].drop(ch).abs().mean())})
        print(f"  mean |corr| between channels: {np.abs(off).mean():.2f}")
        tops.append(top_headlines(space, articles))

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(RESULTS_DIR / "axes_diagnostics.csv", index=False, float_format="%.4f")
    (RESULTS_DIR / "axes_top_headlines.txt").write_text("\n\n".join(tops), encoding="utf-8")

    pd.set_option("display.width", 140)
    for metric, title in [
        ("loo_pair_acc", "Leave-one-pair-out anchor accuracy (1.0 = every held-out anchor on its side)"),
        ("split_half_r", "Split-half reliability of session direction scores"),
        ("mean_abs_corr_other_channels", "Mean |corr| with the other six channels (lower = more distinct)"),
    ]:
        print(f"\n{title}")
        print(diag.pivot(index="channel", columns="encoder", values=metric)
              .loc[list(CHANNELS), list(args.encoders)].round(2).to_string())
    for name, c in corr_tables.items():
        print(f"\nInter-channel correlation of session direction scores [{name}]")
        print(c.to_string())
    print(f"\nWrote {RESULTS_DIR / 'axes_diagnostics.csv'}")
    print(f"Wrote {RESULTS_DIR / 'axes_top_headlines.txt'}")


if __name__ == "__main__":
    main()
