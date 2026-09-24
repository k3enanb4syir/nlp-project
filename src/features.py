# src/features.py
"""
Task 2 -- NLP feature extraction and daily feature-table construction.

METHOD CONSTRAINTS
------------------
Task 2 forbids pre-trained embeddings and transformer models (FinBERT, RoBERTa,
LLM embeddings). Everything here is therefore either a rule-based lexicon count
or a classical count/TF-IDF statistic estimated from the training split only.
No component of this module loads learned weights from anywhere.

WHY TITLES
----------
The Task 2 hint asks whether titles, body text or entities carry the most
signal. This pipeline uses titles as the primary text, with GDELT's own theme
and entity annotations as a structured complement. The reasoning:

  * Titles are a near-lossless summary of the event at high precision. Body text
    from a web scrape is dominated by boilerplate, and the noise-to-signal ratio
    of a 900-word article against a 12-word headline is poor for a lexicon.
  * Titles are uniformly available for every article in GDELT; full text is not,
    and fetching it for ~1M URLs is infeasible and legally fraught.
  * For a daily aggregate, what matters is the *distribution* of stances across
    many articles, not depth within one. Many short signals beat few long ones.

Entities and themes are used as complementary structured features rather than as
the main text, because they are categorical and already de-duplicated by GDELT.

FEATURE GROUPS PRODUCED
-----------------------
  A. VADER sentiment aggregates over the day's titles.
  B. Loughran-McDonald-style financial lexicon counts (hand-curated; see
     FINANCIAL_LEXICONS for why a hand-curated list is used here).
  C. GDELT theme-family counts (macro / conflict / uncertainty).
  D. Entity-density features from GDELT's person and organisation extraction.
  E. News volume and dispersion.
  F. Lagged exchange-rate features (the time-series side).
  G. The concatenated daily title string, left as text so that TF-IDF can be
     fitted on the TRAINING SPLIT ONLY inside the modelling pipeline. Fitting a
     vectoriser here, over all five years, would leak test-period vocabulary and
     document frequencies into training.

LEAKAGE DISCIPLINE
------------------
Every feature for trading day t is computed only from:
  * articles aligned to t (i.e. published strictly before t's fixing), and
  * exchange-rate observations up to and including t-1.
The target is the direction of the fixing on day t. See build_daily_table.

Usage:
    python src/features.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from config import (
        ALIGNED_ARTICLES_CSV,
        DAILY_DATASET_CSV,
        RATE_LAGS,
        RATES_CLEAN_CSV,
        ROLLING_WINDOWS,
    )
except ImportError:  # pragma: no cover
    from src.config import (
        ALIGNED_ARTICLES_CSV,
        DAILY_DATASET_CSV,
        RATE_LAGS,
        RATES_CLEAN_CSV,
        ROLLING_WINDOWS,
    )


# --------------------------------------------------------------------------
# B. Financial / geopolitical lexicons
# --------------------------------------------------------------------------
# The published Loughran-McDonald master dictionary is a licensed spreadsheet
# that would have to be vendored into the repo. Rather than ship a copy, we use
# a compact hand-curated lexicon built on the same principle that motivates LM:
# general-purpose sentiment lexicons mis-score financial prose, because words
# like "liability", "tightening" or "exposure" are neutral in ordinary English
# but directional in a market context. The categories below mirror LM's
# Negative / Uncertainty / Constraining groupings, plus a geopolitical-risk
# group specific to this project's hypothesis.
FINANCIAL_LEXICONS: dict[str, frozenset[str]] = {
    # LM "Negative" analogue -- deterioration in financial conditions
    "fin_negative": frozenset({
        "slump", "plunge", "crash", "collapse", "tumble", "slide", "fall",
        "decline", "drop", "weaken", "weakness", "loss", "losses", "deficit",
        "recession", "downturn", "contraction", "default", "bankruptcy",
        "insolvency", "downgrade", "selloff", "rout", "bearish", "crisis",
        "turmoil", "shock", "distress", "layoffs", "unemployment", "inflation",
        "devaluation", "depreciation", "shortfall", "writedown", "deteriorate",
    }),
    # LM "Positive" analogue -- improvement in financial conditions
    "fin_positive": frozenset({
        "rally", "surge", "soar", "gain", "gains", "rebound", "recover",
        "recovery", "growth", "expand", "expansion", "profit", "profits",
        "upgrade", "bullish", "strengthen", "strength", "boost", "outperform",
        "record", "optimism", "stabilise", "stabilize", "improve", "improvement",
        "appreciation", "surplus", "boom", "resilient",
    }),
    # LM "Uncertainty" -- the category most predictive of volatility
    "fin_uncertainty": frozenset({
        "uncertain", "uncertainty", "risk", "risks", "risky", "volatile",
        "volatility", "doubt", "unclear", "ambiguous", "unpredictable",
        "speculation", "speculative", "could", "may", "might", "possible",
        "potential", "unknown", "fluctuate", "cautious", "caution", "concern",
        "concerns", "worry", "worries", "fear", "fears", "jitters",
    }),
    # LM "Constraining" -- policy and regulatory tightening
    "fin_constraining": frozenset({
        "restrict", "restriction", "restrictions", "ban", "banned", "curb",
        "curbs", "limit", "limits", "cap", "quota", "tariff", "tariffs",
        "sanction", "sanctions", "embargo", "tighten", "tightening",
        "regulation", "regulate", "mandate", "prohibit", "freeze", "block",
        "barrier", "barriers", "levy", "duties",
    }),
    # Project-specific: geopolitical risk vocabulary
    "geo_risk": frozenset({
        "war", "warfare", "conflict", "invasion", "invade", "attack", "strike",
        "strikes", "missile", "drone", "troops", "military", "combat",
        "escalation", "escalate", "ceasefire", "truce", "hostilities",
        "terrorist", "terrorism", "insurgency", "coup", "unrest", "protest",
        "protests", "riot", "clash", "clashes", "nuclear", "retaliation",
        "retaliate", "threat", "threats", "tension", "tensions", "dispute",
    }),
    # Project-specific: monetary-policy vocabulary
    "monetary": frozenset({
        "fed", "federal", "reserve", "fomc", "ecb", "boj", "pboc", "rate",
        "rates", "hike", "cut", "cuts", "easing", "hawkish", "dovish",
        "yield", "yields", "bond", "bonds", "treasury", "liquidity",
        "stimulus", "tapering", "policy", "inflation", "cpi", "deflation",
    }),
}

_TOKEN_RE = re.compile(r"[a-z']+")


def lexicon_counts(tokens: list[str]) -> dict[str, int]:
    """Count how many tokens fall in each lexicon category."""
    return {
        name: sum(1 for t in tokens if t in vocab)
        for name, vocab in FINANCIAL_LEXICONS.items()
    }


# --------------------------------------------------------------------------
# C. GDELT theme families
# --------------------------------------------------------------------------
THEME_FAMILIES = {
    "theme_currency": ("ECON_WORLDCURRENCIES", "ECON_CURRENCY_EXCHANGE_RATE"),
    "theme_monetary": ("ECON_CENTRALBANK", "ECON_INTEREST_RATES",
                       "ECON_INFLATION", "EPU_CATS_MONETARY_POLICY"),
    "theme_fiscal": ("ECON_DEBT", "ECON_BUDGET_DEFICIT", "ECON_TAXATION",
                     "EPU_CATS_FISCAL_POLICY"),
    "theme_trade": ("ECON_FREETRADE", "ECON_TRADE_DISPUTE", "ECON_BOYCOTT",
                    "EPU_CATS_TRADE_POLICY"),
    "theme_conflict": ("ARMEDCONFLICT", "TERROR", "NUCLEAR", "MILITARY_",
                       "FRAGILITY_CONFLICT_AND_VIOLENCE"),
    "theme_sanctions": ("SANCTION", "BLOCKADE", "EMBARGO"),
    "theme_uncertainty": ("EPU_UNCERTAINTY", "EPU_CATS_NATIONAL_SECURITY"),
    "theme_markets": ("ECON_STOCKMARKET", "ECON_OILPRICE", "ECON_GOLDPRICE"),
}


def theme_family_flags(themes: str) -> dict[str, int]:
    """Binary indicator per theme family for one article."""
    s = themes if isinstance(themes, str) else ""
    return {
        name: int(any(frag in s for frag in frags))
        for name, frags in THEME_FAMILIES.items()
    }


# --------------------------------------------------------------------------
# A + B + C + D: article-level scoring
# --------------------------------------------------------------------------

def score_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """
    Attach per-article VADER, lexicon and theme features.

    VADER is applied to `title_clean` (which retains punctuation and case)
    because its rules read exclamation marks, capitalisation and degree
    modifiers as intensity signals.
    """
    analyzer = SentimentIntensityAnalyzer()

    compound, pos, neu, neg = [], [], [], []
    lex_rows: list[dict[str, int]] = []
    theme_rows: list[dict[str, int]] = []

    for title_clean, title_norm, themes in zip(
        articles["title_clean"].fillna(""),
        articles["title_norm"].fillna(""),
        articles["themes"].fillna(""),
    ):
        scores = analyzer.polarity_scores(title_clean)
        compound.append(scores["compound"])
        pos.append(scores["pos"])
        neu.append(scores["neu"])
        neg.append(scores["neg"])

        lex_rows.append(lexicon_counts(_TOKEN_RE.findall(title_norm)))
        theme_rows.append(theme_family_flags(themes))

    out = articles.copy()
    out["vader_compound"] = compound
    out["vader_pos"] = pos
    out["vader_neu"] = neu
    out["vader_neg"] = neg

    for col in FINANCIAL_LEXICONS:
        out[col] = [r[col] for r in lex_rows]
    for col in THEME_FAMILIES:
        out[col] = [r[col] for r in theme_rows]

    # D. Entity density. The scraper already stores GDELT's extracted person /
    # organisation / location entities as counts rather than as raw strings, so
    # these only need coercing to a numeric dtype.
    for col in ("n_persons", "n_orgs", "n_locations"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0)

    return out


# --------------------------------------------------------------------------
# E: aggregate article-level features to one row per trading day
# --------------------------------------------------------------------------

def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse articles onto the trading day their news was aligned to.

    Both the mean and the dispersion of sentiment are kept: a day on which the
    press is uniformly negative is a different state from a day on which it is
    violently split, and only the dispersion distinguishes them.
    """
    g = scored.groupby("trading_day")

    daily = pd.DataFrame({
        # E. volume
        "n_articles": g.size(),
        "n_domains": g["domain"].nunique(),

        # A. VADER aggregates
        "vader_mean": g["vader_compound"].mean(),
        "vader_std": g["vader_compound"].std(),
        "vader_min": g["vader_compound"].min(),
        "vader_max": g["vader_compound"].max(),
        "vader_pos_share": g["vader_compound"].apply(lambda s: (s > 0.05).mean()),
        "vader_neg_share": g["vader_compound"].apply(lambda s: (s < -0.05).mean()),
        "vader_pos_mean": g["vader_pos"].mean(),
        "vader_neg_mean": g["vader_neg"].mean(),

        # GDELT's own tone, kept as a documented comparison baseline
        "gdelt_tone_mean": g["gdelt_tone"].mean(),
        "gdelt_tone_std": g["gdelt_tone"].std(),

        # D. entity density
        "persons_mean": g["n_persons"].mean(),
        "orgs_mean": g["n_orgs"].mean(),
        "locations_mean": g["n_locations"].mean(),
    })

    # B. lexicon rates, normalised per article so volume does not dominate
    for col in FINANCIAL_LEXICONS:
        daily[f"{col}_rate"] = g[col].mean()

    # C. theme family shares
    for col in THEME_FAMILIES:
        daily[f"{col}_share"] = g[col].mean()

    # Net financial tone: a single directional summary from the lexicons.
    daily["fin_net_tone"] = daily["fin_positive_rate"] - daily["fin_negative_rate"]

    # G. daily text, kept raw for train-only TF-IDF fitting downstream
    daily["titles_concat"] = g["title_norm"].apply(lambda s: " ".join(s.dropna()))

    daily = daily.reset_index().rename(columns={"trading_day": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# --------------------------------------------------------------------------
# F: exchange-rate (time-series) features and the target
# --------------------------------------------------------------------------

def build_rate_features(rates: pd.DataFrame) -> pd.DataFrame:
    """
    Build lagged return features and the binary direction target.

    CRITICAL ORDERING: `log_return` on row t is the move INTO day t, i.e. the
    target. Every predictor is therefore shifted by at least one trading day, so
    that a model predicting day t sees only information available after day
    t-1's fixing.
    """
    df = rates[["date", "rate"]].copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Target-side quantities.
    df["log_return"] = np.log(df["rate"] / df["rate"].shift(1))
    df["direction"] = (df["log_return"] > 0).astype(int)

    # --- predictors: all strictly lagged ---
    df["prev_rate"] = df["rate"].shift(1)

    for lag in RATE_LAGS:
        # ret_lag1 is the return into day t-1, known once t-1 has fixed.
        df[f"ret_lag{lag}"] = df["log_return"].shift(lag)

    prev_ret = df["log_return"].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"vol_{w}"] = prev_ret.rolling(w).std()
        df[f"mom_{w}"] = prev_ret.rolling(w).mean()

    # Distance of the last fixing from its own recent average -- a simple
    # mean-reversion / trend-position feature.
    for w in ROLLING_WINDOWS:
        ma = df["rate"].shift(1).rolling(w).mean()
        df[f"dist_ma{w}"] = (df["rate"].shift(1) - ma) / ma

    # Day-of-week: FX flows have weekday seasonality (month-end, weekend risk).
    df["dow"] = df["date"].dt.dayofweek

    # Persistence baseline input: yesterday's realised direction.
    df["prev_direction"] = df["direction"].shift(1)

    return df


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_daily_table() -> pd.DataFrame:
    """Join news features onto the trading calendar and emit the model table."""
    print("Loading aligned articles ...")
    articles = pd.read_csv(ALIGNED_ARTICLES_CSV, low_memory=False)
    articles["trading_day"] = pd.to_datetime(articles["trading_day"])
    print(f"  {len(articles):,} aligned articles")

    print("Scoring articles (VADER + lexicons + themes) ...")
    scored = score_articles(articles)

    print("Aggregating to trading days ...")
    daily_news = aggregate_daily(scored)
    print(f"  {len(daily_news):,} trading days carry news")

    print("Building exchange-rate features ...")
    rates = pd.read_csv(RATES_CLEAN_CSV)
    rate_feats = build_rate_features(rates)

    # LEFT join on the rate calendar: the trading calendar is authoritative, and
    # a session with no qualifying news is a real state we must represent.
    df = rate_feats.merge(daily_news, on="date", how="left")

    news_cols = [c for c in daily_news.columns if c not in ("date", "titles_concat")]
    df["has_news"] = df["n_articles"].notna().astype(int)
    # Zero-fill count-like columns; leave distributional statistics as NaN so a
    # newsless day is not silently recorded as neutral sentiment.
    for col in ["n_articles", "n_domains"]:
        df[col] = df[col].fillna(0)
    df["titles_concat"] = df["titles_concat"].fillna("")

    print(f"  merged table: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  sessions with news: {int(df['has_news'].sum()):,} / {len(df):,}")

    return df


def main() -> None:
    df = build_daily_table()
    DAILY_DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DAILY_DATASET_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {DAILY_DATASET_CSV}")
    print(f"  rows {len(df):,}  columns {df.shape[1]}")
    print(f"  date range {df['date'].min().date()} -> {df['date'].max().date()}")
    up = df["direction"].mean()
    print(f"  target base rate (UP): {up:.3f}")


if __name__ == "__main__":
    main()
