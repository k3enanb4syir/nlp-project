# src/preprocessing.py
"""
Task 1 -- Data cleansing, text preprocessing and temporal alignment.

This module turns the raw GDELT article pull and the raw Bank Indonesia JISDOR
export into a single article-level dataset in which every article is attached to
the trading day whose fixing it could plausibly have influenced.

-----------------------------------------------------------------------------
1. TEXT CLEANING -- WHAT IS RETAINED AND WHAT IS DROPPED (requirement 3b)
-----------------------------------------------------------------------------
The unit of text is the article TITLE. Titles are preferred over body text
because they are editorially compressed statements of the event, they are
available uniformly across every outlet in GDELT, and they carry none of the
navigation chrome, cookie notices, share prompts and related-article lists that
dominate scraped article bodies.

RETAINED, deliberately:
  * Numerals and percentages ("Fed holds at 5.5%") -- magnitude is signal.
  * Negation words (not/no/never/without) -- inverting them inverts the
    sentiment, so a stopword list that strips them would be actively harmful.
  * Sentence punctuation and original capitalisation, in `title_clean`. VADER
    uses "!"  and ALL-CAPS as intensity amplifiers, so stripping them would
    discard information the lexicon is designed to read.

DROPPED, deliberately:
  * Outlet boilerplate suffixes ("... | Chicago News", "... - KRRV-FM"). These
    are constant per publisher, so they add no discriminative signal while
    inflating TF-IDF weights for publisher names. They are detected
    data-drivenly rather than by a hand-written list -- see strip_boilerplate.
  * HTML entities and stray markup left in GDELT's PAGE_TITLE field.
  * Non-English titles: an English sentiment lexicon cannot score them.
  * Duplicate and syndicated copies of the same story (same URL, or the same
    title on the same day), which would otherwise let one wire story vote many
    times in the daily aggregate.
  * Very short titles (< 4 tokens), which are almost always section headers
    ("Business", "World News") rather than reporting.

A second lowercased, punctuation-stripped column `title_norm` is emitted for the
bag-of-words / TF-IDF path, which does not benefit from case or punctuation.

-----------------------------------------------------------------------------
2. SOURCE FILTERING (requirement 1c, stage 2)
-----------------------------------------------------------------------------
GDELT indexes tens of thousands of publishers, and its raw feed is dominated by
hyper-local outlets -- in a sample snapshot the top domains were local UK and
regional US papers, and only ~6% of records came from major national or
international mastheads. Those local outlets do report on national politics, but
their coverage is not what moves a currency market and their volume swamps the
outlets that do.

We therefore whitelist established national and international news and financial
publishers (see REPUTABLE_DOMAINS). This is applied here rather than in the
scraper so that the whitelist can be revised without re-downloading five years
of data.

-----------------------------------------------------------------------------
3. TEMPORAL ALIGNMENT (requirement 4)
-----------------------------------------------------------------------------
Two clocks have to be reconciled:

  * Bank Indonesia publishes the JISDOR fixing once per BUSINESS day, at
    ~10:00 WIB (UTC+7), i.e. 03:00 UTC.
  * GDELT news runs 24/7, including weekends and Indonesian public holidays.

Our GDELT sampling snapshot is fixed at 12:00 UTC, which is 19:00 WIB -- always
AFTER that same day's fixing has been published. It is therefore impossible for
an article in this corpus to have influenced the rate printed on its own
calendar date, and aligning it to that date would inject look-ahead bias.

THE RULE: every article is aligned to the first trading day STRICTLY AFTER its
publication date.

    article published Mon (19:00 WIB)  -> Tue's fixing
    article published Fri (19:00 WIB)  -> Mon's fixing
    article published Sat or Sun       -> Mon's fixing
    article published on a holiday     -> next open trading day's fixing

Equivalently, the fixing on trading day t absorbs every article published from
the previous trading day t-1 through the calendar day before t. This is the
natural consequence of the rule above and is what makes weekend and holiday news
accumulate onto the next session, as requirement 4b asks.

The trading calendar is taken from the JISDOR series itself rather than from a
generic holiday library, so Indonesian public holidays and any unscheduled BI
closures are handled implicitly and exactly.

Usage:
    python src/preprocessing.py
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

try:
    from config import (
        ALIGNED_ARTICLES_CSV,
        ARTICLES_CLEAN_CSV,
        BI_RAW_CSV,
        CLEANED_DIR,
        GDELT_RAW_CSV,
        RATES_CLEAN_CSV,
        STUDY_END,
        STUDY_START,
    )
    from bi_scraper import load_raw_export
except ImportError:  # pragma: no cover
    from src.config import (
        ALIGNED_ARTICLES_CSV,
        ARTICLES_CLEAN_CSV,
        BI_RAW_CSV,
        CLEANED_DIR,
        GDELT_RAW_CSV,
        RATES_CLEAN_CSV,
        STUDY_END,
        STUDY_START,
    )
    from src.bi_scraper import load_raw_export


# --------------------------------------------------------------------------
# Source whitelist
# --------------------------------------------------------------------------
# Established national / international news agencies and financial press.
# Grouped by rationale so the choice is auditable.
REPUTABLE_DOMAINS = frozenset({
    # Global wire services -- the outlets that actually move FX desks
    "reuters.com", "apnews.com", "afp.com", "bloomberg.com",
    # Dedicated financial press
    "cnbc.com", "ft.com", "wsj.com", "marketwatch.com", "barrons.com",
    "investing.com", "fxstreet.com", "forexlive.com", "tradingeconomics.com",
    "economist.com", "fortune.com", "businessinsider.com", "nasdaq.com",
    "kitco.com", "seekingalpha.com", "thestreet.com", "benzinga.com",
    "finance.yahoo.com", "money.cnn.com", "morningstar.com",
    # Major international general news
    "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "cnn.com", "nbcnews.com", "cbsnews.com",
    "abcnews.go.com", "usatoday.com", "latimes.com", "npr.org",
    "foxbusiness.com", "thehill.com", "politico.com", "axios.com",
    "newsweek.com", "time.com", "independent.co.uk", "telegraph.co.uk",
    "aljazeera.com", "dw.com", "france24.com", "euronews.com", "rte.ie",
    "globalnews.ca", "cbc.ca", "theglobeandmail.com",
    # Asia-Pacific -- directly relevant to a USD/IDR cross
    "scmp.com", "japantimes.co.jp", "straitstimes.com", "channelnewsasia.com",
    "asia.nikkei.com", "nikkei.com", "business-standard.com",
    "economictimes.indiatimes.com", "livemint.com", "thehindubusinessline.com",
    "thejakartapost.com", "jakartaglobe.id", "antaranews.com",
    "bangkokpost.com", "koreaherald.com", "abc.net.au", "afr.com",
    "smh.com.au", "theaustralian.com.au", "nzherald.co.nz",
    # Middle East / other regions with heavy geopolitical coverage
    "arabnews.com", "gulfnews.com", "thenationalnews.com", "haaretz.com",
    "timesofisrael.com", "jpost.com", "middleeasteye.net",
    "africanews.com", "businesslive.co.za", "news24.com",
})


def _domain_matches(domain: str) -> bool:
    """Whitelist test tolerant of subdomains (e.g. 'edition.cnn.com')."""
    if not isinstance(domain, str) or not domain:
        return False
    d = domain.strip().lower().removeprefix("www.")
    if d in REPUTABLE_DOMAINS:
        return True
    return any(d.endswith("." + allowed) for allowed in REPUTABLE_DOMAINS)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

# Publishers separate the headline from their masthead with a pipe, dash or
# en/em dash. This splits on the LAST such separator.
_SEP_RE = re.compile(r"\s+[|–—·>-]\s+")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MARKUP_RE = re.compile(r"<[^>]+>")
# Keep letters, digits, and the punctuation VADER actually reads.
_PUNCT_KEEP_RE = re.compile(r"[^\w\s!?.,'%$-]", flags=re.UNICODE)
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")


def learn_boilerplate(df: pd.DataFrame, min_share: float = 0.30,
                      max_words: int = 6) -> dict[str, set[str]]:
    """
    Learn each publisher's masthead suffix from the data instead of hard-coding.

    For every domain we look at the final separator-delimited segment of each
    title. If one segment recurs in at least `min_share` of that domain's titles
    and is short enough to be a masthead rather than a headline clause, it is
    recorded as boilerplate for that domain.

    Doing this per-domain is what makes it safe: "Reuters" is boilerplate on
    reuters.com but genuine content in a headline on another outlet.
    """
    tails: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()

    for domain, title in zip(df["domain"], df["title"]):
        if not isinstance(title, str) or not isinstance(domain, str):
            continue
        totals[domain] += 1
        parts = _SEP_RE.split(title)
        if len(parts) >= 2:
            tail = parts[-1].strip()
            if tail and len(tail.split()) <= max_words:
                tails[domain][tail] += 1

    boilerplate: dict[str, set[str]] = {}
    for domain, counter in tails.items():
        n = totals[domain]
        if n < 5:                       # too little evidence to generalise
            continue
        found = {tail for tail, cnt in counter.items() if cnt / n >= min_share}
        if found:
            boilerplate[domain] = found
    return boilerplate


def strip_boilerplate(title: str, domain: str,
                      boilerplate: dict[str, set[str]]) -> str:
    """Remove a learned masthead suffix from one title."""
    if not isinstance(title, str):
        return ""
    known = boilerplate.get(domain)
    if not known:
        return title
    # Strip repeatedly: some outlets append two suffixes ("... | Business | XYZ").
    out = title
    for _ in range(3):
        parts = _SEP_RE.split(out)
        if len(parts) >= 2 and parts[-1].strip() in known:
            out = _SEP_RE.split(out)
            out = " ".join(out[:-1]) if len(out) > 2 else out[0]
            out = out.strip()
        else:
            break
    return out or title


def clean_title(title: str) -> str:
    """Normalise a single title for the sentiment path."""
    if not isinstance(title, str):
        return ""
    out = _MARKUP_RE.sub(" ", title)
    out = _URL_RE.sub(" ", out)
    out = _PUNCT_KEEP_RE.sub(" ", out)
    return _WS_RE.sub(" ", out).strip()


def normalise_title(title: str) -> str:
    """Lowercased, punctuation-free variant for the bag-of-words path."""
    if not isinstance(title, str):
        return ""
    out = re.sub(r"[^\w\s]", " ", title.lower())
    return _WS_RE.sub(" ", out).strip()


def looks_english(title: str) -> bool:
    """
    Cheap ASCII-ratio language guard.

    GDELT's language field is not always populated on the GKG rows, so we also
    require that the title be overwhelmingly Latin-script. This is crude but
    adequate: the goal is only to stop an English lexicon being pointed at text
    it cannot score.
    """
    if not title:
        return False
    non_latin = len(_NON_LATIN_RE.findall(title))
    return non_latin / max(len(title), 1) < 0.15


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------

def clean_articles(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply stage-2 filtering and text normalisation, reporting attrition."""
    report: list[tuple[str, int]] = [("raw rows from scraper", len(raw))]

    df = raw.copy()
    df["query_date"] = pd.to_datetime(df["query_date"], errors="coerce")
    df = df.dropna(subset=["query_date", "title", "domain"])
    report.append(("with usable date/title/domain", len(df)))

    df["domain"] = df["domain"].astype(str).str.strip().str.lower()

    # --- reputable-source whitelist ---
    df = df[df["domain"].map(_domain_matches)]
    report.append(("from whitelisted publishers", len(df)))

    # --- de-duplication ---
    # Same URL twice is the same article seen in two snapshots.
    df = df.drop_duplicates(subset=["url"])
    report.append(("after URL de-duplication", len(df)))
    # The same headline on the same day across outlets is wire syndication.
    df = df.drop_duplicates(subset=["query_date", "title"])
    report.append(("after syndication de-duplication", len(df)))

    # --- text normalisation ---
    boilerplate = learn_boilerplate(df)
    df["title_stripped"] = [
        strip_boilerplate(t, d, boilerplate)
        for t, d in zip(df["title"], df["domain"])
    ]
    df["title_clean"] = df["title_stripped"].map(clean_title)
    df["title_norm"] = df["title_clean"].map(normalise_title)

    # --- quality gates ---
    df = df[df["title_clean"].map(looks_english)]
    report.append(("English-script titles", len(df)))

    df["n_tokens"] = df["title_norm"].str.split().str.len().fillna(0).astype(int)
    df = df[df["n_tokens"] >= 4]
    report.append(("titles with >= 4 tokens", len(df)))

    # --- restrict to the study window ---
    df = df[
        (df["query_date"].dt.date >= STUDY_START)
        & (df["query_date"].dt.date <= STUDY_END)
    ]
    report.append(("inside study window", len(df)))

    print("\nCleaning attrition:")
    prev = None
    for label, n in report:
        delta = "" if prev is None else f"  ({n - prev:+,})"
        print(f"  {label:38s} {n:>9,}{delta}")
        prev = n
    print(f"  learned boilerplate suffixes for {len(boilerplate)} domains")

    return df.reset_index(drop=True)


def align_to_trading_days(articles: pd.DataFrame,
                          rates: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each article to the first trading day STRICTLY AFTER its publication.

    Implemented with a backward-looking merge on the *rates* side so that the
    result is the next available fixing. `allow_exact_matches=False` is the line
    that enforces "strictly after" and therefore prevents the look-ahead bias
    described in the module docstring -- the original Task 1 submission aligned
    news to its own calendar date, which let post-fixing news predict a fixing
    that had already been published.
    """
    left = articles.sort_values("query_date").copy()
    right = rates[["date", "rate"]].sort_values("date").rename(
        columns={"date": "trading_day"}
    )

    aligned = pd.merge_asof(
        left,
        right,
        left_on="query_date",
        right_on="trading_day",
        direction="forward",
        allow_exact_matches=False,   # strictly after -- see docstring
    )

    n_before = len(aligned)
    # Articles published after the final fixing have no future session to map to.
    aligned = aligned.dropna(subset=["trading_day"])
    dropped = n_before - len(aligned)

    lag = (aligned["trading_day"] - aligned["query_date"]).dt.days
    print("\nAlignment:")
    print(f"  articles aligned                  {len(aligned):>9,}")
    print(f"  dropped (past last fixing)        {dropped:>9,}")
    print(f"  publication -> fixing lag (days)  "
          f"min={lag.min()} median={lag.median():.0f} max={lag.max()}")
    print("  lag distribution:")
    for days, cnt in lag.value_counts().sort_index().head(6).items():
        print(f"      +{days}d : {cnt:>8,}")

    return aligned


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw GDELT articles ...")
    raw_articles = pd.read_csv(GDELT_RAW_CSV, dtype=str, low_memory=False)
    # Numeric columns come back as strings from dtype=str; restore them.
    for col in ["gdelt_tone", "gdelt_positive", "gdelt_negative",
                "gdelt_polarity", "gdelt_wordcount",
                "n_persons", "n_orgs", "n_locations"]:
        if col in raw_articles.columns:
            raw_articles[col] = pd.to_numeric(raw_articles[col], errors="coerce")

    print("Loading Bank Indonesia JISDOR export ...")
    rates = load_raw_export(BI_RAW_CSV)
    rates = rates[
        (rates["date"].dt.date >= STUDY_START) & (rates["date"].dt.date <= STUDY_END)
    ].reset_index(drop=True)
    rates.to_csv(RATES_CLEAN_CSV, index=False)
    print(f"  {len(rates):,} trading days "
          f"{rates['date'].min().date()} -> {rates['date'].max().date()}")

    articles = clean_articles(raw_articles)
    # Gzipped: the article-level tables are ~75 MB uncompressed, which is
    # awkward to version-control. pandas reads .gz transparently.
    articles.to_csv(ARTICLES_CLEAN_CSV, index=False, encoding="utf-8",
                    compression="gzip")

    aligned = align_to_trading_days(articles, rates)

    keep = [
        "query_date", "trading_day", "title", "title_clean", "title_norm",
        "url", "domain", "themes", "n_persons", "n_orgs", "n_locations",
        "gdelt_tone", "gdelt_positive", "gdelt_negative",
        "gdelt_polarity", "gdelt_wordcount", "n_tokens", "rate",
    ]
    keep = [c for c in keep if c in aligned.columns]
    aligned[keep].to_csv(ALIGNED_ARTICLES_CSV, index=False, encoding="utf-8",
                         compression="gzip")

    covered = aligned["trading_day"].nunique()
    print(f"\nWrote:")
    print(f"  {RATES_CLEAN_CSV}")
    print(f"  {ARTICLES_CLEAN_CSV}")
    print(f"  {ALIGNED_ARTICLES_CSV}")
    print(f"\nTrading days with >=1 aligned article: {covered:,} / {len(rates):,} "
          f"({100*covered/len(rates):.1f}%)")


if __name__ == "__main__":
    main()
