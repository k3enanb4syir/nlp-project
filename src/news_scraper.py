# src/news_scraper.py
"""
Task 1 (revised) -- Geopolitical news acquisition from the GDELT Project.

WHY THIS REPLACES THE ORIGINAL BIGQUERY EVENTS PULL
---------------------------------------------------
The first iteration queried `gdelt-bq.gdeltv2.events` keeping only
(SQLDATE, SOURCEURL, EventCode, GoldsteinScale, AvgTone). Two fatal problems:

  1. NO TEXT. The GDELT *events* table stores coded event metadata, not article
     prose. There was nothing for a lexicon or a vectoriser to read, which makes
     Task 2's feature-extraction requirement impossible to satisfy.
  2. NO TEMPORAL COVERAGE. `LIMIT 50000` without an ORDER BY returned an
     arbitrary block of rows; coverage collapsed to 45 distinct calendar days
     out of the ~1827 in the study window.

WHY THE GKG FILES RATHER THAN THE DOC 2.0 API
----------------------------------------------
The DOC 2.0 API returns clean article titles and would have been the tidiest
source, but it is aggressively rate-limited: measured success rate was 2/6
requests at a 12s spacing and 0/6 at 16s, with 10-30s response latencies
indicating server-side saturation. Completing 1827 day-queries that way was not
feasible.

GDELT's Global Knowledge Graph (GKG) files are served as plain static objects
from data.gdeltproject.org with no rate limiting, and they are strictly richer:
each record carries the page title, a full theme taxonomy, a tone vector, and
extracted person/organisation entities. Measured throughput with 6 parallel
workers was ~11 MB/s, putting a full five-year pull at roughly 15 minutes.

SAMPLING DESIGN
---------------
GDELT publishes a GKG snapshot every 15 minutes (~96/day, ~5 MB each). Pulling
all of them for five years would be ~875 GB, so we sample ONE snapshot per day
at a fixed 12:00 UTC.

12:00 UTC is chosen deliberately: it is the window of maximum overlap between
the European trading afternoon and the US pre-market, i.e. the point in the
global day when FX-relevant reporting density is highest. Holding the sampling
time fixed keeps the daily sampling frame consistent, so day-to-day variation in
the extracted features reflects genuine variation in news content rather than
variation in when we happened to look.

Every calendar day is sampled -- including weekends and holidays -- because the
alignment rule in src/preprocessing.py deliberately rolls non-trading-day news
forward onto the next trading session.

FILTERING STRATEGY (Task 1, requirement 1c)
-------------------------------------------
Two independently justifiable stages:

  Stage 1 -- retrieval-time (this module). Keep only English-language articles
      whose GDELT theme annotations place them in a macro-financial or
      geopolitical frame (see THEME_FILTERS). This discards the bulk of GDELT's
      feed, which is local crime, sport, weather and celebrity coverage.

  Stage 2 -- post-hoc (src/preprocessing.py). Reputable-domain whitelisting,
      duplicate removal, and title quality heuristics.

Usage:
    python src/news_scraper.py                 # resume/continue the pull
    python src/news_scraper.py --workers 6
    python src/news_scraper.py --consolidate-only
    python src/news_scraper.py --probe 3       # yield check on N sample days
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import html
import io
import json
import re
import threading
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

try:
    from config import (
        GDELT_CACHE_DIR,
        GDELT_RAW_CSV,
        GDELT_SAMPLE_CSV,
        STUDY_END,
        STUDY_START,
    )
except ImportError:  # pragma: no cover
    from src.config import (
        GDELT_CACHE_DIR,
        GDELT_RAW_CSV,
        GDELT_SAMPLE_CSV,
        STUDY_END,
        STUDY_START,
    )

# GKG rows carry very large fields (the GCAM vector, the extras XML blob).
csv.field_size_limit(10**9)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# The study window (STUDY_START / STUDY_END) and every output path come from
# src/config.py, so the scraper cannot drift out of step with the rest of the
# pipeline.

# Fixed daily sampling time (UTC). See "SAMPLING DESIGN" above.
SAMPLE_TIME = "120000"
# If the primary snapshot is missing (GDELT outages leave occasional holes),
# fall back to the neighbouring quarter-hours before giving up on the day.
FALLBACK_TIMES = ["114500", "121500", "113000", "123000"]

BASE_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NLP-coursework-research/1.0)"}

# GKG v2.1 column layout (27 tab-separated fields).
COL_DATE = 1
COL_DOMAIN = 3
COL_URL = 4
COL_THEMES = 7
COL_LOCATIONS = 9
COL_PERSONS = 11
COL_ORGS = 13
COL_TONE = 15
COL_ALLNAMES = 23
COL_EXTRAS = 26
GKG_NCOLS = 27

_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)

# --- Stage-1 relevance filter -------------------------------------------------
# An article is kept if ANY of its GDELT themes matches one of the rules below.
#
# PRECISION MATTERS MORE THAN RECALL HERE. A first pass keyed on the bare
# `ECON_` and `EPU_` prefixes retained 440-1100 articles/day, but inspection
# showed it was dominated by noise: the generic tags EPU_POLICY (545 hits in a
# single snapshot), EPU_POLICY_GOVERNMENT and EPU_ECONOMY_HISTORIC fire on
# almost any article that mentions a government or the economy in passing, so
# school-board and celebrity stories were being swept in. The filter below
# therefore enumerates specific high-precision themes instead of whole families.
#
# Three transmission channels from global geopolitics to the USD are targeted:
#   (a) macro-financial conditions, (b) economic policy uncertainty,
#   (c) armed conflict and coercive statecraft.

# Exact theme codes -- monetary, currency and fiscal conditions.
THEME_EXACT = frozenset({
    # Currency / FX -- the most directly relevant family
    "ECON_CURRENCY_EXCHANGE_RATE",
    "ECON_WORLDCURRENCIES",
    "ECON_WORLDCURRENCIES_DOLLAR",
    "ECON_WORLDCURRENCIES_DOLLARS",
    "ECON_WORLDCURRENCIES_EURO",
    "ECON_WORLDCURRENCIES_YEN",
    "ECON_WORLDCURRENCIES_YUAN",
    "ECON_WORLDCURRENCIES_RUPIAH",
    # Monetary policy
    "ECON_CENTRALBANK",
    "ECON_INTEREST_RATES",
    "ECON_INFLATION",
    "EPU_CATS_MONETARY_POLICY",
    "EPU_POLICY_INTEREST_RATE",
    "EPU_POLICY_INTEREST_RATES",
    # Fiscal / sovereign credit
    "ECON_DEBT",
    "ECON_BUDGET_DEFICIT",
    "ECON_TAXATION",
    "EPU_CATS_FISCAL_POLICY",
    "EPU_POLICY_DEFICIT",
    "EPU_POLICY_BUDGET",
    # Markets & commodity prices that co-move with the dollar
    "ECON_STOCKMARKET",
    "ECON_OILPRICE",
    "ECON_GOLDPRICE",
    "ECON_GASOLINEPRICE",
    "ECON_COST_OF_LIVING",
    # Trade policy
    "ECON_FREETRADE",
    "ECON_TRADE_DISPUTE",
    "ECON_BOYCOTT",
    "ECON_SUBSIDIES",
    "ECON_FOREIGNINVEST",
    "ECON_EMERGINGECON",
    # Explicit uncertainty / systemic risk
    "EPU_UNCERTAINTY",
    "EPU_CATS_FINANCIAL_REGULATION",
    "EPU_CATS_TRADE_POLICY",
    "EPU_CATS_NATIONAL_SECURITY",
    "EPU_CATS_SOVEREIGN_DEBT_CURRENCY_CRISES",
    "ECON_BANKRUPTCY",
    "ECON_IMFBAILOUT",
})

# Substring rules -- coercive statecraft and armed conflict. These theme
# families have many suffixed variants, so matching on a fragment is the
# practical way to catch them without enumerating every code.
THEME_SUBSTRINGS = (
    "ARMEDCONFLICT",
    "SANCTION",
    "TERROR",
    "NUCLEAR",
    "BLOCKADE",
    "WMD",
    "MILITARY_",            # trailing underscore avoids bare 'MILITARY' tags
    "FRAGILITY_CONFLICT_AND_VIOLENCE",
    "TRADE_DISPUTE",
    "DIPLOM",
    "PEACEKEEPING",
)


def is_relevant(themes: str) -> bool:
    """Stage-1 relevance test against the GDELT V1THEMES field."""
    if not themes:
        return False
    for theme in themes.split(";"):
        if not theme:
            continue
        if theme in THEME_EXACT:
            return True
        for frag in THEME_SUBSTRINGS:
            if frag in theme:
                return True
    return False


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _clean_title(raw: str) -> str:
    """
    Decode the HTML entities GDELT leaves in PAGE_TITLE and normalise whitespace.

    `html.unescape` is used rather than hand-rolled regexes because GDELT
    occasionally emits out-of-range numeric character references (e.g. from
    mojibake on the source page). Decoding those manually with chr() raises
    ValueError; html.unescape leaves them as literal text instead.
    """
    if not raw:
        return ""
    return " ".join(html.unescape(raw).split())


def parse_gkg(content: bytes, query_date: str) -> list[dict]:
    """
    Decompress one GKG snapshot and return the relevant article records.

    Filtering happens here rather than after writing, so the ~5 MB download is
    reduced to a few hundred rows before anything touches disk.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return []

    text = archive.read(archive.namelist()[0]).decode("utf-8", "replace")
    records: list[dict] = []

    for line in text.split("\n"):
        # Cheap structural guard: skip truncated/malformed rows.
        if line.count("\t") < GKG_NCOLS - 1:
            continue
        row = line.split("\t")

        themes = row[COL_THEMES]
        if not is_relevant(themes):
            continue

        match = _TITLE_RE.search(row[COL_EXTRAS])
        if not match:
            # Without a title there is no text to extract features from.
            continue
        title = _clean_title(match.group(1))
        if not title:
            continue

        # V1.5TONE is a comma-separated vector:
        #   tone, positive, negative, polarity, activity_density,
        #   self_group_density, word_count
        tone_parts = row[COL_TONE].split(",")
        def _f(idx: int):
            try:
                return float(tone_parts[idx])
            except (IndexError, ValueError):
                return None

        # Entity fields are stored as COUNTS rather than as the raw
        # semicolon-delimited strings. Downstream code only ever uses their
        # cardinality, and the raw strings are by far the largest fields in a
        # GKG row -- keeping them inflated the consolidated CSV past 1.3 GB.
        def _n(value: str) -> int:
            return len([p for p in value.split(";") if p.strip()]) if value else 0

        records.append(
            {
                "query_date": query_date,
                "gkg_datetime": row[COL_DATE],
                "title": title,
                "url": row[COL_URL],
                "domain": row[COL_DOMAIN],
                "themes": themes,
                "n_persons": _n(row[COL_PERSONS]),
                "n_orgs": _n(row[COL_ORGS]),
                "n_locations": _n(row[COL_LOCATIONS]),
                # GDELT's own tone metrics. Retained as a documented comparison
                # baseline -- the NLP features in src/features.py are computed
                # independently from the title text.
                "gdelt_tone": _f(0),
                "gdelt_positive": _f(1),
                "gdelt_negative": _f(2),
                "gdelt_polarity": _f(3),
                "gdelt_wordcount": _f(6),
            }
        )

    return records


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def _daterange(start: date, end: date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def fetch_day(session: requests.Session, day: date) -> list[dict] | None:
    """
    Download and parse the 12:00 UTC GKG snapshot for `day`.

    Returns parsed records, or None if every candidate snapshot 404s (which we
    treat as a genuine GDELT coverage hole rather than a transient error).
    """
    stamp_day = day.strftime("%Y%m%d")
    for time_part in [SAMPLE_TIME] + FALLBACK_TIMES:
        url = BASE_URL.format(stamp=f"{stamp_day}{time_part}")
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=180)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 404:
                break            # this quarter-hour genuinely absent; try next
            if resp.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue
            return parse_gkg(resp.content, day.isoformat())
    return None


_print_lock = threading.Lock()


def scrape_all(workers: int) -> None:
    """Download every day's snapshot in parallel, caching each day as JSON."""
    GDELT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_days = list(_daterange(STUDY_START, STUDY_END))
    pending = [d for d in all_days if not (GDELT_CACHE_DIR / f"{d.isoformat()}.json").exists()]

    print(f"Study window  : {STUDY_START} -> {STUDY_END} ({len(all_days)} days)")
    print(f"Already cached: {len(all_days) - len(pending)}")
    print(f"To fetch      : {len(pending)}  (workers={workers})")
    if not pending:
        print("Nothing to fetch; cache is complete.")
        return

    started = time.time()
    done = 0
    missing: list[str] = []

    # One Session per worker thread: requests.Session is not thread-safe.
    local = threading.local()

    def session_for_thread() -> requests.Session:
        if not hasattr(local, "session"):
            s = requests.Session()
            s.headers.update(HEADERS)
            local.session = s
        return local.session

    def work(day: date):
        target = GDELT_CACHE_DIR / f"{day.isoformat()}.json"
        try:
            records = fetch_day(session_for_thread(), day)
        except Exception as exc:
            # A single corrupt snapshot must not abort the whole run. The day is
            # left uncached so that a re-run retries it.
            with _print_lock:
                print(f"  {day}: ERROR {exc.__class__.__name__}: {exc}", flush=True)
            return day, None
        if records is None:
            # Cache the hole too, so a resume does not retry it forever.
            target.write_text("[]", encoding="utf-8")
            return day, None
        target.write_text(json.dumps(records), encoding="utf-8")
        return day, len(records)

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for day, count in pool.map(work, pending):
            done += 1
            if count is None:
                missing.append(day.isoformat())
            if done % 50 == 0 or done == 1:
                elapsed = time.time() - started
                eta = (len(pending) - done) * (elapsed / done) / 60
                with _print_lock:
                    print(
                        f"[{done}/{len(pending)}] {day} -> "
                        f"{'MISSING' if count is None else str(count) + ' articles'} "
                        f"| {elapsed/done:.2f}s/day | ETA {eta:.0f} min",
                        flush=True,
                    )

    print(f"Fetch loop complete in {(time.time()-started)/60:.1f} min.")
    if missing:
        print(f"  {len(missing)} day(s) had no retrievable snapshot: {missing[:10]}")


# --------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------

def _normalise(rec: dict) -> dict:
    """
    Coerce one cached record into the compact consolidated schema.

    Early cache files stored the raw semicolon-delimited person/organisation/
    location strings; the current scraper stores their counts directly. This
    accepts either shape so the cache does not have to be re-downloaded.
    """
    def _n(key_count: str, key_str: str) -> int:
        if key_count in rec:
            try:
                return int(rec[key_count] or 0)
            except (TypeError, ValueError):
                return 0
        raw = rec.get(key_str) or ""
        return len([p for p in str(raw).split(";") if p.strip()])

    return {
        "query_date": rec.get("query_date"),
        "gkg_datetime": rec.get("gkg_datetime"),
        "title": rec.get("title"),
        "url": rec.get("url"),
        "domain": rec.get("domain"),
        "themes": rec.get("themes"),
        "n_persons": _n("n_persons", "persons"),
        "n_orgs": _n("n_orgs", "organizations"),
        "n_locations": _n("n_locations", "locations"),
        "gdelt_tone": rec.get("gdelt_tone"),
        "gdelt_positive": rec.get("gdelt_positive"),
        "gdelt_negative": rec.get("gdelt_negative"),
        "gdelt_polarity": rec.get("gdelt_polarity"),
        "gdelt_wordcount": rec.get("gdelt_wordcount"),
    }


def consolidate() -> pd.DataFrame:
    """
    Merge the per-day JSON cache into a single gzip-compressed raw CSV.

    Only the Stage-1 theme filter has been applied at this point; deduplication,
    domain whitelisting and text normalisation belong to src/preprocessing.py.

    The output is gzipped and carries entity COUNTS rather than raw entity
    strings. An uncompressed full-fidelity dump of this corpus is ~1.3 GB, which
    is neither committable nor comfortable on a laptop; the compact form is a
    small fraction of that and loses nothing the pipeline consumes.

    Days are streamed and appended in chunks rather than concatenated in memory,
    because holding ~1M rows as a single DataFrame alongside the per-day frames
    is what pushes a modest machine into swap.
    """
    if not GDELT_CACHE_DIR.exists():
        raise SystemExit(f"No cache directory at {GDELT_CACHE_DIR}; run the scraper first.")

    files = sorted(GDELT_CACHE_DIR.glob("*.json"))
    if not files:
        raise SystemExit("Cache is empty; run the scraper first.")

    GDELT_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if GDELT_RAW_CSV.exists():
        GDELT_RAW_CSV.unlink()

    total = 0
    empty_days = 0
    wrote_header = False
    buffer: list[dict] = []
    sample_rows: list[dict] = []

    def flush(rows: list[dict]) -> None:
        nonlocal wrote_header
        if not rows:
            return
        pd.DataFrame.from_records(rows).to_csv(
            GDELT_RAW_CSV,
            index=False,
            encoding="utf-8",
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            compression="gzip",
        )
        wrote_header = True

    for path in files:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            print(f"  warning: unreadable cache file {path.name}, skipping")
            continue
        if not records:
            empty_days += 1
            continue

        normalised = [_normalise(r) for r in records]
        buffer.extend(normalised)
        sample_rows.extend(normalised[:2])
        total += len(normalised)

        if len(buffer) >= 100_000:
            flush(buffer)
            buffer = []

    flush(buffer)

    # Task 1 asks for "sample raw data" in the repo; the full pull is large and
    # gitignored, so a small stratified sample is committed alongside it.
    pd.DataFrame.from_records(sample_rows[:3000]).to_csv(
        GDELT_SAMPLE_CSV, index=False, encoding="utf-8"
    )

    size_mb = GDELT_RAW_CSV.stat().st_size / 1e6
    print(f"Consolidated {len(files)} cached days -> {total:,} article rows")
    print(f"  days with zero relevant articles: {empty_days}")
    print(f"  full raw : {GDELT_RAW_CSV}  ({size_mb:.0f} MB gzipped)")
    print(f"  sample   : {GDELT_SAMPLE_CSV}")
    return pd.DataFrame.from_records(sample_rows[:100])


def probe(n_days: int) -> None:
    """Fetch a few spread-out days and report filter yield, without caching."""
    session = requests.Session()
    session.headers.update(HEADERS)
    span = (STUDY_END - STUDY_START).days
    picks = [STUDY_START + timedelta(days=int(span * i / max(n_days - 1, 1))) for i in range(n_days)]
    for day in picks:
        t0 = time.time()
        recs = fetch_day(session, day)
        if recs is None:
            print(f"{day}: MISSING snapshot")
            continue
        titles = [r["title"] for r in recs[:3]]
        print(f"{day}: {len(recs):4d} relevant articles in {time.time()-t0:.1f}s")
        for t in titles:
            print(f"      - {t[:95]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GDELT GKG geopolitical news scraper")
    parser.add_argument("--workers", type=int, default=6,
                        help="parallel download workers (default 6)")
    parser.add_argument("--consolidate-only", action="store_true",
                        help="skip fetching; rebuild the raw CSV from the day cache")
    parser.add_argument("--probe", type=int, metavar="N",
                        help="yield-check N sample days and exit")
    args = parser.parse_args()

    if args.probe:
        probe(args.probe)
        return
    if not args.consolidate_only:
        scrape_all(args.workers)
    consolidate()


if __name__ == "__main__":
    main()
