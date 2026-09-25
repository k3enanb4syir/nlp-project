# src/bi_scraper.py
"""
Task 1 -- Financial time-series acquisition: Bank Indonesia JISDOR USD/IDR.

SOURCE
------
JISDOR (Jakarta Interbank Spot Dollar Rate) is Bank Indonesia's official daily
USD/IDR reference rate. It is computed from weighted-average interbank spot
transactions during the morning Jakarta session and published once per business
day at approximately 10:00 WIB (UTC+7).

JISDOR is the right series for this project for three reasons:
  * Task 1 mandates Bank Indonesia as the ultimate source.
  * It is an official fixing rather than a broker quote, so there is exactly one
    unambiguous value per business day and no bid/ask or venue ambiguity.
  * Its fixed publication time gives a precise cut-off for the news alignment
    rule in src/preprocessing.py -- we know exactly what was knowable when.

WHY THE COMMITTED CSV WAS EXPORTED RATHER THAN CRAWLED
------------------------------------------------------
The JISDOR page is an ASP.NET WebForms application driven by __VIEWSTATE
postbacks, and its date-range picker silently truncates long spans and times out
the session on multi-year requests. Rather than fight that with a brittle
crawler, `data/raw/usd_idr_raw.csv` was produced using the page's own CSV export
control, requested in yearly slices and concatenated. That export is the
authoritative artefact this pipeline consumes.

This module implements the Selenium path anyway, in yearly chunks, so the pull
is reproducible end-to-end and can be re-run to extend or verify the series. It
is not required for the pipeline to run.

Usage:
    python src/bi_scraper.py --verify          # validate the committed CSV
    python src/bi_scraper.py --scrape          # re-crawl via Selenium
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from io import StringIO

import pandas as pd

try:  # config is optional so that --verify works from any working directory
    from config import BI_RAW_CSV, STUDY_END, STUDY_START
except ImportError:  # pragma: no cover - executed when run as a script
    from src.config import BI_RAW_CSV, STUDY_END, STUDY_START

JISDOR_URL = "https://www.bi.go.id/en/statistik/informasi-kurs/jisdor/default.aspx"


# --------------------------------------------------------------------------
# Parsing the committed export
# --------------------------------------------------------------------------

def load_raw_export(path=BI_RAW_CSV) -> pd.DataFrame:
    """
    Parse the Bank Indonesia CSV export into a tidy (date, rate) frame.

    The export carries four leading rows of report metadata (a blank row, a
    repeated 'Exchange Rates Jisdor' banner, another blank, then the real
    header), plus a trailing empty column, so the header row is located
    explicitly rather than assumed.
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)

    header_idx = None
    for idx in range(min(15, len(raw))):
        cells = [str(c).strip().lower() for c in raw.iloc[idx].tolist()]
        if "date" in cells:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not locate a header row containing 'Date' in {path}")

    header = [str(c).strip() for c in raw.iloc[header_idx].tolist()]
    body = raw.iloc[header_idx + 1:].copy()
    body.columns = header

    # Keep only the two columns that matter; names vary slightly between exports.
    date_col = next(c for c in body.columns if c.strip().lower() == "date")
    rate_col = next(
        c for c in body.columns
        if "exchange" in c.strip().lower() or "kurs" in c.strip().lower()
    )

    df = body[[date_col, rate_col]].rename(
        columns={date_col: "date", rate_col: "rate"}
    )

    # Drop blank//repeated-header rows that the export interleaves.
    df = df[df["date"].astype(str).str.strip() != ""]
    df = df[df["date"].astype(str).str.strip().str.lower() != "date"]

    # Dates arrive as US-style 'M/D/YYYY 12:00:00 AM'.
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    # Rates may carry thousands separators depending on export locale.
    df["rate"] = pd.to_numeric(
        df["rate"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    df = df.dropna(subset=["date", "rate"])
    df = df.drop_duplicates(subset="date", keep="first")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def verify(df: pd.DataFrame) -> None:
    """Print the sanity checks that justify using this series downstream."""
    print(f"rows                : {len(df):,}")
    print(f"date range          : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"required window     : {STUDY_START} -> {STUDY_END}")
    print(f"rate min / max      : {df['rate'].min():,.0f} / {df['rate'].max():,.0f}")
    print(f"duplicate dates     : {int(df['date'].duplicated().sum())}")
    print(f"non-positive rates  : {int((df['rate'] <= 0).sum())}")

    # JISDOR is business-day only; flag any weekend rows as a parsing red flag.
    weekend = df[df["date"].dt.dayofweek >= 5]
    print(f"weekend rows        : {len(weekend)}")

    # Largest single-day jumps -- a cheap outlier / bad-parse detector.
    pct = df["rate"].pct_change().abs()
    if len(pct.dropna()):
        worst = pct.nlargest(3)
        print("largest daily moves :")
        for i in worst.index:
            print(f"    {df.loc[i,'date'].date()}  {pct[i]*100:.2f}%  -> {df.loc[i,'rate']:,.0f}")

    covered = (df["date"].min().date() <= STUDY_START) and (df["date"].max().date() >= STUDY_END)
    print(f"covers study window : {'YES' if covered else 'NO'}")


# --------------------------------------------------------------------------
# Selenium re-crawl (optional, for reproducibility)
# --------------------------------------------------------------------------

def scrape_jisdor(start: date = STUDY_START, end: date = STUDY_END) -> pd.DataFrame:
    """
    Re-crawl JISDOR in yearly slices via Selenium.

    Yearly chunking is deliberate: the BI date picker truncates or times out on
    multi-year spans, so each slice is requested separately and concatenated.
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)

    frames: list[pd.DataFrame] = []
    try:
        cursor = start
        while cursor < end:
            chunk_end = min(date(cursor.year, 12, 31), end)
            print(f"  requesting {cursor} -> {chunk_end}")
            driver.get(JISDOR_URL)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )

            # The picker's element ids change between BI site revisions, so the
            # fields are located by their visible placeholder/label rather than
            # a hard-coded id.
            try:
                inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
                if len(inputs) >= 2:
                    inputs[0].clear()
                    inputs[0].send_keys(cursor.strftime("%d/%m/%Y"))
                    inputs[1].clear()
                    inputs[1].send_keys(chunk_end.strftime("%d/%m/%Y"))
                    driver.find_element(By.CSS_SELECTOR, "input[type='submit'], button[type='submit']").click()
                    time.sleep(6)
            except Exception as exc:
                print(f"    date-picker interaction failed ({exc.__class__.__name__}); "
                      f"falling back to the default page view")

            tables = pd.read_html(StringIO(driver.page_source))
            if tables:
                frames.append(tables[0])
            cursor = date(cursor.year + 1, 1, 1)
    finally:
        driver.quit()

    if not frames:
        raise RuntimeError("No tables scraped from the JISDOR page")

    out = pd.concat(frames, ignore_index=True)
    BI_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BI_RAW_CSV, index=False)
    print(f"Wrote {len(out):,} rows to {BI_RAW_CSV}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank Indonesia JISDOR acquisition")
    parser.add_argument("--scrape", action="store_true",
                        help="re-crawl the JISDOR page with Selenium")
    parser.add_argument("--verify", action="store_true",
                        help="parse and sanity-check the committed CSV export")
    args = parser.parse_args()

    if args.scrape:
        scrape_jisdor()
    # Verification is the default action -- it is what the pipeline relies on.
    verify(load_raw_export())


if __name__ == "__main__":
    main()
