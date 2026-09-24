# Global Geopolitical Event Prediction: Impact on Dollar Exchange Rates

End-to-end NLP and analytical pipeline testing the hypothesis that **global
geopolitical news significantly influences and helps predict fluctuations in the
US Dollar exchange rate**, using the USD/IDR JISDOR reference rate.

Covers **Task 1** (data acquisition, preprocessing and temporal alignment) and
**Task 2** (NLP feature extraction, baseline modelling and evaluation).

## Project Team (Team: Miguel!)

* **Muhammad Keenan Basyir**
* **Muhammad Gibran Basyir**
* **Thomas Nadandra Aryawida**

---

## Headline result

Across a strictly chronological five-year split, **no model beat the
majority-class baseline on test directional accuracy** (best 0.568 vs a 0.580
baseline). On balanced metrics several models showed weak but non-zero skill
(best MCC 0.077). Adding NLP features to the rate-only baseline did not produce a
reliable improvement.

This is the expected regime for daily FX direction, and establishing it with a
leakage-free pipeline — one that would have surfaced a strong signal had one
existed — is the intended outcome of a baseline task. Full discussion in
[notebook/02_baseline_results.ipynb](notebook/02_baseline_results.ipynb).

---

## Data sources

1. **The GDELT Project** — Global Knowledge Graph (GKG) v2.1 files. Each record
   carries the article title, a theme taxonomy, a tone vector and extracted
   person / organisation / location entities.
2. **Bank Indonesia** — daily JISDOR USD/IDR reference rate, the official fixing
   published each business day at ~10:00 WIB.

Study window: **1 September 2021 – 1 September 2026** (1,202 trading days).

### Why GKG files rather than the GDELT DOC API

The DOC 2.0 API returns clean titles but is aggressively rate-limited — measured
success was 2/6 requests at 12s spacing and 0/6 at 16s, with 10–30s latencies.
1,827 day-queries were not feasible that way. The GKG files are served as static
objects with no rate limiting and are strictly richer. Measured throughput with 6
parallel workers was ~11 MB/s.

### Sampling design

GDELT publishes a GKG snapshot every 15 minutes (~96/day, ~5 MB each); five years
of all of them is ~875 GB. We sample **one snapshot per day at 12:00 UTC** — the
window of maximum overlap between the European trading afternoon and the US
pre-market. Holding the sampling time fixed keeps the daily frame consistent, so
day-to-day feature variation reflects news content rather than when we looked.

---

## Pipeline

```text
   GDELT GKG (15-min files)            Bank Indonesia JISDOR
   1 snapshot/day @ 12:00 UTC          daily business-day fixing
            │                                    │
            ▼                                    ▼
   src/news_scraper.py                   src/bi_scraper.py
   • theme relevance filter              • parse CSV export
   • title / tone / entity extract       • validate coverage
            │                                    │
            └──────────────┬─────────────────────┘
                           ▼
                  src/preprocessing.py
      • reputable-publisher whitelist
      • de-duplication (URL + syndication)
      • boilerplate stripping, text normalisation
      • TEMPORAL ALIGNMENT: article → next trading day strictly after
                           │
                           ▼
                    src/features.py
      • VADER sentiment aggregates
      • LM-style financial lexicon rates
      • GDELT theme-family shares, entity density
      • lagged returns, rolling volatility & momentum
                           │
                           ▼
                    src/dataset.py           ← strict chronological split
                           │
                           ▼
              src/models.py → src/evaluate.py
```

### Temporal alignment rule (Task 1, requirement 4)

The GDELT snapshot is taken at 12:00 UTC = **19:00 WIB**, always *after* that
day's fixing has been published at ~10:00 WIB. No article in the corpus could
have influenced the rate printed on its own calendar date.

> **Every article is aligned to the first trading day strictly after its
> publication date.**

| Published | Aligned to |
|---|---|
| Monday | Tuesday's fixing |
| Friday | Monday's fixing |
| Saturday / Sunday | Monday's fixing |
| Indonesian public holiday | next open session |

Equivalently, the fixing on day *t* absorbs all news from the previous trading
day through the calendar day before *t*. The trading calendar is taken from the
JISDOR series itself, so Indonesian holidays are handled exactly.

---

## Results

Test block (2025-12-01 → 2026-09-01, 176 sessions; 58.0% UP days):

| model | accuracy | macro-F1 | MCC |
|---|---|---|---|
| majority (constant UP) | **0.580** | 0.367 | 0.000 |
| logreg_rates | 0.568 | 0.498 | 0.053 |
| arima(1,0,1) | 0.557 | 0.533 | 0.071 |
| persistence | 0.551 | **0.539** | **0.077** |
| xgb_rates | 0.551 | 0.503 | 0.032 |
| logreg_combined | 0.545 | 0.532 | 0.064 |
| xgb_combined | 0.540 | 0.516 | 0.036 |
| logreg_news | 0.517 | 0.504 | 0.007 |
| logreg_tfidf | 0.472 | 0.441 | −0.112 |

Accuracy alone is misleading — the constant predictor tops it while having zero
discriminative power (MCC 0). MCC and macro-F1 are the metrics that separate the
models.

---

## Repository structure

```text
nlp-project/
├── data/
│   ├── raw/
│   │   ├── usd_idr_raw.csv              # Bank Indonesia JISDOR export
│   │   ├── sample_gdelt_articles.csv    # committed sample of the raw pull
│   │   ├── gdelt_articles_raw.csv.gz    # full pull (gitignored, ~252 MB)
│   │   └── gdelt_daily/                 # per-day scrape cache (gitignored)
│   ├── cleaned/
│   │   ├── usd_idr_clean.csv            # tidy trading calendar + rate
│   │   ├── articles_clean.csv.gz        # filtered, normalised articles (gitignored)
│   │   ├── aligned_dataset.csv.gz       # article-level, aligned to trading day
│   │   └── daily_dataset.csv            # one row per trading day + target
│   └── splits/
│       ├── train.csv  validation.csv  test.csv
├── src/
│   ├── config.py           # paths, split boundaries, constants
│   ├── bi_scraper.py       # Bank Indonesia acquisition + validation
│   ├── news_scraper.py     # GDELT GKG scraper + theme filter
│   ├── preprocessing.py    # cleaning, filtering, temporal alignment
│   ├── features.py         # VADER, lexicons, themes, rate features
│   ├── dataset.py          # chronological train/val/test split
│   ├── models.py           # baseline + combined model ladder
│   ├── evaluate.py         # metrics and reporting
│   └── viz.py              # shared plotting style
├── notebook/
│   ├── 01_eda.ipynb                # exploratory data analysis
│   └── 02_baseline_results.ipynb   # experimental results
├── scripts/make_notebooks.py       # regenerates the notebooks
├── results/
│   ├── metrics.csv
│   └── test_predictions.csv
└── requirements.txt
```

---

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

python src/news_scraper.py       # ~15 min, ~9 GB downloaded, 252 MB retained
python src/bi_scraper.py         # validates the committed JISDOR export
python src/preprocessing.py      # cleaning + temporal alignment
python src/features.py           # NLP + rate feature extraction
python src/dataset.py            # chronological split
python src/models.py             # train, evaluate, write results/

python scripts/make_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebook/*.ipynb
```

The scraper is resumable — re-running it fetches only days missing from the
cache.

---

## Method constraints

Per the Task 2 specification, **no pre-trained embeddings or transformer models**
(FinBERT, RoBERTa, LLM embeddings) are used anywhere. All text features are
rule-based lexicon counts or classical TF-IDF / bag-of-words statistics estimated
from the training split only.

## Leakage discipline

* News is aligned strictly *after* publication date — never to its own fixing.
* All rate predictors are lagged by at least one trading day.
* Imputer medians, scaler statistics, the TF-IDF vocabulary and IDF weights, and
  ARIMA coefficients are fitted on the **training block only**.
* Hyperparameters are chosen on validation; the test block is scored once.
* `src/dataset.py` asserts that the three blocks do not overlap in time.

## Key numbers

| | |
|---|---|
| Calendar days scraped | 1,827 |
| Raw articles after theme filter | 926,215 |
| After publisher whitelist + de-duplication | 53,887 |
| Aligned to a trading day | 53,843 |
| Trading-day news coverage | 1,190 / 1,202 (99.0%) |
| Mean articles per trading session | 45.2 |
| Distinct publishers | 76 |
