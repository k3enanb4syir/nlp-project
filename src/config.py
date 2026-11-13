# src/config.py
"""
Central configuration for the Task 1 / Task 2 pipeline.

Every path and modelling constant lives here so that the scraper, preprocessing,
feature-extraction, training and evaluation modules agree on a single source of
truth, and so that the chronological split boundaries are stated in exactly one
place (they are the thing most likely to be accidentally violated).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

# Raw inputs
GDELT_CACHE_DIR = RAW_DIR / "gdelt_daily"
GDELT_RAW_CSV = RAW_DIR / "gdelt_articles_raw.csv.gz"
GDELT_SAMPLE_CSV = RAW_DIR / "sample_gdelt_articles.csv"
BI_RAW_CSV = RAW_DIR / "usd_idr_raw.csv"

# Cleaned / derived artefacts
ARTICLES_CLEAN_CSV = CLEANED_DIR / "articles_clean.csv.gz"
RATES_CLEAN_CSV = CLEANED_DIR / "usd_idr_clean.csv"
ALIGNED_ARTICLES_CSV = CLEANED_DIR / "aligned_dataset.csv.gz"
DAILY_DATASET_CSV = CLEANED_DIR / "daily_dataset.csv"

# Modelling tables (chronological split)
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "validation.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

# --------------------------------------------------------------------------
# Study window (Task 1 requirement 2a)
# --------------------------------------------------------------------------
STUDY_START = date(2021, 9, 1)
STUDY_END = date(2026, 9, 1)

# --------------------------------------------------------------------------
# Temporal alignment (Task 1 requirement 4)
# --------------------------------------------------------------------------
# Bank Indonesia publishes the JISDOR USD/IDR reference rate once per business
# day, fixed from interbank transactions in the morning Jakarta session and
# released at approximately 10:00 Western Indonesia Time (WIB = UTC+7).
BI_FIXING_HOUR_WIB = 10
WIB_UTC_OFFSET_HOURS = 7

# Our GDELT sampling snapshot is 12:00 UTC == 19:00 WIB, i.e. always AFTER the
# same day's fixing has already been published. Consequently no article in this
# corpus can have influenced the fixing printed on its own calendar date, and
# every article is aligned to the NEXT trading day strictly after its
# publication date. See src/preprocessing.py for the implementation.
GDELT_SAMPLE_HOUR_UTC = 12

# --------------------------------------------------------------------------
# Chronological split boundaries (Task 2 requirement 3a)
# --------------------------------------------------------------------------
# Strictly ordered in time, no shuffling, no overlap: the validation block sits
# entirely after the training block and the test block entirely after that, so
# no future information can reach a model through either tuning or evaluation.
# Roughly a 70 / 15 / 15 split of the five-year window.
TRAIN_END = date(2025, 2, 28)     # train: STUDY_START .. TRAIN_END
VAL_END = date(2025, 11, 30)      # val  : TRAIN_END+1 .. VAL_END
                                  # test : VAL_END+1  .. STUDY_END

# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
# Number of lagged exchange-rate returns fed to the models.
RATE_LAGS = (1, 2, 3, 5, 10)
# Rolling windows (trading days) for realised volatility / momentum features.
ROLLING_WINDOWS = (5, 10, 21)

# TF-IDF over the daily concatenation of article titles.
TFIDF_MAX_FEATURES = 300
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.85
TFIDF_NGRAM_RANGE = (1, 2)

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Task 3 -- headline embeddings (src/embeddings.py)
# --------------------------------------------------------------------------
# Cached per-article embedding matrices. Regenerable and ~80 MB each for the
# transformer encoders, so the directory is gitignored.
EMBED_DIR = DATA_DIR / "embeddings"

# Static, general-purpose: pre-trained GloVe (Wikipedia + Gigaword, 2014).
GLOVE_MODEL = "glove-wiki-gigaword-300"
# Static, in-domain: FastText trained on TRAINING-PERIOD headlines only.
# Small corpus (~42k headlines), hence a modest dimension.
FASTTEXT_DIM = 100
FASTTEXT_EPOCHS = 20
# Smooth-inverse-frequency weight a / (a + p(w)), Arora et al. (2017).
SIF_A = 1e-3

# Contextual, domain-specific: BERT further pre-trained on Reuters TRC2 and
# fine-tuned on Financial PhraseBank sentiment.
FINBERT_MODEL = "ProsusAI/finbert"
# Contextual, general-purpose: sentence-transformer trained with a contrastive
# objective, so cosine similarity between outputs is meaningful.
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Headlines average 11 words; a few run past 100 tokens, so 128 truncates
# almost nothing. The count actually truncated is logged at encode time.
TRANSFORMER_MAX_LENGTH = 128
EMBED_BATCH_SIZE = 64

# --------------------------------------------------------------------------
# Task 3 -- channel axes, novelty and daily pooling (src/semantic_axes.py)
# --------------------------------------------------------------------------
# One small CSV per encoder, one row per trading session. Small and costly to
# regenerate without a GPU, so these ARE committed.
TEXT_FEATURES_DIR = CLEANED_DIR / "text_features"
# Per session and channel, only the K headlines most relevant to the channel's
# topic contribute to its direction score (~10% of a typical 45-headline day).
AXIS_TOP_K = 5
# Headline novelty compares each headline with those of the previous N sessions.
NOVELTY_LOOKBACK_SESSIONS = 5
# Half-life (sessions) of the decayed memory used by centroid novelty.
NOVELTY_MEMORY_HALFLIFE = 2.0
# Anchor split-half reliability: number of random 5-vs-5 splits.
SPLIT_HALF_REPEATS = 50
