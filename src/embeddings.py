# src/embeddings.py
"""
Task 3, step 1 -- headline embeddings from four text encoders.

Every aligned article title is embedded once with each encoder and cached to
data/embeddings/<encoder>.npy, row-aligned with data/cleaned/aligned_dataset.
Later steps (semantic axes, novelty, UMAP) load these matrices instead of
re-running the models. The same encoder objects are also used to embed the
anchor sentences in step 2, so headlines and anchors always pass through an
identical tokenise -> encode -> pool path.

THE FOUR ENCODERS
-----------------
They form a 2 x 2 design: static vs contextual, general vs domain.

                  general-purpose                 domain-specific
  static          glove    (GloVe 300d, SIF)      fasttext (trained on our
                                                            train-period headlines)
  contextual      sbert    (all-mpnet-base-v2)    finbert  (ProsusAI/finbert)

glove
    Pre-trained 300d GloVe vectors (Wikipedia + Gigaword). Titles are lowercased
    and stripped of punctuation (preprocessing.normalise_title), split on
    whitespace, and pooled with SIF (below). Words outside the 400k vocabulary
    are skipped.

fasttext
    A FastText skip-gram model trained here, on TRAINING-PERIOD titles only, so
    no validation- or test-period text shapes the vectors. FastText builds each
    word vector from character n-grams (3-6), so it still produces a vector for
    names that first appear in the test period. Pooled with SIF like GloVe.

finbert
    BERT-base further pre-trained on Reuters financial news and fine-tuned for
    3-class financial sentiment. WordPiece tokenisation; the sentence vector is
    the MEAN of the last hidden layer over non-padding tokens. We do not use the
    [CLS] vector: in a classification-tuned model [CLS] is specialised for the
    sentiment head and is a poor general representation of the sentence. The
    model's own sentiment probabilities are saved alongside as a separate,
    3-column feature set.

sbert
    all-mpnet-base-v2, a sentence-transformer trained with a contrastive
    objective on over a billion sentence pairs. Mean pooling over tokens, then
    L2 normalisation, which is the recipe that model was trained with. Unlike
    FinBERT, cosine similarity between its outputs is what the training
    objective optimised, so it is the reference encoder for the axis method.

SIF POOLING FOR THE STATIC ENCODERS
-----------------------------------
A plain average of word vectors is dominated by frequent words ("the", "to",
"says"). Smooth inverse frequency (Arora, Liang and Ma, 2017) fixes this in
two steps:
  1. weight each word by a / (a + p(w)), where p(w) is the word's relative
     frequency, so frequent words count less;
  2. remove each sentence vector's projection onto the first principal
     component of all sentence vectors, which mostly encodes word frequency and
     syntax shared by every sentence.
Both p(w) and the principal component are estimated on TRAINING-PERIOD titles
only. A word never seen in training gets p(w) = 0 and therefore the maximum
weight of 1: it is rare by definition.

LEAKAGE DISCIPLINE
------------------
  * Pre-trained encoders are frozen. They never see exchange-rate data, so no
    label information reaches the vectors. All four were published before the
    test window begins (December 2025).
  * Everything estimated from our own corpus (FastText weights, SIF word
    frequencies, SIF principal component) uses training-period articles only.
  * Embedding an article uses only that article's title, so every row can be
    computed independently of its neighbours in time.

CACHE LAYOUT
------------
data/embeddings/
    glove.npy, fasttext.npy, finbert.npy, sbert.npy   (n_articles x dim, float16)
    finbert_sentiment.npy                              (n_articles x 3, float32)
    fasttext_train.model                               (trained FastText model)
    manifest.json                                      (model ids, dims, timings,
                                                        row fingerprint)
The manifest stores a fingerprint of the article URLs in row order. Loading a
cache whose fingerprint no longer matches the aligned dataset raises an error
instead of silently returning misaligned rows.

Usage:
    python src/embeddings.py                       # all four encoders
    python src/embeddings.py --encoders sbert glove
    python src/embeddings.py --force               # ignore existing caches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

try:
    from config import (
        ALIGNED_ARTICLES_CSV,
        EMBED_BATCH_SIZE,
        EMBED_DIR,
        FASTTEXT_DIM,
        FASTTEXT_EPOCHS,
        FINBERT_MODEL,
        GLOVE_MODEL,
        RANDOM_SEED,
        SBERT_MODEL,
        SIF_A,
        TRAIN_END,
        TRANSFORMER_MAX_LENGTH,
    )
    from preprocessing import clean_title, normalise_title
except ImportError:  # pragma: no cover
    from src.config import (
        ALIGNED_ARTICLES_CSV,
        EMBED_BATCH_SIZE,
        EMBED_DIR,
        FASTTEXT_DIM,
        FASTTEXT_EPOCHS,
        FINBERT_MODEL,
        GLOVE_MODEL,
        RANDOM_SEED,
        SBERT_MODEL,
        SIF_A,
        TRAIN_END,
        TRANSFORMER_MAX_LENGTH,
    )
    from src.preprocessing import clean_title, normalise_title


ENCODERS = ("glove", "fasttext", "finbert", "sbert")
MANIFEST = EMBED_DIR / "manifest.json"
FASTTEXT_PATH = EMBED_DIR / "fasttext_train.model"
FINBERT_SENTIMENT_PATH = EMBED_DIR / "finbert_sentiment.npy"


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def load_articles() -> pd.DataFrame:
    """Aligned articles in their canonical row order (the order of the cache)."""
    df = pd.read_csv(ALIGNED_ARTICLES_CSV, low_memory=False)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def train_mask(articles: pd.DataFrame) -> np.ndarray:
    """True for articles aligned to a TRAINING-period session."""
    return (articles["trading_day"] <= pd.Timestamp(TRAIN_END)).to_numpy()


def fingerprint(articles: pd.DataFrame) -> str:
    """Hash of the article URLs in row order; detects a stale or reordered cache."""
    joined = "\n".join(articles["url"].astype(str))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    """Static-encoder tokenisation: identical to the title_norm column."""
    return normalise_title(clean_title(text)).split()


# --------------------------------------------------------------------------
# Static encoders: word vectors + SIF pooling
# --------------------------------------------------------------------------

class SIFEncoder:
    """Average word vectors with SIF weights, then remove the first component.

    `lookup` maps a word to its vector, or to None if the encoder has no vector
    for it. `fit` must be called on training-period titles before `encode`.
    """

    def __init__(self, name: str, lookup: Callable[[str], np.ndarray | None],
                 dim: int, a: float = SIF_A):
        self.name = name
        self.lookup = lookup
        self.dim = dim
        self.a = a
        self.word_prob: dict[str, float] = {}
        self.pc: np.ndarray | None = None

    def _weighted_average(self, texts: list[str]) -> tuple[np.ndarray, int]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        n_empty = 0
        for i, text in enumerate(texts):
            vecs, weights = [], []
            for w in _tokens(text):
                v = self.lookup(w)
                if v is None:
                    continue
                vecs.append(v)
                weights.append(self.a / (self.a + self.word_prob.get(w, 0.0)))
            if vecs:
                out[i] = np.average(np.asarray(vecs), axis=0, weights=weights)
            else:
                n_empty += 1
        return out, n_empty

    def fit(self, train_texts: list[str]) -> "SIFEncoder":
        """Estimate word frequencies and the common component on training titles."""
        counts = Counter(w for t in train_texts for w in _tokens(t))
        total = sum(counts.values())
        self.word_prob = {w: c / total for w, c in counts.items()}

        X, _ = self._weighted_average(train_texts)
        X = X[np.abs(X).sum(axis=1) > 0]
        # First right-singular vector of the (uncentred) sentence matrix, as in
        # the original SIF paper.
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        self.pc = vt[0]
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.pc is None:
            raise RuntimeError(f"{self.name}: call fit() on training titles first")
        X, n_empty = self._weighted_average(texts)
        X -= np.outer(X @ self.pc, self.pc)
        self.last_n_empty = n_empty
        return X


def _glove_encoder(train_texts: list[str]) -> SIFEncoder:
    import gensim.downloader

    print(f"  loading {GLOVE_MODEL} (first run downloads ~380 MB) ...")
    kv = gensim.downloader.load(GLOVE_MODEL)

    def lookup(w: str) -> np.ndarray | None:
        return kv[w] if w in kv.key_to_index else None

    return SIFEncoder("glove", lookup, kv.vector_size).fit(train_texts)


def _fasttext_encoder(train_texts: list[str]) -> SIFEncoder:
    from gensim.models import FastText

    if FASTTEXT_PATH.exists():
        print(f"  loading trained FastText from {FASTTEXT_PATH.name}")
        model = FastText.load(str(FASTTEXT_PATH))
    else:
        print(f"  training FastText on {len(train_texts):,} train-period titles ...")
        sentences = [_tokens(t) for t in train_texts]
        # workers=1 keeps training deterministic for a fixed seed. bucket is the
        # size of the hashed subword table: gensim's default of 2M (~800 MB at
        # 100d) is sized for billion-word corpora; 200k is ample for ~42k titles.
        model = FastText(
            sentences=sentences, vector_size=FASTTEXT_DIM, window=5,
            min_count=3, sg=1, min_n=3, max_n=6, epochs=FASTTEXT_EPOCHS,
            bucket=200_000, seed=RANDOM_SEED, workers=1,
        )
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(FASTTEXT_PATH))

    wv = model.wv

    def lookup(w: str) -> np.ndarray | None:
        # Subword n-grams give a vector even for unseen words; an all-zero
        # vector means none of its n-grams were seen either.
        v = wv[w]
        return v if np.any(v) else None

    return SIFEncoder("fasttext", lookup, wv.vector_size).fit(train_texts)


# --------------------------------------------------------------------------
# Contextual encoders
# --------------------------------------------------------------------------

def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class FinBERTEncoder:
    """Mean-pooled last hidden layer of FinBERT, plus its sentiment head."""

    name = "finbert"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(FINBERT_MODEL)
                      .to(self.device).eval())
        cfg = self.model.config
        self.dim = cfg.hidden_size
        self.labels = [cfg.id2label[i].lower() for i in range(cfg.num_labels)]

    def encode_with_sentiment(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        # Sort by length so each batch pads to a similar length, then restore.
        lengths = np.array([len(t) for t in texts])
        order = np.argsort(lengths)
        emb = np.zeros((len(texts), self.dim), dtype=np.float32)
        probs = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        self.last_n_truncated = 0

        with torch.no_grad():
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                idx = order[start:start + EMBED_BATCH_SIZE]
                batch = [texts[i] for i in idx]
                full = self.tokenizer(batch, add_special_tokens=True, verbose=False)["input_ids"]
                self.last_n_truncated += sum(len(ids) > TRANSFORMER_MAX_LENGTH for ids in full)

                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=TRANSFORMER_MAX_LENGTH,
                                     return_tensors="pt").to(self.device)
                with torch.autocast(self.device, enabled=self.device == "cuda"):
                    out = self.model(**enc, output_hidden_states=True)
                hidden = out.hidden_states[-1].float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                emb[idx] = pooled.cpu().numpy()
                probs[idx] = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
        return emb, probs

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encode_with_sentiment(texts)[0]


class SBERTEncoder:
    """Sentence-transformer with its native mean pooling and L2 normalisation."""

    name = "sbert"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.device = _device()
        self.model = SentenceTransformer(SBERT_MODEL, device=self.device)
        self.model.max_seq_length = TRANSFORMER_MAX_LENGTH
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: list[str]) -> np.ndarray:
        # Untruncated lengths, only to count how many titles exceed the limit.
        tok = self.model.tokenizer(texts, add_special_tokens=True, verbose=False)["input_ids"]
        self.last_n_truncated = sum(len(ids) > TRANSFORMER_MAX_LENGTH for ids in tok)
        return self.model.encode(
            texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def get_encoder(name: str, articles: pd.DataFrame | None = None):
    """Build a ready-to-use encoder. Static encoders are fitted on train titles.

    The returned object has `.encode(list_of_raw_titles) -> np.ndarray`, so the
    anchor sentences in step 2 go through exactly the same path as headlines.
    """
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")
    if name in ("glove", "fasttext"):
        if articles is None:
            articles = load_articles()
        train_texts = articles.loc[train_mask(articles), "title_clean"].tolist()
        return (_glove_encoder if name == "glove" else _fasttext_encoder)(train_texts)
    return FinBERTEncoder() if name == "finbert" else SBERTEncoder()


def _read_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def embed_corpus(name: str, articles: pd.DataFrame | None = None,
                 force: bool = False) -> np.ndarray:
    """Embed every aligned title with `name` and cache it. Returns float32."""
    if articles is None:
        articles = load_articles()
    path = EMBED_DIR / f"{name}.npy"
    fp = fingerprint(articles)
    manifest = _read_manifest()
    if not force and path.exists() and manifest.get(name, {}).get("fingerprint") == fp:
        print(f"[{name}] cache is current, skipping (use --force to rebuild)")
        return np.load(path).astype(np.float32)

    print(f"[{name}] building encoder ...")
    t0 = time.perf_counter()
    encoder = get_encoder(name, articles)
    texts = articles["title_clean"].tolist()

    print(f"[{name}] encoding {len(texts):,} titles ...")
    info: dict = {}
    if name == "finbert":
        X, probs = encoder.encode_with_sentiment(texts)
        np.save(FINBERT_SENTIMENT_PATH, probs)
        info["sentiment_columns"] = encoder.labels
    else:
        X = encoder.encode(texts)
    for attr in ("last_n_truncated", "last_n_empty"):
        if hasattr(encoder, attr):
            info[attr.removeprefix("last_")] = int(getattr(encoder, attr))

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, X.astype(np.float16))
    elapsed = time.perf_counter() - t0

    model_id = {"glove": GLOVE_MODEL, "fasttext": f"trained:{FASTTEXT_PATH.name}",
                "finbert": FINBERT_MODEL, "sbert": SBERT_MODEL}[name]
    manifest = _read_manifest()   # re-read: another encoder may have finished
    manifest[name] = {
        "model": model_id, "dim": int(X.shape[1]), "n": int(X.shape[0]),
        "fingerprint": fp, "seconds": round(elapsed, 1),
        "device": getattr(encoder, "device", "cpu"), **info,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    extras = ", ".join(f"{k}={v}" for k, v in info.items() if k != "sentiment_columns")
    print(f"[{name}] done: {X.shape} in {elapsed:.0f}s" + (f" ({extras})" if extras else ""))
    return X


def load_embeddings(name: str, articles: pd.DataFrame | None = None) -> np.ndarray:
    """Load a cached matrix, verifying it is row-aligned with the articles."""
    if articles is None:
        articles = load_articles()
    entry = _read_manifest().get(name)
    path = EMBED_DIR / f"{name}.npy"
    if entry is None or not path.exists():
        raise FileNotFoundError(f"no cache for {name!r}; run: python src/embeddings.py --encoders {name}")
    if entry["fingerprint"] != fingerprint(articles):
        raise RuntimeError(f"{name} cache is stale (article rows changed); rebuild with --force")
    return np.load(path).astype(np.float32)


def load_finbert_sentiment(articles: pd.DataFrame | None = None) -> pd.DataFrame:
    """FinBERT's per-article class probabilities as a DataFrame."""
    load_embeddings("finbert", articles)   # runs the alignment check
    cols = _read_manifest()["finbert"]["sentiment_columns"]
    probs = np.load(FINBERT_SENTIMENT_PATH)
    return pd.DataFrame(probs, columns=[f"finbert_{c}" for c in cols])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    parser.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = parser.parse_args()

    articles = load_articles()
    n_train = int(train_mask(articles).sum())
    print(f"{len(articles):,} aligned articles ({n_train:,} in the training period)\n")
    for name in args.encoders:
        embed_corpus(name, articles, force=args.force)
        print()


if __name__ == "__main__":
    main()
