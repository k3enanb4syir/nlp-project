# src/models.py
"""
Task 2 -- Baseline and combined model training.

TASK FORMULATION (requirement 2a)
---------------------------------
Binary classification of the direction of the next Bank Indonesia JISDOR fixing.

    y_t = 1  if  log(rate_t / rate_{t-1}) > 0      (USD strengthens vs IDR)
    y_t = 0  otherwise

Direction is chosen over next-day level regression deliberately. Regressing the
level is a near-trivial task that a random walk wins -- the rate is highly
persistent, so predicting rate_t = rate_{t-1} gives a very low RMSE while
carrying no information whatsoever. That low error would flatter the model and
tell us nothing about the project hypothesis. Direction strips out the
persistence and asks the question the hypothesis actually poses: does
geopolitical news carry information about which way the dollar moves next?

MODEL LADDER (requirements 2b and 2c)
-------------------------------------
Each rung adds exactly one kind of information, so any improvement is
attributable:

  1. majority       -- constant predictor; the floor any model must clear.
  2. persistence    -- yesterday's direction repeated; the classic random-walk
                       momentum baseline for financial direction.
  3. arima          -- ARIMA on the log-return series; the required classical
                       time-series baseline, using price history only.
  4. logreg_rates   -- logistic regression on engineered rate features only.
  5. xgb_rates      -- gradient boosting on the same rate-only features.
  6. logreg_news    -- NLP features only, no price history. Isolates whether the
                       text carries standalone signal.
  7. logreg_combined / xgb_combined -- rate features + NLP features.
  8. logreg_tfidf   -- rate + NLP + TF-IDF over the day's headlines.

Rungs 1-5 are the "historical exchange rate data only" baselines; rungs 6-8 are
the combined models.

LEAKAGE DISCIPLINE
------------------
Everything fitted -- imputer medians, scaler statistics, the TF-IDF vocabulary
and its document frequencies, ARIMA coefficients and all model parameters -- is
estimated on the TRAINING BLOCK ALONE and then applied unchanged to validation
and test. Hyperparameters are chosen on validation; the test block is scored
once at the end.

Usage:
    python src/models.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from config import (
        RANDOM_SEED,
        RESULTS_DIR,
        TFIDF_MAX_DF,
        TFIDF_MAX_FEATURES,
        TFIDF_MIN_DF,
        TFIDF_NGRAM_RANGE,
    )
    from dataset import load_daily, split_chronologically
    from evaluate import confusion, directional_metrics, save, summarise
except ImportError:  # pragma: no cover
    from src.config import (
        RANDOM_SEED,
        RESULTS_DIR,
        TFIDF_MAX_DF,
        TFIDF_MAX_FEATURES,
        TFIDF_MIN_DF,
        TFIDF_NGRAM_RANGE,
    )
    from src.dataset import load_daily, split_chronologically
    from src.evaluate import confusion, directional_metrics, save, summarise

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "direction"
TEXT_COL = "titles_concat"


# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------

def rate_features(df: pd.DataFrame) -> list[str]:
    """Engineered features derived purely from exchange-rate history."""
    prefixes = ("ret_lag", "vol_", "mom_", "dist_ma")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    cols += [c for c in ("dow", "prev_direction") if c in df.columns]
    return cols


def news_features(df: pd.DataFrame) -> list[str]:
    """Features derived from the news text, themes and entities."""
    prefixes = ("vader_", "fin_", "theme_", "gdelt_tone")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    cols += [
        c for c in ("n_articles", "n_domains", "has_news",
                    "persons_mean", "orgs_mean", "locations_mean")
        if c in df.columns
    ]
    return cols


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def numeric_pipeline(estimator, scale: bool = True) -> Pipeline:
    """
    Impute then optionally scale, then fit the estimator.

    The imputer is part of the pipeline rather than applied beforehand so that
    its medians are learned from the training fold only.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def text_pipeline(numeric_cols: list[str], C: float) -> Pipeline:
    """
    Numeric features + TF-IDF over the day's concatenated headlines.

    The TfidfVectorizer lives inside the pipeline so that its vocabulary and IDF
    weights are learned from training days only. Fitting it on the full series
    would leak test-period vocabulary -- for example, the name of a 2026 crisis
    would acquire an IDF weight from documents the model is not supposed to have
    seen.
    """
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    min_df=TFIDF_MIN_DF,
                    max_df=TFIDF_MAX_DF,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    sublinear_tf=True,
                    stop_words="english",
                ),
                TEXT_COL,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("features", pre),
        ("model", LogisticRegression(C=C, max_iter=3000,
                                     random_state=RANDOM_SEED)),
    ])


# --------------------------------------------------------------------------
# Non-learned baselines
# --------------------------------------------------------------------------

def majority_baseline(train: pd.DataFrame, block: pd.DataFrame) -> np.ndarray:
    """Always predict the class that was most common in training."""
    cls = int(train[TARGET].mode().iloc[0])
    return np.full(len(block), cls, dtype=int)


def persistence_baseline(block: pd.DataFrame) -> np.ndarray:
    """Predict that today repeats yesterday's realised direction."""
    return block["prev_direction"].fillna(0).astype(int).to_numpy()


def arima_baseline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   order: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """
    ARIMA on log returns; the sign of the one-step-ahead forecast is the call.

    Parameters are estimated on the training returns only. The validation and
    test returns are then *appended* with `refit=False`, which advances the
    Kalman filter through the later data without re-estimating coefficients.
    The resulting fitted values are genuine one-step-ahead forecasts: each uses
    observations up to t-1 and training-period parameters, so nothing leaks.
    """
    from statsmodels.tsa.arima.model import ARIMA

    tr = train["log_return"].astype(float).reset_index(drop=True)
    later = pd.concat([val["log_return"], test["log_return"]]).astype(float)
    later = later.reset_index(drop=True)

    res = ARIMA(tr, order=order, enforce_stationarity=False,
                enforce_invertibility=False).fit()

    # Continue the filter through val+test using the training-fit parameters.
    extended = pd.concat([tr, later], ignore_index=True)
    res_ext = res.apply(extended, refit=False)
    fitted = np.asarray(res_ext.fittedvalues, dtype=float)

    n_tr, n_val = len(tr), len(val)
    return {
        "train": (fitted[:n_tr] > 0).astype(int),
        "validation": (fitted[n_tr:n_tr + n_val] > 0).astype(int),
        "test": (fitted[n_tr + n_val:] > 0).astype(int),
        "_scores": {
            "train": fitted[:n_tr],
            "validation": fitted[n_tr:n_tr + n_val],
            "test": fitted[n_tr + n_val:],
        },
    }


# --------------------------------------------------------------------------
# Training driver
# --------------------------------------------------------------------------

def _record(rows: list[dict], model: str, split: str, y_true, y_pred, y_prob=None):
    rows.append({"model": model, "split": split,
                 **directional_metrics(y_true, y_pred, y_prob)})


def _fit_eval(rows: list[dict], name: str, pipe, cols, splits,
              use_frame=False, collect: dict | None = None):
    """Fit on train, score every split, and record metrics."""
    train = splits["train"]
    X_tr = train[cols] if use_frame else train[cols].to_numpy(dtype=float)
    pipe.fit(X_tr, train[TARGET].astype(int))

    for split_name, block in splits.items():
        X = block[cols] if use_frame else block[cols].to_numpy(dtype=float)
        pred = pipe.predict(X)
        prob = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None
        _record(rows, name, split_name, block[TARGET].astype(int), pred, prob)
        # Retain test predictions so the driver can print confusion matrices
        # and write a per-day error table for later error analysis.
        if collect is not None and split_name == "test":
            collect[name] = pred
    return pipe


def select_on_validation(splits, cols, candidates, builder, label) -> tuple:
    """Pick the hyperparameter with the best VALIDATION macro-F1."""
    train, val = splits["train"], splits["validation"]
    best, best_score = None, -np.inf
    for cand in candidates:
        pipe = builder(cand)
        pipe.fit(train[cols].to_numpy(dtype=float), train[TARGET].astype(int))
        pred = pipe.predict(val[cols].to_numpy(dtype=float))
        score = directional_metrics(val[TARGET].astype(int), pred)["f1_macro"]
        if score > best_score:
            best, best_score = cand, score
    print(f"  {label}: selected {best} (validation macro-F1 {best_score:.4f})")
    return best, best_score


def run() -> pd.DataFrame:
    df = load_daily()
    splits = split_chronologically(df)
    train, val, test = splits["train"], splits["validation"], splits["test"]

    rate_cols = rate_features(df)
    news_cols = news_features(df)
    combined_cols = rate_cols + news_cols

    print(f"\nFeature groups: {len(rate_cols)} rate, {len(news_cols)} news, "
          f"{len(combined_cols)} combined")
    print(f"Split sizes   : train {len(train)}, val {len(val)}, test {len(test)}")

    rows: list[dict] = []
    test_predictions: dict[str, np.ndarray] = {}

    # ---- 1. majority -----------------------------------------------------
    print("\n[1/8] majority baseline")
    for name, block in splits.items():
        _record(rows, "majority", name, block[TARGET].astype(int),
                majority_baseline(train, block))

    # ---- 2. persistence --------------------------------------------------
    print("[2/8] persistence (random-walk) baseline")
    for name, block in splits.items():
        _record(rows, "persistence", name, block[TARGET].astype(int),
                persistence_baseline(block))
    test_predictions["persistence"] = persistence_baseline(test)

    # ---- 3. ARIMA --------------------------------------------------------
    print("[3/8] ARIMA on log returns")
    best_order, best_f1 = None, -np.inf
    for order in [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 2), (5, 0, 0)]:
        try:
            out = arima_baseline(train, val, test, order)
        except Exception as exc:
            print(f"    order {order} failed: {exc.__class__.__name__}")
            continue
        f1 = directional_metrics(val[TARGET].astype(int), out["validation"])["f1_macro"]
        if f1 > best_f1:
            best_order, best_f1, best_out = order, f1, out
    if best_order is None:
        print("    all ARIMA orders failed; skipping")
    else:
        print(f"  ARIMA: selected order {best_order} "
              f"(validation macro-F1 {best_f1:.4f})")
        for name, block in splits.items():
            _record(rows, f"arima{best_order}", name, block[TARGET].astype(int),
                    best_out[name], best_out["_scores"][name])

    # ---- 4/5. rate-only learned models -----------------------------------
    print("[4/8] logistic regression (rates only)")
    C, _ = select_on_validation(
        splits, rate_cols, [0.01, 0.1, 1.0, 10.0],
        lambda c: numeric_pipeline(LogisticRegression(C=c, max_iter=3000,
                                                      random_state=RANDOM_SEED)),
        "logreg_rates",
    )
    _fit_eval(rows, "logreg_rates",
              numeric_pipeline(LogisticRegression(C=C, max_iter=3000,
                                                  random_state=RANDOM_SEED)),
              rate_cols, splits, collect=test_predictions)

    print("[5/8] gradient boosting (rates only)")
    xgb_grid = [
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.1},
    ]
    params, _ = select_on_validation(
        splits, rate_cols, xgb_grid, lambda p: _make_xgb(p), "xgb_rates")
    _fit_eval(rows, "xgb_rates", _make_xgb(params), rate_cols, splits,
              collect=test_predictions)

    # ---- 6. news-only ----------------------------------------------------
    print("[6/8] logistic regression (news features only)")
    C, _ = select_on_validation(
        splits, news_cols, [0.01, 0.1, 1.0, 10.0],
        lambda c: numeric_pipeline(LogisticRegression(C=c, max_iter=3000,
                                                      random_state=RANDOM_SEED)),
        "logreg_news",
    )
    _fit_eval(rows, "logreg_news",
              numeric_pipeline(LogisticRegression(C=C, max_iter=3000,
                                                  random_state=RANDOM_SEED)),
              news_cols, splits, collect=test_predictions)

    # ---- 7. combined -----------------------------------------------------
    print("[7/8] combined models (rates + news)")
    C, _ = select_on_validation(
        splits, combined_cols, [0.01, 0.1, 1.0, 10.0],
        lambda c: numeric_pipeline(LogisticRegression(C=c, max_iter=3000,
                                                      random_state=RANDOM_SEED)),
        "logreg_combined",
    )
    combined_lr = _fit_eval(
        rows, "logreg_combined",
        numeric_pipeline(LogisticRegression(C=C, max_iter=3000,
                                            random_state=RANDOM_SEED)),
        combined_cols, splits, collect=test_predictions)

    params, _ = select_on_validation(
        splits, combined_cols, xgb_grid, lambda p: _make_xgb(p), "xgb_combined")
    _fit_eval(rows, "xgb_combined", _make_xgb(params), combined_cols, splits,
              collect=test_predictions)

    # ---- 8. combined + TF-IDF -------------------------------------------
    print("[8/8] combined + TF-IDF over headlines")
    tfidf_cols = combined_cols + [TEXT_COL]
    best_C, best_score = None, -np.inf
    for c in [0.05, 0.1, 0.5, 1.0]:
        pipe = text_pipeline(combined_cols, c)
        pipe.fit(train[tfidf_cols], train[TARGET].astype(int))
        pred = pipe.predict(val[tfidf_cols])
        score = directional_metrics(val[TARGET].astype(int), pred)["f1_macro"]
        if score > best_score:
            best_C, best_score = c, score
    print(f"  logreg_tfidf: selected C={best_C} "
          f"(validation macro-F1 {best_score:.4f})")
    tfidf_pipe = text_pipeline(combined_cols, best_C)
    _fit_eval(rows, "logreg_tfidf", tfidf_pipe, tfidf_cols, splits,
              use_frame=True, collect=test_predictions)

    results = pd.DataFrame(rows)

    # ---- reporting -------------------------------------------------------
    summarise(results[results["split"] == "validation"])
    summarise(results[results["split"] == "test"])

    print("\nTest-set confusion matrices")
    for name, pred in test_predictions.items():
        print(f"\n  {name}")
        print(confusion(test[TARGET].astype(int), pred))

    _report_top_tfidf(tfidf_pipe, combined_cols)
    _report_coefficients(combined_lr, combined_cols)

    # Per-day test predictions, kept for the error analysis in Task 4.
    preds = pd.DataFrame({"date": test["date"].to_numpy(),
                          "actual": test[TARGET].astype(int).to_numpy()})
    for name, pred in test_predictions.items():
        preds[name] = pred
    preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'test_predictions.csv'}")

    save(results)
    return results


def _make_xgb(params: dict):
    from xgboost import XGBClassifier
    return numeric_pipeline(
        XGBClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=4,
        ),
        scale=False,   # trees are scale-invariant
    )


def _report_coefficients(pipe: Pipeline, cols: list[str], k: int = 12) -> None:
    """Show which features the combined linear model leans on."""
    try:
        coefs = pipe.named_steps["model"].coef_[0]
    except Exception:
        return
    s = pd.Series(coefs, index=cols).sort_values()
    print(f"\nlogreg_combined -- strongest coefficients "
          f"(positive => pushes prediction toward USD UP)")
    for name, val in pd.concat([s.head(k // 2), s.tail(k // 2)]).items():
        print(f"    {name:<28} {val:+.4f}")


def _report_top_tfidf(pipe: Pipeline, numeric_cols: list[str], k: int = 12) -> None:
    """Show the headline terms the TF-IDF model weights most heavily."""
    try:
        pre = pipe.named_steps["features"]
        vec = pre.named_transformers_["tfidf"]
        coefs = pipe.named_steps["model"].coef_[0]
    except Exception:
        return
    terms = vec.get_feature_names_out()
    text_coefs = coefs[len(numeric_cols):]
    if len(text_coefs) != len(terms):
        return
    s = pd.Series(text_coefs, index=terms).sort_values()
    print("\nlogreg_tfidf -- most influential headline terms")
    print("  toward DOWN:", ", ".join(s.head(k).index))
    print("  toward UP  :", ", ".join(s.tail(k).index[::-1]))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run()


if __name__ == "__main__":
    main()
