"""
Generate the Task 2 notebooks as .ipynb files.

The notebooks are generated from this script rather than hand-authored so that
their content is version-controlled as readable Python and can be regenerated
after a pipeline change. Run `python scripts/make_notebooks.py`, then execute the
notebooks to populate their outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebook"

BOOTSTRAP = """\
import sys, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import viz

viz.apply_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 60)
print("project root:", ROOT)
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


# ---------------------------------------------------------------------------
# Notebook 1 -- Exploratory Data Analysis
# ---------------------------------------------------------------------------

def build_eda() -> nbf.NotebookNode:
    cells = [
        md("""
# Task 2 -- Exploratory Data Analysis

Global Geopolitical Event Prediction: Impact on USD Exchange Rates

This notebook explores the aligned dataset built in Task 1 and the daily feature
table built for Task 2. It covers:

1. The USD/IDR JISDOR series and the prediction target
2. News corpus composition, volume and coverage
3. The temporal alignment rule in practice
4. Sentiment and lexicon feature distributions
5. Relationships between news features and next-day direction

Every time-series chart is shaded by the chronological train / validation / test
split, because that split is the most important structural constraint on the
analysis.
        """),
        code(BOOTSTRAP),
        code("""
daily = pd.read_csv(config.DAILY_DATASET_CSV, parse_dates=["date"])
aligned = pd.read_csv(config.ALIGNED_ARTICLES_CSV, low_memory=False,
                      parse_dates=["query_date", "trading_day"])
rates = pd.read_csv(config.RATES_CLEAN_CSV, parse_dates=["date"])

print(f"daily feature table : {daily.shape[0]:,} trading days x {daily.shape[1]} cols")
print(f"aligned articles    : {len(aligned):,}")
print(f"JISDOR fixings      : {len(rates):,}")
print(f"window              : {daily.date.min().date()} -> {daily.date.max().date()}")
        """),

        md("## 1. The exchange rate series and the target"),
        code("""
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(daily.date, daily.rate, color=viz.COLOR_RATE, lw=1.4)
ax.set_ylabel("IDR per USD")
ax.set_title("Bank Indonesia JISDOR USD/IDR reference rate")
viz.shade_splits(ax, pd.Timestamp(config.TRAIN_END), pd.Timestamp(config.VAL_END))

ax = axes[1]
ax.plot(daily.date, daily.log_return * 100, color=viz.TEXT_SECONDARY, lw=0.8)
ax.axhline(0, color=viz.GRID, lw=1)
ax.set_ylabel("daily log return (%)")
ax.set_title("Daily log returns")
viz.shade_splits(ax, pd.Timestamp(config.TRAIN_END), pd.Timestamp(config.VAL_END),
                 label=False)

plt.tight_layout()
plt.show()
        """),
        code("""
# The rupiah depreciated steadily across the window, so the UP class dominates.
# That base rate is what every model has to be judged against.
print("Target: 1 = USD strengthens against IDR (rate rises)")
print(f"overall UP rate : {daily.direction.mean():.3f}")

blocks = {
    "train":      daily[daily.date.dt.date <= config.TRAIN_END],
    "validation": daily[(daily.date.dt.date > config.TRAIN_END)
                        & (daily.date.dt.date <= config.VAL_END)],
    "test":       daily[daily.date.dt.date > config.VAL_END],
}
summary = pd.DataFrame({
    name: {
        "days": len(b),
        "UP rate": b.direction.mean(),
        "mean |return| %": (b.log_return.abs() * 100).mean(),
        "return std %": (b.log_return * 100).std(),
    }
    for name, b in blocks.items()
}).T
summary.round(4)
        """),
        md("""
The UP rate differs between blocks (train 0.56, validation 0.51, test 0.58). This
is a real property of the period, not a splitting error, and it is exactly why
accuracy alone is a misleading metric here: a constant "UP" predictor scores
differently on each block. Macro-F1 and MCC are reported alongside accuracy for
this reason.
        """),

        md("## 2. News corpus composition"),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# Left: monthly article volume -- one series, so no legend needed.
monthly = (aligned.set_index("trading_day")
                  .resample("MS").size().rename("articles").reset_index())
axes[0].plot(monthly.trading_day, monthly.articles, color=viz.COLOR_NEWS, lw=1.6)
axes[0].set_title("Articles per month (after filtering)")
axes[0].set_ylabel("articles")
axes[0].tick_params(axis="x", rotation=0)

# Right: top publishers -- horizontal bars, direct-labelled.
top = aligned.domain.value_counts().head(12).sort_values()
bars = axes[1].barh(top.index, top.values, color=viz.COLOR_RATE, height=0.7)
axes[1].set_title("Top 12 publishers in the filtered corpus")
axes[1].grid(axis="x", visible=True)
axes[1].grid(axis="y", visible=False)
for bar, val in zip(bars, top.values):
    axes[1].text(val + 30, bar.get_y() + bar.get_height() / 2, f"{val:,}",
                 va="center", fontsize=8, color=viz.TEXT_SECONDARY)
axes[1].set_xlim(0, top.values.max() * 1.15)

plt.tight_layout()
plt.show()
        """),
        code("""
per_day = aligned.groupby("trading_day").size()
print(f"trading days with news : {len(per_day):,} / {len(rates):,} "
      f"({100*len(per_day)/len(rates):.1f}%)")
print(f"articles per trading day: mean {per_day.mean():.1f}, "
      f"median {per_day.median():.0f}, min {per_day.min()}, max {per_day.max()}")
print(f"distinct publishers     : {aligned.domain.nunique()}")
        """),

        md("""
## 3. The temporal alignment rule in practice

Every article is attached to the first trading day **strictly after** its
publication date, because the GDELT snapshot is sampled at 12:00 UTC (19:00 WIB)
— after Bank Indonesia has already published that day's fixing at ~10:00 WIB.

The lag histogram below is a direct check that the rule behaves as intended:
weekday news lands on the next session (+1 day), while Friday and weekend news
accumulates onto Monday (+2 to +3 days). Longer lags are Indonesian public
holidays.
        """),
        code("""
lag = (aligned.trading_day - aligned.query_date).dt.days
counts = lag.value_counts().sort_index()
counts = counts[counts.index <= 8]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar([f"+{d}d" for d in counts.index], counts.values,
              color=viz.COLOR_RATE, width=0.62)
ax.set_title("Publication-to-fixing lag (weekends and holidays roll forward)")
ax.set_ylabel("articles")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + counts.values.max() * 0.015,
            f"{val:,}", ha="center", fontsize=8, color=viz.TEXT_SECONDARY)
ax.set_ylim(0, counts.values.max() * 1.12)
plt.tight_layout()
plt.show()

wd = aligned.assign(dow=aligned.query_date.dt.day_name()).groupby("dow").size()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print("articles by publication weekday:")
print(wd.reindex(order).to_string())
        """),

        md("## 4. Sentiment and lexicon feature distributions"),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# Article-level VADER compound distribution.
import features as feat
art = aligned.sample(min(15000, len(aligned)), random_state=0).copy()
scored = feat.score_articles(art)

axes[0].hist(scored.vader_compound, bins=60, color=viz.COLOR_RATE,
             edgecolor=viz.SURFACE, linewidth=0.4)
axes[0].set_title("Article-level VADER compound score")
axes[0].set_xlabel("compound")
axes[0].set_ylabel("articles")

# Daily mean sentiment over time, with a rolling trend for legibility.
axes[1].plot(daily.date, daily.vader_mean, color=viz.GRID, lw=0.7,
             label="daily mean")
axes[1].plot(daily.date, daily.vader_mean.rolling(21).mean(),
             color=viz.COLOR_NEWS, lw=1.8, label="21-day average")
axes[1].axhline(0, color=viz.TEXT_SECONDARY, lw=0.8, alpha=0.5)
axes[1].set_title("Daily mean headline sentiment")
axes[1].set_ylabel("VADER compound")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()

print(f"share of headlines scored negative : {(scored.vader_compound < -0.05).mean():.3f}")
print(f"share scored positive              : {(scored.vader_compound >  0.05).mean():.3f}")
print(f"share scored neutral               : {(scored.vader_compound.abs() <= 0.05).mean():.3f}")
        """),
        code("""
lex_cols = [c for c in daily.columns if c.endswith("_rate")
            and c.startswith(("fin_", "geo_", "monetary"))]

fig, ax = plt.subplots(figsize=(10, 4.2))
for i, col in enumerate(lex_cols[:3]):
    ax.plot(daily.date, daily[col].rolling(21).mean(),
            color=viz.CATEGORICAL[i], lw=1.6, label=col.replace("_rate", ""))
ax.set_title("Lexicon hit-rate per article (21-day average)")
ax.set_ylabel("matches per article")
ax.legend(loc="upper left", ncols=3)
viz.shade_splits(ax, pd.Timestamp(config.TRAIN_END), pd.Timestamp(config.VAL_END),
                 label=False)
plt.tight_layout()
plt.show()

daily[lex_cols].describe().T.round(3)
        """),

        md("""
## 5. Do news features relate to next-day direction?

The honest way to look at this before modelling is the point-biserial
correlation between each feature and the binary target, computed on the
**training block only** — inspecting this on the test block would be a form of
manual leakage.
        """),
        code("""
train = daily[daily.date.dt.date <= config.TRAIN_END].copy()

news_cols = [c for c in daily.columns
             if c.startswith(("vader_", "fin_", "theme_", "gdelt_tone"))
             or c in ("n_articles", "n_domains", "persons_mean",
                      "orgs_mean", "locations_mean")]

corr = (train[news_cols]
        .apply(lambda s: s.corr(train["direction"]))
        .dropna()
        .sort_values())

top = pd.concat([corr.head(8), corr.tail(8)])
colors = [viz.COLOR_DOWN if v < 0 else viz.COLOR_UP for v in top.values]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top.index, top.values, color=colors, height=0.7)
ax.axvline(0, color=viz.TEXT_SECONDARY, lw=1)
ax.set_title("Correlation of news features with next-day UP (train block only)")
ax.set_xlabel("point-biserial correlation")
ax.grid(axis="x", visible=True)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

print(f"strongest absolute correlation: {corr.abs().max():.4f}")
        """),
        md("""
### Reading this chart

The correlations are all very small — the strongest is well under 0.1. That is
the expected result for daily FX direction and is the single most important
finding of the EDA: **any predictive signal from news is weak**, so the
modelling in `02_baseline_results.ipynb` should be read with the majority-class
baseline firmly in view, and small accuracy differences should not be
over-interpreted.
        """),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


# ---------------------------------------------------------------------------
# Notebook 2 -- Baseline experimental results
# ---------------------------------------------------------------------------

def build_results() -> nbf.NotebookNode:
    cells = [
        md("""
# Task 2 -- Baseline Experimental Results

This notebook reports the initial experiments. The model ladder is defined in
`src/models.py`; run it first to produce `results/metrics.csv`.

    python src/features.py
    python src/dataset.py
    python src/models.py

**Task formulation.** Binary classification of the direction of the next JISDOR
fixing: `y_t = 1` if `log(rate_t / rate_{t-1}) > 0`.

**Metrics.** Directional accuracy (headline), macro-F1 and MCC (robust to the
class imbalance), and ROC-AUC (threshold-free ranking quality).
        """),
        code(BOOTSTRAP),
        code("""
metrics = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
preds = pd.read_csv(config.RESULTS_DIR / "test_predictions.csv", parse_dates=["date"])
print(f"{metrics.model.nunique()} models x {metrics.split.nunique()} splits")
metrics.model.unique()
        """),

        md("## 1. Validation leaderboard (used for model selection)"),
        code("""
cols = ["model", "accuracy", "majority_baseline", "f1_macro", "mcc", "roc_auc",
        "pred_up_rate"]
(metrics[metrics.split == "validation"][cols]
 .sort_values("f1_macro", ascending=False)
 .round(4)
 .reset_index(drop=True))
        """),

        md("## 2. Test leaderboard (scored once, after selection)"),
        code("""
test_tbl = (metrics[metrics.split == "test"][cols]
            .sort_values("f1_macro", ascending=False)
            .round(4)
            .reset_index(drop=True))
test_tbl
        """),

        md("""
### 2.1 Accuracy against the majority baseline

Accuracy is plotted against the majority-class rate for the test block. A bar
below the dashed line is a model that would have been beaten by always
predicting "UP".
        """),
        code("""
t = metrics[metrics.split == "test"].sort_values("accuracy", ascending=False)
base = t.majority_baseline.iloc[0]

fig, ax = plt.subplots(figsize=(10, 4.6))
colors = [viz.COLOR_UP if a >= base else viz.COLOR_DOWN for a in t.accuracy]
bars = ax.bar(t.model, t.accuracy, color=colors, width=0.62)
ax.axhline(base, color=viz.TEXT_SECONDARY, ls="--", lw=1.2)
ax.text(len(t) - 0.4, base + 0.004, f"majority baseline {base:.3f}",
        ha="right", fontsize=9, color=viz.TEXT_SECONDARY)
ax.set_ylabel("directional accuracy")
ax.set_title("Test-set directional accuracy vs the majority-class baseline")
ax.set_ylim(0.40, max(t.accuracy.max(), base) + 0.045)
viz.bar_labels(ax, bars, fmt="{:.3f}")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()
        """),
        md("""
### 2.2 The metric that actually separates the models

Accuracy rewards the degenerate constant predictor, which scores 0.580 on the
test block simply because 58% of test days were UP — while having zero
discriminative power. MCC exposes that: it is 0 for any constant predictor.
        """),
        code("""
t2 = metrics[metrics.split == "test"].sort_values("mcc")

fig, ax = plt.subplots(figsize=(10, 4.6))
colors = [viz.COLOR_DOWN if v < 0 else viz.COLOR_UP for v in t2.mcc]
ax.barh(t2.model, t2.mcc, color=colors, height=0.68)
ax.axvline(0, color=viz.TEXT_SECONDARY, lw=1)
ax.set_xlabel("Matthews correlation coefficient")
ax.set_title("Test-set MCC (0 = no skill, including any constant predictor)")
ax.grid(axis="x", visible=True)
ax.grid(axis="y", visible=False)
# Pad both ends so the outermost value labels do not collide with the tick
# labels or run off the axes.
span = t2.mcc.max() - min(t2.mcc.min(), 0)
ax.set_xlim(min(t2.mcc.min(), 0) - span * 0.18, t2.mcc.max() + span * 0.18)
for i, (m, v) in enumerate(zip(t2.model, t2.mcc)):
    ax.text(v + (0.003 if v >= 0 else -0.003), i, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8, color=viz.TEXT_SECONDARY)
plt.tight_layout()
plt.show()
        """),

        md("## 3. Does adding news help? Baseline vs combined"),
        code("""
pairs = [("logreg_rates", "logreg_combined"), ("xgb_rates", "xgb_combined")]
rows = []
for base_m, comb_m in pairs:
    for split in ["validation", "test"]:
        b = metrics[(metrics.model == base_m) & (metrics.split == split)]
        c = metrics[(metrics.model == comb_m) & (metrics.split == split)]
        if b.empty or c.empty:
            continue
        rows.append({
            "split": split,
            "pair": f"{base_m} -> {comb_m}",
            "acc_rates": b.accuracy.iloc[0],
            "acc_combined": c.accuracy.iloc[0],
            "d_acc": c.accuracy.iloc[0] - b.accuracy.iloc[0],
            "mcc_rates": b.mcc.iloc[0],
            "mcc_combined": c.mcc.iloc[0],
            "d_mcc": c.mcc.iloc[0] - b.mcc.iloc[0],
        })
pd.DataFrame(rows).round(4)
        """),
        code("""
# Grouped comparison on the test block: rate-only vs rate+news, two series only.
fig, ax = plt.subplots(figsize=(8.5, 4.4))
labels = ["logistic regression", "gradient boosting"]
x = np.arange(len(labels))
w = 0.34

rates_acc, comb_acc = [], []
for base_m, comb_m in pairs:
    rates_acc.append(metrics[(metrics.model == base_m) & (metrics.split == "test")].accuracy.iloc[0])
    comb_acc.append(metrics[(metrics.model == comb_m) & (metrics.split == "test")].accuracy.iloc[0])

b1 = ax.bar(x - w/2 - 0.01, rates_acc, w, label="rates only", color=viz.COLOR_RATE)
b2 = ax.bar(x + w/2 + 0.01, comb_acc, w, label="rates + news", color=viz.COLOR_NEWS)
ax.axhline(base, color=viz.TEXT_SECONDARY, ls="--", lw=1.2)
ax.set_xticks(x, labels)
ax.set_ylabel("test directional accuracy")
ax.set_title("Adding NLP features to the rate-only baseline")
ax.set_ylim(0.40, 0.66)
ax.legend(loc="upper right")
viz.bar_labels(ax, list(b1) + list(b2), fmt="{:.3f}")
plt.tight_layout()
plt.show()
        """),

        md("## 4. Error structure on the test block"),
        code("""
from sklearn.metrics import confusion_matrix

show = [m for m in ["persistence", "logreg_rates", "logreg_combined", "logreg_tfidf"]
        if m in preds.columns]

fig, axes = plt.subplots(1, len(show), figsize=(3.1 * len(show), 3.2))
for ax, name in zip(np.atleast_1d(axes), show):
    cm = confusion_matrix(preds.actual, preds[name], labels=[0, 1])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_title(name, fontsize=10)
    ax.set_xticks([0, 1], ["DOWN", "UP"], fontsize=8)
    ax.set_yticks([0, 1], ["DOWN", "UP"], fontsize=8)
    ax.set_xlabel("predicted", fontsize=9)
    if name == show[0]:
        ax.set_ylabel("actual", fontsize=9)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else viz.TEXT_PRIMARY)
plt.tight_layout()
plt.show()
        """),
        code("""
# How often does each model simply say UP?
rate_up = preds.drop(columns=["date", "actual"]).mean().sort_values()
print("share of test days predicted UP (actual = "
      f"{preds.actual.mean():.3f}):")
print(rate_up.round(3).to_string())
        """),

        md("""
## 5. Findings

1. **No model beats the majority-class baseline on raw test accuracy.** The test
   block was 58.0% UP days; the best model reached 56.8%. On accuracy alone, the
   honest conclusion is that none of these baselines demonstrates usable skill.

2. **On balanced metrics the picture is slightly better but still weak.** The
   majority predictor scores MCC 0 and macro-F1 0.367. Several models beat that
   on macro-F1, meaning they do make genuine two-sided predictions rather than
   collapsing to one class — but the MCC values are all below 0.08, which is
   close to noise for a 176-day test block.

3. **Adding NLP features did not produce a reliable improvement.** The effect
   depends on the model: news helped the tree model (`xgb_rates` → `xgb_combined`,
   MCC 0.032 → 0.078) and hurt the linear one (`logreg_rates` → `logreg_combined`,
   MCC 0.053 → 0.007). Opposite moves of similar size are what noise looks like;
   with this sample size neither is distinguishable from chance.

4. **TF-IDF overfits badly.** `logreg_tfidf` was the weakest model on test
   (accuracy 0.483, MCC −0.088) despite competitive validation numbers. 300
   vocabulary features against 828 training days is far too many degrees of
   freedom, and the influential-terms list is dominated by period-specific proper
   nouns rather than transferable economic vocabulary.

5. **This is the expected result, not a bug.** Daily FX direction is close to a
   martingale; published effect sizes for news-sentiment FX prediction are small
   and unstable. The EDA already showed all feature/target correlations below
   0.1. Establishing that the weak-signal regime is real — with a leakage-free
   pipeline that would have revealed a strong signal had one existed — is the
   correct outcome for a baseline task.

### Implications for Task 3

* Reduce text dimensionality rather than expand it (fewer, better features).
* Consider volatility or large-move classification, where news sentiment is
  better established as a predictor than directional sign.
* Aggregate over multi-day windows to raise the signal-to-noise ratio.
* Weight articles by publisher influence or theme relevance rather than treating
  all 45 daily headlines equally.
        """),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def build_ablation() -> nbf.NotebookNode:
    cells = [
        md("""
# Task 3 -- Ablation matrix

Each row adds or removes one kind of information relative to its neighbour,
so a change in score can be attributed. The rows are fitted by
`src/ablation.py`; run it first to produce the `results/ablation_*.csv` files.

    python src/ablation.py --build-raw   # once; needs data/embeddings/ (C-raw only)
    python src/ablation.py

| Row | Features | Question |
|---|---|---|
| A | rates only | time-series baseline |
| B | rates + Task 2 lexicons | does basic NLP help? |
| C1 / C2 | rates + GloVe / FastText axes | static embeddings |
| C3 / C4 | rates + FinBERT / SBERT axes | contextual; domain vs general |
| C-fbs | rates + FinBERT sentiment | why not just FinBERT sentiment? |
| C-raw | rates + PCA of raw embeddings (best encoder) | do axes beat dumping vectors? |
| C5 | best C + novelty and concentration | does news surprise add signal? |
| C6 / C6-stack | late fusion with / without the novelty gate | fusion strategy |
| drop-* | C5 minus one channel | which channel matters |

**Selection.** Every row picks its model family (logistic regression or
XGBoost), hyperparameters and decay half-life on validation macro-F1. "Best
encoder" is the C1-C4 row with the best validation macro-F1, so it can differ
by target. The test block is scored once.

**Uncertainty.** With 176 test days a 0.02 MCC change is noise. Every row
reports MCC(row) - MCC(A) with a 95% circular block-bootstrap CI (10-session
blocks, 2,000 resamples, both models scored on the same resampled days).
        """),
        code(BOOTSTRAP),
        code("""
from anchors import CHANNELS

metrics = pd.read_csv(config.ABLATION_METRICS_CSV)
boot = pd.read_csv(config.ABLATION_BOOTSTRAP_CSV)
TARGETS = ["direction", "large_move"]
ORDER = (["A", "B", "C1", "C2", "C3", "C4", "C-fbs", "C-raw", "C5", "C6", "C6-stack"]
         + [f"drop-{c}" for c in CHANNELS])
print(f"{metrics.row.nunique()} rows x {metrics.target.nunique()} targets")
        """),

        md("## 1. The matrix (test block)"),
        code("""
cols = ["description", "encoder", "family", "decay_halflife", "accuracy", "f1_macro",
        "mcc", "roc_auc", "d_mcc", "ci_low", "ci_high", "p_gt0"]

def matrix(target):
    t = metrics[(metrics.target == target) & (metrics.split == "test")]
    t = t.merge(boot[boot.target == target], on=["target", "row"]).set_index("row")
    return t.reindex([r for r in ORDER if r in t.index])[cols].round(3)

matrix("direction")
        """),
        code('matrix("large_move")'),

        md("""
## 2. MCC difference against Model A

A dot right of zero beats the rates-only baseline on the test block. The bar is
the 95% block-bootstrap CI of that difference; a bar that crosses zero is not
distinguishable from no effect.
        """),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
rows = [r for r in ORDER if r != "A"]
ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
for ax, target in zip(axes, TARGETS):
    b = boot[(boot.target == target) & boot.row.isin(rows)]
    y = b.row.map(ypos)
    ax.errorbar(b.d_mcc, y, xerr=[b.d_mcc - b.ci_low, b.ci_high - b.d_mcc],
                fmt="o", color=viz.BLUE, ecolor=viz.BLUE, elinewidth=1.5,
                markersize=6, capsize=0)
    ax.axvline(0, color=viz.TEXT_SECONDARY, lw=1)
    for edge in ("C1", "C5", f"drop-{CHANNELS[0]}"):   # group separators
        ax.axhline(ypos[edge] + 0.5, color=viz.GRID, lw=1)
    ax.set_title(target)
    ax.set_xlabel("test MCC minus Model A's")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(list(ypos.values()), list(ypos.keys()))
fig.suptitle("Difference in test MCC vs Model A, with 95% block-bootstrap CI", x=0.01, ha="left")
plt.tight_layout()
plt.show()
        """),

        md("""
## 3. Does validation predict test?

Every choice in the matrix (encoder, family, half-life) is made on validation.
If validation and test scores are unrelated, those choices are close to random.
Late-fusion rows are left out because their meta-model is fitted on
validation, so their validation scores are in-sample.
        """),
        code("""
v = metrics[metrics.split == "validation"]
t = metrics[metrics.split == "test"]
m = v.merge(t, on=["target", "row"], suffixes=("_val", "_test"))
m = m[~m.val_in_sample_val]

fig, ax = plt.subplots(figsize=(6.6, 5.2))
for target, color in zip(TARGETS, [viz.BLUE, viz.ORANGE]):
    s = m[m.target == target]
    r = s[["mcc_val", "mcc_test"]].corr().iloc[0, 1]
    ax.scatter(s.mcc_val, s.mcc_test, s=40, color=color,
               edgecolor=viz.SURFACE, linewidth=1.5, label=f"{target} (r = {r:+.2f})")
ax.axhline(0, color=viz.TEXT_SECONDARY, lw=1)
ax.axvline(0, color=viz.TEXT_SECONDARY, lw=1)
ax.set_xlabel("validation MCC")
ax.set_ylabel("test MCC")
ax.set_title("Validation MCC vs test MCC, one dot per ablation row")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
        """),
        code("""
# What validation chose, per target.
sel = (metrics[(metrics.split == "test") & metrics.row.isin(["A", "C1", "C2", "C3", "C4", "C5"])]
       [["target", "row", "encoder", "family", "decay_halflife", "params", "n_features"]])
sel.sort_values(["target", "row"]).reset_index(drop=True)
        """),

        md("""
## 4. Which channel matters? (drop-one)

Change in test MCC when one channel's direction and attention scores are
removed from C5. A negative bar means the model got worse without the channel,
i.e. the channel was helping.
        """),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, target in zip(axes, TARGETS):
    t = metrics[(metrics.target == target) & (metrics.split == "test")].set_index("row")
    d = pd.Series({c: t.loc[f"drop-{c}", "mcc"] - t.loc["C5", "mcc"] for c in CHANNELS})
    ax.barh(list(d.index)[::-1], d.values[::-1], height=0.62,
            color=[viz.COLOR_DOWN if x < 0 else viz.COLOR_UP for x in d.values[::-1]])
    ax.axvline(0, color=viz.TEXT_SECONDARY, lw=1)
    ax.set_title(f"{target} (C5 on {t.loc['C5', 'encoder']})")
    ax.set_xlabel("test MCC without the channel minus C5's")
    ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
        """),

        md("""
## 5. Findings

1. **No text row beats Model A on direction with a CI that excludes zero.** The
   largest gain is C4 (SBERT axes, +0.112, CI −0.044 to +0.279, ahead of A in
   91% of resamples). Validation picked GloVe as the best encoder for
   direction, so C5, C6 and the drop-one rows are built on GloVe. Of those,
   only drop-trade is ahead of A (+0.075, CI −0.089 to +0.256).

2. **For direction, validation does not predict test; for large moves, it
   does.** Across ablation rows, validation and test MCC correlate at
   r = +0.10 for direction and r = +0.78 for large moves (section 3). So the
   validation-based choices are close to random for direction, while the
   large-move results are more trustworthy.

3. **Large moves are where text looks most useful.** Every C1–C4 row is ahead
   of A in at least 71% of resamples, and C2 (FastText) gains +0.105 (CI
   −0.042 to +0.241). Model A itself is close to zero (MCC 0.010). This fits the
   literature: news predicts magnitude better than sign.

4. **Axes beat raw embeddings, but within noise.** For the encoder validation
   picked, the axes row is ahead of C-raw on both targets (direction +0.046 vs
   −0.052; large move +0.105 vs +0.074), with 17 or fewer interpretable
   features against up to 32 opaque ones.

5. **Novelty and the gate add nothing reliable.** C5 is below its best C on
   both targets. The gated C6 beats the ungated C6-stack on direction and loses
   on large moves.

6. **FinBERT sentiment is no better than the axes.** C-fbs is −0.042
   (direction) and +0.046 (large move), never ahead of the best axes row.

7. **Drop-one finds no channel that clearly carries the signal.** For
   direction, removing any one of the seven GloVe channels *raises* test MCC.
   For large moves (FastText), removing em or geo lowers it, and four others
   raise it. The single CI that excludes
   zero in the whole matrix is drop-trade on large moves (+0.131, CI +0.008 to
   +0.258): 1 of 34 comparisons, about what chance alone would give at the 95%
   level.

**Bottom line for the report.** The channel-axis features are a defensible,
interpretable representation, and the large-move target shows a consistent
positive tilt. But with 176 test days, no fusion or feature set is
statistically distinguishable from the rates-only baseline. Report it in
those words.
        """),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def build_feature_analysis() -> nbf.NotebookNode:
    cells = [
        md("""
# Task 3 -- Feature analysis

Four views of what the text features measure and how the models use them.
Everything here is read from CSVs written by `src/analysis.py`, which needs the
gitignored embedding cache; this notebook does not.

    python src/ablation.py
    python src/analysis.py

1. **Embedding map** -- SBERT headlines in the plane of their first two
   principal components, with the seven channel axes drawn as arrows.
2. **SHAP + headline retrieval** -- which features the C5 model leans on, and
   the headlines behind its largest text-driven calls.
3. **Event check** -- do the channels react to known events? No returns used.
4. **Error table** -- the days where C5 and Model A disagree.

The plan named UMAP and LightGBM. A linear map is used because only a linear
projection turns an axis into an arrow, and SHAP comes from XGBoost's built-in
exact TreeSHAP, because the C5 models are XGBoost.
        """),
        code(BOOTSTRAP),
        code("""
from anchors import CHANNELS

R = config.RESULTS_DIR
pts = pd.read_csv(config.ANALYSIS_MAP_CSV, parse_dates=["date"])
arrows = pd.read_csv(config.ANALYSIS_MAP_AXES_CSV)
imp = pd.read_csv(config.ANALYSIS_SHAP_CSV)
shifts = pd.read_csv(config.ANALYSIS_SHIFTS_CSV, parse_dates=["date"])
heads = pd.read_csv(config.ANALYSIS_SHIFT_HEADLINES_CSV, parse_dates=["date"])
events = pd.read_csv(config.ANALYSIS_EVENTS_CSV)
errors = pd.read_csv(config.ANALYSIS_ERRORS_CSV, parse_dates=["date"])
pd.set_option("display.max_colwidth", 110)
        """),

        md("""
## 1. Embedding map

Each dot is a headline; orange dots come from sessions with a large move. Each
arrow starts at its channel's topic vector and points toward the USD/IDR-up
pole. Arrows are scaled up by a common factor so they are visible; the table
below gives each axis's true share in this plane.
        """),
        code("""
fig, ax = plt.subplots(figsize=(9, 7))
for flag, color, label in [(0, viz.BLUE, "normal session"), (1, viz.ORANGE, "large-move session")]:
    s = pts[pts.large_move == flag]
    ax.scatter(s.x, s.y, s=6, color=color, alpha=0.35, linewidths=0, label=label)
scale = 0.35 / arrows.in_plane_share.max()
for r in arrows.itertuples():
    ax.annotate("", xy=(r.x0 + scale * r.dx, r.y0 + scale * r.dy), xytext=(r.x0, r.y0),
                arrowprops=dict(arrowstyle="-|>", color=viz.TEXT_PRIMARY, lw=1.6))
    ax.text(r.x0 + scale * r.dx * 1.12, r.y0 + scale * r.dy * 1.12, r.channel,
            fontsize=10, color=viz.TEXT_PRIMARY, ha="center", va="center")
ax.set_xlabel("PC1 (training headlines)")
ax.set_ylabel("PC2")
ax.set_title(f"SBERT headline map; channel arrows scaled x{scale:.1f}")
# Keep every arrow tip and label inside the frame.
tips_x = arrows.x0 + scale * arrows.dx * 1.2
tips_y = arrows.y0 + scale * arrows.dy * 1.2
ax.set_xlim(min(pts.x.min(), tips_x.min()) - 0.05, max(pts.x.max(), tips_x.max()) + 0.08)
ax.set_ylim(min(pts.y.min(), tips_y.min()) - 0.05, max(pts.y.max(), tips_y.max()) + 0.05)
ax.legend(loc="upper right", markerscale=3)
plt.tight_layout()
plt.show()
        """),
        code("""
print(f"large-move share among mapped headlines: {pts.large_move.mean():.3f}")
arrows[["channel", "in_plane_share"]].sort_values("in_plane_share", ascending=False).round(3)
        """),

        md("""
## 2. SHAP: what the C5 models lean on

Mean |SHAP| over the test block, in log-odds. Blue bars are rate features,
orange bars are text features (channel scores, novelty).
        """),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, (target, t) in zip(axes, imp.groupby("target", sort=False)):
    t = t.sort_values("mean_abs_shap").tail(12)
    colors = [viz.COLOR_RATE if g == "rates" else viz.COLOR_NEWS for g in t.group]
    ax.barh(t.feature, t.mean_abs_shap, color=colors, height=0.66)
    ax.set_title(f"{target} (C5 on {t.encoder.iloc[0]})")
    ax.set_xlabel("mean |SHAP| on test (log-odds)")
    ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

share = imp.groupby(["target", "group"]).mean_abs_shap.sum()
(share / share.groupby(level=0).transform("sum")).unstack().round(3)
        """),
        md("""
### 2.1 The largest text-driven calls, and the headlines behind them

`text_shift` is the summed SHAP of all text features on that day: how far the
news moved the model's log-odds. `channel` is the axis feature with the
largest |SHAP|. Below each table: the headlines that channel's top-k pooling
actually used that day.
        """),
        code("""
cols = ["date", "actual", "pred", "prob", "text_shift", "rate_shift", "channel", "top_axis_shap"]
shifts[shifts.target == "direction"][cols].round(3).reset_index(drop=True)
        """),
        code("""
(heads[(heads.target == "direction") & (heads["rank"] <= 3)]
 [["date", "channel", "rank", "title", "relevance", "direction"]].round(3).reset_index(drop=True))
        """),
        code("""
shifts[shifts.target == "large_move"][cols].round(3).reset_index(drop=True)
        """),
        code("""
(heads[(heads.target == "large_move") & (heads["rank"] <= 3)]
 [["date", "channel", "rank", "title", "relevance", "direction"]].round(3).reset_index(drop=True))
        """),

        md("""
## 3. Event face-validity check

Attention of the risk, geo and trade channels (SBERT), as a z-score against the
training distribution, around three events. Each event has one channel that
should react: risk for SVB, geo for 7 October, trade for the April 2025
tariffs. News is aligned to the session after it is published, so the reaction
starts after the dashed line.
        """),
        code("""
tf = pd.read_csv(config.TEXT_FEATURES_DIR / "sbert.csv", parse_dates=["date"]).sort_values("date")
train = tf[tf.date <= pd.Timestamp(config.TRAIN_END)]
show = {"risk": viz.BLUE, "geo": viz.ORANGE, "trade": viz.AQUA}

fig, axes = plt.subplots(1, len(config.ANALYSIS_EVENTS), figsize=(13, 4.2), sharey=True)
for ax, (name, day, expected) in zip(axes, config.ANALYSIS_EVENTS):
    d = pd.Timestamp(day)
    pos = tf.date.searchsorted(d)
    w = tf.iloc[max(0, pos - 15): pos + 15]
    ends = {}
    for ch, color in show.items():
        col = f"ax_{ch}_attn"
        z = (w[col] - train[col].mean()) / train[col].std()
        ax.plot(w.date, z, color=color, lw=2.2 if ch == expected else 1.4,
                alpha=1 if ch == expected else 0.75)
        ends[ch] = z.iloc[-1]
    # Direct labels at the line ends, nudged apart so they never overlap.
    placed = []
    for ch, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y = max([y] + [p + 0.55 for p in placed[-1:]])
        placed.append(y)
        ax.text(w.date.iloc[-1], y, f" {ch}", color=viz.TEXT_PRIMARY, fontsize=9, va="center")
    ax.axvline(d, color=viz.TEXT_SECONDARY, ls="--", lw=1)
    ax.axhline(0, color=viz.GRID, lw=1)
    ax.set_title(f"{name} (expect: {expected})", fontsize=10)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
axes[0].set_ylabel("attention z-score vs training")
plt.tight_layout()
plt.show()
        """),
        md("""
Rank of the expected channel's attention spike among all seven channels (1 =
the biggest spike), and its size, for every encoder:
        """),
        code("""
exp = events[events.channel == events.expected]
pd.concat({"rank": exp.pivot(index="event", columns="encoder", values="attn_rank"),
           "z": exp.pivot(index="event", columns="encoder", values="attn_z_max").round(2)}, axis=1)
        """),
        code("""
# All seven channels for SBERT: attention spike and mean direction (+ = USD/IDR up).
(events[events.encoder == "sbert"]
 .sort_values(["event", "attn_rank"])
 [["event", "window", "channel", "attn_z_max", "dir_z_mean", "attn_rank"]]
 .round(2).reset_index(drop=True))
        """),

        md("""
## 4. Error table: where do C5 and Model A disagree?

Test days split by which model was right. For each group: how many days, the
mean absolute return, the news volume and the novelty of the day's news.
        """),
        code("""
(errors.groupby(["target", "group"])
 .agg(days=("date", "size"), abs_return=("abs_return", "mean"),
      n_articles=("n_articles", "mean"), nov_centroid=("nov_centroid", "mean"))
 .round(4))
        """),
        code("""
# Days where the news model was right and the rates model wrong, with the
# day's most FX-relevant headlines.
(errors[errors.group == "C5 right, A wrong"]
 [["target", "date", "actual", "abs_return", "top_headlines"]]
 .sort_values(["target", "abs_return"], ascending=[True, False])
 .groupby("target").head(6).round(4).reset_index(drop=True))
        """),
        code("""
(errors[errors.group == "A right, C5 wrong"]
 [["target", "date", "actual", "abs_return", "top_headlines"]]
 .sort_values(["target", "abs_return"], ascending=[True, False])
 .groupby("target").head(6).round(4).reset_index(drop=True))
        """),

        md("""
## 5. Findings

1. **The channels pass the event check for financial shocks, and only partly
   for war.** The April 2025 tariffs produce the largest trade-channel spike in
   every encoder (z = 3.2 to 7.5, rank 1 of 7). SVB moves the four financial
   channels (dollar, risk, fed, EM) and leaves geo at its lowest; risk-off and
   EM-outflow directions both point USD-up, as theory says. After 7 October
   the geo channel does spike (SBERT z = 2.4) and points toward escalation, but
   EM and oil spike more. Geo ranks 3rd to 6th, so the geo axis is the weakest
   of the three.

2. **Text features carry about half of each C5 model's attributions.** Channel
   scores and novelty account for 52% (direction) and 55% (large move) of mean
   |SHAP| on test. The models use the text heavily; the ablation shows this
   does not turn into a reliable gain.

3. **Retrieval exposes the encoder difference.** For large moves (FastText),
   the headlines behind the biggest text-driven calls are on-topic: Strait of
   Hormuz, US–Israel strikes on Iran, oil supply. For direction (GloVe), the
   top-k pooling often picks off-topic headlines ("Bakery tourism is booming"
   counted as trade news). GloVe's topic relevance is too loose to trust.

4. **News helps on novel, big-move days.** On large moves, the 20 days where C5
   was right and A wrong have nearly twice the absolute return (0.36% vs
   0.19%) and much more novel news (nov_centroid 0.82 vs 0.68) than the days
   where only A was right. Direction shows no such pattern.

5. **The map is weak evidence.** The first two components explain 6.5% of
   variance, and at most 13% of any channel axis lies in that plane. The
   channel axes run mostly orthogonal to the corpus's main topic directions.
   This is expected (they contrast minimal pairs within a topic), but it means
   the 2-D picture shows little of what the axes measure.
        """),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


NOTEBOOKS = {
    "01": ("01_eda.ipynb", build_eda),
    "02": ("02_baseline_results.ipynb", build_results),
    "03": ("03_ablation.ipynb", build_ablation),
    "04": ("04_feature_analysis.ipynb", build_feature_analysis),
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate the project notebooks.")
    parser.add_argument("--only", nargs="+", choices=sorted(NOTEBOOKS),
                        help="regenerate only these notebooks (default: all)")
    args = parser.parse_args()

    NB_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.only or sorted(NOTEBOOKS):
        name, builder = NOTEBOOKS[key]
        path = NB_DIR / name
        nbf.write(builder(), path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
