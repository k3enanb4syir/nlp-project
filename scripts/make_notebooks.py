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

3. **Adding NLP features did not produce a reliable improvement.** The combined
   models moved accuracy slightly down and MCC slightly up relative to their
   rate-only counterparts. With this sample size, neither movement is
   distinguishable from chance.

4. **TF-IDF overfits badly.** `logreg_tfidf` was the weakest model on test
   (accuracy 0.472, MCC −0.112) despite competitive validation numbers. 300
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


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, nb in (("01_eda.ipynb", build_eda()),
                     ("02_baseline_results.ipynb", build_results())):
        path = NB_DIR / name
        nbf.write(nb, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
