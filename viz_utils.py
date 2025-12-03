import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# ---------------------------------------------------------------------
# 1. Compute mean + confidence interval for a metric grouped by rel_game
# ---------------------------------------------------------------------
def compute_ci(df, metric, group_col="rel_game", ci=0.95):
    grouped = (
        df.groupby(group_col)[metric]
        .agg(["mean", "count"])
        .rename(columns={"count": "n"})
        .reset_index()
    )

    grouped["se"] = df.groupby(group_col)[metric].apply(lambda x: sem(x, nan_policy="omit")).values
    grouped["ci_low"] = grouped["mean"] - 1.96 * grouped["se"]
    grouped["ci_high"] = grouped["mean"] + 1.96 * grouped["se"]

    return grouped

# ---------------------------------------------------------------------
# 2. Line plot with confidence intervals
# ---------------------------------------------------------------------
def plot_with_ci(df, metric, title=None, xlabel="Games relative to event", ylabel=None):
    ci_df = compute_ci(df, metric)

    plt.figure(figsize=(8, 5))
    plt.plot(ci_df["rel_game"], ci_df["mean"], marker="o", label="Mean")
    plt.fill_between(
        ci_df["rel_game"],
        ci_df["ci_low"],
        ci_df["ci_high"],
        alpha=0.2,
        label="95% CI",
    )
    plt.axvline(0, linestyle="--", color="black", linewidth=1)

    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 3. Multi-metric facet grid (3 metrics side-by-side)
# ---------------------------------------------------------------------
def facet_metrics(df, metrics, titles=None, ylabel="Metric value"):
    n = len(metrics)
    plt.figure(figsize=(6 * n, 5))

    for i, metric in enumerate(metrics):
        ci_df = compute_ci(df, metric)

        plt.subplot(1, n, i + 1)
        plt.plot(ci_df["rel_game"], ci_df["mean"], marker="o")
        plt.fill_between(ci_df["rel_game"], ci_df["ci_low"], ci_df["ci_high"], alpha=0.2)
        plt.axvline(0, linestyle="--", color="black")

        ttl = titles[i] if titles else metric
        plt.title(ttl)
        plt.xlabel("Games relative to event")
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 4. Scatter + regression line for cross-sectional relations (RQ1)
# ---------------------------------------------------------------------
def scatter_with_regression(df, x, y, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x=x, y=y, scatter_kws={"s": 40}, line_kws={"color": "red"})
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 5. Box + violin comparison for pre vs post event windows
# ---------------------------------------------------------------------
def pre_post_violin(df, metric, pre_range=(-5, -1), post_range=(1, 5), title=None):
    pre = df[df["rel_game"].between(pre_range[0], pre_range[1])][metric].dropna()
    post = df[df["rel_game"].between(post_range[0], post_range[1])][metric].dropna()

    plt.figure(figsize=(7, 5))
    sns.violinplot(data=[pre, post])
    plt.xticks([0, 1], ["Pre", "Post"])
    plt.ylabel(metric)

    if title:
        plt.title(title)
    else:
        plt.title(f"{metric}: Pre vs Post Event")

    plt.tight_layout()
    plt.show()
