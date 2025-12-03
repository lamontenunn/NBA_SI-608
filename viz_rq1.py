import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# ------------------------------------------------------------
# RQ1: Do cohesive networks associate with success?
#
# This updated module removes the incorrect rel_game‑based facet
# logic and replaces it with 3 simple histograms:
#   - net_density
#   - net_clustering
#   - net_reciprocity
#
# These work correctly on team_metrics (which has no rel_game).
# ------------------------------------------------------------

def plot_rq1_histograms(team_metrics: pd.DataFrame):
    """
    Plot three simple histograms for the network cohesion metrics:
    density, clustering, reciprocity.
    """
    metrics = [
        ("net_density", "Network Density"),
        ("net_clustering", "Network Clustering"),
        ("net_reciprocity", "Network Reciprocity"),
    ]

    plt.figure(figsize=(15, 4))

    for i, (metric, label) in enumerate(metrics):
        if metric not in team_metrics.columns:
            continue

        plt.subplot(1, 3, i + 1)
        team_metrics[metric].dropna().hist(bins=30)
        plt.title(label)
        plt.xlabel(label)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_rq1_scatter_relations(team_metrics: pd.DataFrame):
    """
    Scatter + regression plots:
      network metric vs point differential.
    This function remains unchanged and does not rely on rel_game.
    """

    df = team_metrics.dropna(
        subset=["net_density", "net_clustering", "net_reciprocity", "point_diff"]
    )

    plt.figure(figsize=(15, 4))

    metrics = [
        ("net_density", "Network Density"),
        ("net_clustering", "Network Clustering"),
        ("net_reciprocity", "Network Reciprocity"),
    ]

    for i, (metric, label) in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.regplot(data=df, x=metric, y="point_diff", scatter_kws={"s": 40})
        plt.title(f"{label} vs Point Differential")
        plt.xlabel(label)
        plt.ylabel("Point Differential")

    plt.tight_layout()
    plt.show()
