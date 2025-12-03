import pandas as pd
from viz_utils import (
    plot_with_ci,
    facet_metrics,
    pre_post_violin,
)

# -------------------------------------------------------------------
# RQ2: How does the departure of a key player affect network structure?
#
# This module produces:
# - CI line plots for each cohesion metric over rel_game
# - Faceted multi‑metric panels (density, clustering, reciprocity)
# - Pre/Post violin comparison for each metric
# -------------------------------------------------------------------

def plot_rq2_ci(event_panel: pd.DataFrame):
    """
    Plot confidence‑interval line plots for each network metric
    around star departures.
    """
    for metric, label in [
        ("net_density", "Network Density"),
        ("net_clustering", "Network Clustering"),
        ("net_reciprocity", "Network Reciprocity"),
    ]:
        if metric in event_panel.columns:
            plot_with_ci(
                event_panel,
                metric=metric,
                title=f"{label} Around Star Departures",
                ylabel=label,
            )

def plot_rq2_facets(event_panel: pd.DataFrame):
    """
    Facet grid showing all 3 cohesion metrics together.
    """
    metrics = ["net_density", "net_clustering", "net_reciprocity"]
    titles = ["Density", "Clustering", "Reciprocity"]

    facet_metrics(
        event_panel,
        metrics=metrics,
        titles=titles,
        ylabel="Metric Value",
    )

def plot_rq2_pre_post(event_panel: pd.DataFrame):
    """
    Pre vs Post departure violin/box comparison for network metrics.
    """
    for metric in ["net_density", "net_clustering", "net_reciprocity"]:
        if metric in event_panel.columns:
            pre_post_violin(
                event_panel,
                metric=metric,
                pre_range=(-5, -1),
                post_range=(1, 5),
                title=f"Pre/Post Comparison for {metric}",
            )
