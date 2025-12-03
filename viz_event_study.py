# viz_event_study.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_event_study(event_panel: pd.DataFrame, metric: str,
                     window_before: int = 10, window_after: int = 10,
                     title_suffix: str = ""):
    """
    Plot mean of `metric` by rel_game across all departure events.
    """
    mask = event_panel["rel_game"].between(-window_before, window_after)
    df = event_panel.loc[mask].copy()

    grouped = df.groupby("rel_game")[metric].agg(["mean", "count"]).reset_index()

    plt.figure()
    plt.plot(grouped["rel_game"], grouped["mean"], marker="o")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Games relative to star departure (0 = first missed game)")
    plt.ylabel(f"Average {metric}")
    title = f"{metric} around star departures"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pre_post_box(event_panel: pd.DataFrame, metric: str,
                      pre_window=(-5, -1), post_window=(1, 5)):
    """
    Boxplot comparing `metric` before vs after departure.
    """
    pre = event_panel[
        event_panel["rel_game"].between(pre_window[0], pre_window[1])
    ][metric].dropna()
    post = event_panel[
        event_panel["rel_game"].between(post_window[0], post_window[1])
    ][metric].dropna()

    plt.figure()
    plt.boxplot([pre, post], labels=["Pre", "Post"])
    plt.ylabel(metric)
    plt.title(f"{metric} before vs after star departure")
    plt.tight_layout()
    plt.show()
