import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# RQ3: Which network properties best predict team success?
#
# This module provides:
# - Logistic regression coefficient plot (density, clustering, reciprocity → win)
# - Simple feature importance visualization for these 3 metrics
#
# 
# This is descriptive, not cross‑validated modeling.
# It is meant only to visualize which cohesion metrics are predictive.
# ------------------------------------------------------------

def _fit_logit(df: pd.DataFrame):
    """
    Fit a basic logistic regression:
      win ~ net_density + net_clustering + net_reciprocity

    Returns fitted model and the list of feature names.
    """
    clean = df.dropna(
        subset=["win", "net_density", "net_clustering", "net_reciprocity"]
    ).copy()

    X = clean[["net_density", "net_clustering", "net_reciprocity"]]
    y = clean["win"]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=200)),
        ]
    )

    model.fit(X, y)

    feature_names = ["net_density", "net_clustering", "net_reciprocity"]
    coefs = model.named_steps["logit"].coef_[0]

    return feature_names, coefs


def plot_rq3_logit_coefficients(team_metrics: pd.DataFrame):
    """
    Plot logistic regression coefficients for predicting win.
    """
    feat, coef = _fit_logit(team_metrics)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=coef, y=feat, orient="h")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Logistic Regression Coefficients Predicting Win")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Network Metric")
    plt.tight_layout()
    plt.show()


def plot_rq3_feature_importance(team_metrics: pd.DataFrame):
    """
    Simple importance = |standardized coefficient|.
    """
    feat, coef = _fit_logit(team_metrics)
    importance = [abs(x) for x in coef]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=importance, y=feat, orient="h")
    plt.title("Feature Importance (|Logit Coefficients|)")
    plt.xlabel("Absolute Importance")
    plt.ylabel("Network Metric")
    plt.tight_layout()
    plt.show()
