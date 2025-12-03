# viz_departures.py
import matplotlib.pyplot as plt

def plot_absence_length_hist(departures):
    """Histogram of how long star absences are."""
    plt.figure()
    departures["absence_length"].hist(
        bins=range(1, departures["absence_length"].max() + 2)
    )
    plt.xlabel("Absence length (consecutive games missed)")
    plt.ylabel("Number of events")
    plt.title("Distribution of star departure absence lengths")
    plt.tight_layout()
    plt.show()

def plot_pre_run_vs_absence(departures):
    """Scatter: how many games played before vs how long they then miss."""
    plt.figure()
    plt.scatter(departures["pre_run_length"], departures["absence_length"])
    plt.xlabel("Pre-run length (games played before absence)")
    plt.ylabel("Absence length (games missed)")
    plt.title("Pre-departure run vs absence length")
    plt.tight_layout()
    plt.show()

def plot_events_per_team(departures):
    """Bar chart: how many departure events per team."""
    events_per_team = departures["team_id"].value_counts().sort_values(ascending=False)
    plt.figure()
    events_per_team.plot(kind="bar")
    plt.xlabel("Team")
    plt.ylabel("Number of star departures")
    plt.title("Star departure events per team")
    plt.tight_layout()
    plt.show()

def plot_events_per_season(departures):
    """Bar chart: departures per season."""
    events_per_season = departures["season"].value_counts().sort_index()
    plt.figure()
    events_per_season.plot(kind="bar")
    plt.xlabel("Season")
    plt.ylabel("Number of star departures")
    plt.title("Star departure events per season")
    plt.tight_layout()
    plt.show()
