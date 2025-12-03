from pbp_loader import load_pbp
from player_events import make_player_events
from stars import compute_player_usage, flag_team_stars
from team_games import build_team_games
from appearances_and_departures import (
    build_player_game_appearances,
    build_star_games,
    detect_departures,
    summarize_departures,
)

from quick_metrics import compute_team_assists_per_game
from network_metrics import build_team_passing_edges, compute_passing_network_metrics
from event_study import build_departure_event_panel
from viz_departures import (
    plot_absence_length_hist,
    plot_pre_run_vs_absence,
    plot_events_per_team,
    plot_events_per_season,
)
from viz_event_study import plot_event_study, plot_pre_post_box

def main():
    # Load all available seasons from the NBA-Data directory (2015–2021)
    pbp = load_pbp("NBA-Data")

    events_long = make_player_events(pbp)
    usage = compute_player_usage(events_long)
    stars = flag_team_stars(usage, star_quantile=0.9)

    team_games = build_team_games(pbp)
    appearances = build_player_game_appearances(events_long)
    star_games = build_star_games(team_games, appearances, stars)

    departures = detect_departures(star_games, min_pre_run=5, min_absence=3)
    print(summarize_departures(departures))

    # --- 1) Visuals about the departures themselves ---
    plot_absence_length_hist(departures)
    plot_pre_run_vs_absence(departures)
    plot_events_per_team(departures)
    plot_events_per_season(departures)

    # --- 2) Build metrics (assists + passing network) and team_metrics ---
    assists = compute_team_assists_per_game(pbp)

    passing_edges = build_team_passing_edges(pbp)
    net_metrics = compute_passing_network_metrics(passing_edges)

    team_metrics = (
        team_games.merge(
            assists,
            on=["season", "season_start_year", "team_id", "game_id"],
            how="left",
        )
        .merge(
            net_metrics,
            on=["season", "season_start_year", "team_id", "game_id"],
            how="left",
        )
    )

    team_metrics["assists"] = team_metrics["assists"].fillna(0)

    # --- 3) Build event-study panel and plot around departures ---
    event_panel = build_departure_event_panel(
        departures=departures,
        team_games=team_games,
        team_metrics=team_metrics,
        window_before=10,
        window_after=10,
    )

    # Assists around departures
    plot_event_study(event_panel, metric="assists", title_suffix="assists per game")
    plot_pre_post_box(event_panel, metric="assists")

    # Passing network cohesion metrics around departures
    for metric, label in [
        ("net_density", "network density"),
        ("net_clustering", "network clustering"),
        ("net_reciprocity", "network reciprocity"),
    ]:
        if metric in event_panel.columns:
            plot_event_study(
                event_panel,
                metric=metric,
                title_suffix=label,
            )
            plot_pre_post_box(event_panel, metric=metric)

if __name__ == "__main__":
    main()
