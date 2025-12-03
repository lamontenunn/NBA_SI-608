from pbp_loader import load_pbp
from player_events import make_player_events
from stars import compute_player_usage, flag_team_stars
from team_games import build_team_games
from outcomes import compute_team_outcomes
from appearances_and_departures import (
    build_player_game_appearances,
    build_star_games,
    detect_departures,
    summarize_departures,
)

from quick_metrics import compute_team_assists_per_game
from network_metrics import build_team_passing_edges, compute_passing_network_metrics
from event_study import build_departure_event_panel

# New visualization modules (RQ1, RQ2, RQ3)
from viz_rq1 import (
    plot_rq1_histograms,
    plot_rq1_scatter_relations,
)
from viz_rq2 import (
    plot_rq2_ci,
    plot_rq2_facets,
    plot_rq2_pre_post,
)
from viz_rq3 import (
    plot_rq3_logit_coefficients,
    plot_rq3_feature_importance,
)

def main():
    # Load all available seasons from the NBA-Data directory (2015–2021)
    pbp = load_pbp("NBA-Data")

    # --- Build player-level and team-level activity ---
    events_long = make_player_events(pbp)
    usage = compute_player_usage(events_long)
    stars = flag_team_stars(usage, star_quantile=0.9)

    team_games = build_team_games(pbp)
    appearances = build_player_game_appearances(events_long)
    star_games = build_star_games(team_games, appearances, stars)
    team_outcomes = compute_team_outcomes(pbp)

    # Detect star departures
    departures = detect_departures(star_games, min_pre_run=5, min_absence=3)
    print(summarize_departures(departures))

    # --- Build team_metrics (assists + network metrics + outcomes) ---
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
        .merge(
            team_outcomes,
            on=["season", "season_start_year", "team_id", "game_id"],
            how="left",
        )
    )

    team_metrics["assists"] = team_metrics["assists"].fillna(0)

    # --- Build event-study panel ---
    event_panel = build_departure_event_panel(
        departures=departures,
        team_games=team_games,
        team_metrics=team_metrics,
        window_before=10,
        window_after=10,
    )

    # ===========================================================
    # RQ1: Do cohesive networks associate with team success?
    # ===========================================================
    plot_rq1_histograms(team_metrics)
    plot_rq1_scatter_relations(team_metrics)

    # ===========================================================
    # RQ2: How do key-player departures affect network cohesion?
    # ===========================================================
    plot_rq2_ci(event_panel)
    plot_rq2_facets(event_panel)
    plot_rq2_pre_post(event_panel)

    # ===========================================================
    # RQ3: Which network metrics best predict team success?
    # ===========================================================
    plot_rq3_logit_coefficients(team_metrics)
    plot_rq3_feature_importance(team_metrics)

if __name__ == "__main__":
    main()
