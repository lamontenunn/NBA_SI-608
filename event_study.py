# event_study.py
import pandas as pd

def build_departure_event_panel(
    departures: pd.DataFrame,
    team_games: pd.DataFrame,
    team_metrics: pd.DataFrame,
    window_before: int = 10,
    window_after: int = 10,
) -> pd.DataFrame:
    """
    For each departure event, construct a window of games around the first missed game.

    Returns rows with:
      event_id, season, season_start_year, team_id,
      rel_game, game_id, team_game_index, <metric columns...>

    where rel_game=0 is the first missed game.
    """
    # attach index of first missed game
    events_with_idx = departures.merge(
        team_games[["season", "season_start_year", "team_id", "game_id", "team_game_index"]],
        left_on=["season", "season_start_year", "team_id", "first_missed_game_id"],
        right_on=["season", "season_start_year", "team_id", "game_id"],
        how="left",
        suffixes=("", "_event"),
    )

    panel_rows = []

    for _, ev in events_with_idx.iterrows():
        base_idx = ev["team_game_index"]  # index of first missed game
        season = ev["season"]
        season_start_year = ev["season_start_year"]
        team_id = ev["team_id"]
        event_id = ev["event_id"]

        # games for this team around this index
        mask = (
            (team_games["season"] == season)
            & (team_games["season_start_year"] == season_start_year)
            & (team_games["team_id"] == team_id)
            & (team_games["team_game_index"].between(base_idx - window_before, base_idx + window_after))
        )

        g_subset = team_games.loc[mask, ["game_id", "team_game_index"]].copy()
        if g_subset.empty:
            continue

        g_subset["rel_game"] = g_subset["team_game_index"] - base_idx

        # all metric columns = everything except keys
        metric_cols = [
            c
            for c in team_metrics.columns
            if c not in ["season", "season_start_year", "team_id", "game_id", "team_game_index", "game_date"]
        ]

        merged = g_subset.merge(
            team_metrics[
                ["season", "season_start_year", "team_id", "game_id", "team_game_index"] + metric_cols
            ],
            on=["game_id", "team_game_index"],
            how="left",
        )

        merged["season"] = season
        merged["season_start_year"] = season_start_year
        merged["team_id"] = team_id
        merged["event_id"] = event_id

        panel_rows.append(merged)

    if not panel_rows:
        return pd.DataFrame()

    return pd.concat(panel_rows, ignore_index=True)
