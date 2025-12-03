# appearances_and_departures.py
import pandas as pd

def build_player_game_appearances(events_long: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, team_id, game_id, player_id) who appears in any of the roles
    we tracked in make_player_events.
    """
    app = (
        events_long[["season", "season_start_year", "team_id", "game_id", "player_id"]]
        .drop_duplicates()
        .copy()
    )
    app["played"] = 1
    return app


def build_star_games(team_games: pd.DataFrame,
                     appearances: pd.DataFrame,
                     stars: pd.DataFrame) -> pd.DataFrame:
    """
    For each star player, create a panel of all team games (with played/DNP indicator).
    """
    # Only stars
    star_players = stars[stars["is_star"]][["season", "season_start_year", "team_id", "player_id"]].drop_duplicates()

    # Expand to all games for that team in that season
    sg = star_players.merge(
        team_games[["season", "season_start_year", "team_id", "game_id", "team_game_index"]],
        on=["season", "season_start_year", "team_id"],
        how="left",
    )

    # Merge in appearance indicator
    sg = sg.merge(
        appearances[["season", "season_start_year", "team_id", "game_id", "player_id", "played"]],
        on=["season", "season_start_year", "team_id", "game_id", "player_id"],
        how="left",
    )

    sg["played"] = sg["played"].fillna(0).astype(int)

    return sg


def detect_departures(star_games: pd.DataFrame,
                      min_pre_run: int = 5,
                      min_absence: int = 3) -> pd.DataFrame:
    """
    Identify long-absence 'departure' events for star players.

    A departure = stretch of >= min_pre_run games played,
                   immediately followed by >= min_absence consecutive games missed.

    Returns one row per event with:
      event_id, season, season_start_year, team_id, player_id,
      first_missed_game_id, absence_length, pre_run_length.
    """
    events = []
    group_cols = ["season", "season_start_year", "team_id", "player_id"]

    for key, g in star_games.groupby(group_cols):
        season, season_start_year, team_id, player_id = key
        g = g.sort_values("team_game_index").reset_index(drop=True)

        played = g["played"].tolist()
        n = len(played)
        i = 0

        while i < n:
            if played[i] == 1:
                # Start of a played run
                start_play = i
                while i < n and played[i] == 1:
                    i += 1
                end_play = i - 1
                pre_len = end_play - start_play + 1

                # Now check the following absence run
                j = i
                while j < n and played[j] == 0:
                    j += 1
                absence_len = j - i

                if pre_len >= min_pre_run and absence_len >= min_absence:
                    first_missed_game_id = g.loc[i, "game_id"]
                    events.append({
                        "season": season,
                        "season_start_year": season_start_year,
                        "team_id": team_id,
                        "player_id": player_id,
                        "first_missed_game_id": first_missed_game_id,
                        "absence_length": absence_len,
                        "pre_run_length": pre_len,
                    })

                # Skip over absence block
                i = j
            else:
                i += 1

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df["event_id"] = range(len(events_df))
    return events_df

def summarize_departures(events_df: pd.DataFrame):
    if events_df.empty:
        return {"n_events": 0}
    return {
        "n_events": len(events_df),
        "median_absence": float(events_df["absence_length"].median()),
        "iqr_absence": events_df["absence_length"].quantile([0.25, 0.75]).tolist(),
        "n_players": int(events_df["player_id"].nunique()),
        "n_teams": int(events_df["team_id"].nunique()),
    }
