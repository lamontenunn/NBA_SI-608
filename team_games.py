# team_games.py
import pandas as pd

def build_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, team_id, game_id) with a chronological index (team_game_index).

    Uses both AwayTeam and HomeTeam to create team-specific rows.
    """
    # Unique games with home/away teams + date
    games_unique = (
        pbp[["season", "season_start_year", "game_id", "Date", "AwayTeam", "HomeTeam"]]
        .drop_duplicates(subset=["game_id"])
        .copy()
    )

    games_unique["game_date"] = pd.to_datetime(games_unique["Date"])

    # Expand to one row per team per game
    away_rows = games_unique[["season", "season_start_year", "game_id", "game_date", "AwayTeam"]].copy()
    away_rows = away_rows.rename(columns={"AwayTeam": "team_id"})

    home_rows = games_unique[["season", "season_start_year", "game_id", "game_date", "HomeTeam"]].copy()
    home_rows = home_rows.rename(columns={"HomeTeam": "team_id"})

    team_games = pd.concat([away_rows, home_rows], ignore_index=True).drop_duplicates()

    # Sort by date within season-team and give a running index
    team_games = team_games.sort_values(["season_start_year", "team_id", "game_date", "game_id"])
    team_games["team_game_index"] = team_games.groupby(["season_start_year", "team_id"]).cumcount()

    return team_games
