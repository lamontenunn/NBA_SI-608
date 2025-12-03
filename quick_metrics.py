# quick_metrics.py
import pandas as pd

def compute_team_assists_per_game(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Simple example metric: number of assists recorded by each team in each game.

    Returns columns:
      season, season_start_year, team_id, game_id, assists
    """
    pbp = pbp.copy()

    # Assist events: text contains "(assist by"
    away_assist_mask = pbp["AwayPlay"].fillna("").str.contains("assist by")
    home_assist_mask = pbp["HomePlay"].fillna("").str.contains("assist by")

    away_assists = (
        pbp[away_assist_mask]
        .groupby(["season", "season_start_year", "AwayTeam", "game_id"], as_index=False)
        .size()
        .rename(columns={"AwayTeam": "team_id", "size": "assists"})
    )

    home_assists = (
        pbp[home_assist_mask]
        .groupby(["season", "season_start_year", "HomeTeam", "game_id"], as_index=False)
        .size()
        .rename(columns={"HomeTeam": "team_id", "size": "assists"})
    )

    assists = pd.concat([away_assists, home_assists], ignore_index=True)
    return assists
