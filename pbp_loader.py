# pbp_loader.py
import pandas as pd
import numpy as np
from pathlib import Path

def infer_season_from_date(date_series: pd.Series) -> pd.Series:
    """
    Given a 'Date' column like 'October 27 2015', return season start year, e.g. 2015 for 2015-16.
    """
    dt = pd.to_datetime(date_series)
    year = dt.dt.year
    month = dt.dt.month
    # NBA season starts in Oct; games in Jan–Jun still belong to previous season
    season_start_year = np.where(month >= 10, year, year - 1)
    return season_start_year.astype(int)

def load_pbp(path) -> pd.DataFrame:
    """
    Load play-by-play CSV(s) with columns:
    URL,GameType,Location,Date,Time,WinningTeam,Quarter,SecLeft,AwayTeam,AwayPlay,
    AwayScore,HomeTeam,HomePlay,HomeScore,... etc.

    - Adds: game_id, season, event_team
    - Keeps all original columns.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        dfs = [pd.read_csv(f) for f in files]
        pbp = pd.concat(dfs, ignore_index=True)
    else:
        pbp = pd.read_csv(path)

    # Use URL as game_id (it’s unique per game)
    pbp["game_id"] = pbp["URL"]

    # Infer season from Date
    pbp["season_start_year"] = infer_season_from_date(pbp["Date"])
    # If you want the pretty '2015-16' format:
    pbp["season"] = (
        pbp["season_start_year"].astype(str)
        + "-"
        + (pbp["season_start_year"] + 1).astype(str).str[-2:]
    )

    # Determine which team generated the play text
    away_has_play = pbp["AwayPlay"].fillna("").str.strip() != ""
    home_has_play = pbp["HomePlay"].fillna("").str.strip() != ""

    pbp["event_team"] = pd.Series(pd.NA, index=pbp.index, dtype="object")

    pbp.loc[away_has_play, "event_team"] = pbp.loc[away_has_play, "AwayTeam"].astype("object")
    pbp.loc[~away_has_play & home_has_play, "event_team"] = (
    pbp.loc[~away_has_play & home_has_play, "HomeTeam"].astype("object")
    )

    # Ensure Quarter and SecLeft are numeric for ordering
    pbp["Quarter"] = pbp["Quarter"].astype(int)
    pbp["SecLeft"] = pbp["SecLeft"].astype(int)

    # Optional: sort by game, then by quarter/time (descending seconds left)
    pbp = pbp.sort_values(["season_start_year", "game_id", "Quarter", "SecLeft"], ascending=[True, True, True, False])

    return pbp
