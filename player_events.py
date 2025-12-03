# player_events.py
import pandas as pd
import numpy as np

def extract_player_id(raw):
    """
    Extract 'drumman01' from 'A. Drummond - drumman01'.
    If format is unexpected, fall back to the full string.
    """
    if pd.isna(raw):
        return np.nan
    s = str(raw)
    parts = s.split(" - ")
    if len(parts) == 2:
        return parts[1]
    return s

def make_player_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Make a long-format table of player-game-team 'events' from:
      Shooter, Assister, Rebounder, TurnoverPlayer, FreeThrowShooter, EnterGame, LeaveGame.

    Assumptions:
      - These roles belong to pbp['event_team'] on that row.
    """
    pbp = pbp.copy()

    actor_cols = [
        "Shooter",
        "Assister",
        "Rebounder",
        "TurnoverPlayer",
        "FreeThrowShooter",
        "EnterGame",
        "LeaveGame",
    ]

    use_cols = [
        "season",
        "season_start_year",
        "game_id",
        "Date",
        "event_team",
        "Quarter",
        "SecLeft",
    ] + actor_cols

    df = pbp[use_cols].copy()

    long = df.melt(
        id_vars=["season", "season_start_year", "game_id", "Date", "event_team", "Quarter", "SecLeft"],
        value_vars=actor_cols,
        var_name="role",
        value_name="raw_player",
    )

    # Drop rows with no player
    long = long[long["raw_player"].notna() & (long["raw_player"].astype(str).str.strip() != "")]

    long["player_id"] = long["raw_player"].apply(extract_player_id)
    long["team_id"] = long["event_team"]

    # Keep just what we need for later
    long = long[
        [
            "season",
            "season_start_year",
            "game_id",
            "Date",
            "team_id",
            "player_id",
            "role",
            "Quarter",
            "SecLeft",
        ]
    ].reset_index(drop=True)

    return long
