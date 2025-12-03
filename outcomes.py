# outcomes.py
import pandas as pd


def compute_team_outcomes(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Derive game-level outcomes for each team (points, point differential, win flag).

    The input play-by-play must include: season, season_start_year, game_id,
    AwayTeam, HomeTeam, AwayScore, HomeScore.
    """
    required_cols = [
        "season",
        "season_start_year",
        "game_id",
        "AwayTeam",
        "HomeTeam",
        "AwayScore",
        "HomeScore",
    ]
    missing = [c for c in required_cols if c not in pbp.columns]
    if missing:
        raise KeyError(f"Missing required columns in pbp: {missing}")

    pbp_local = pbp.copy()
    score_cols = ["AwayScore", "HomeScore"]
    pbp_local[score_cols] = pbp_local[score_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Final scores per game: scores only increase, so max gives the final.
    game_scores = (
        pbp_local.groupby(
            ["season", "season_start_year", "game_id", "AwayTeam", "HomeTeam"],
            as_index=False,
        )[score_cols]
        .max()
        .rename(columns={"AwayTeam": "away_team", "HomeTeam": "home_team"})
    )

    away_rows = game_scores.rename(columns={"away_team": "team_id"}).copy()
    away_rows["points_for"] = away_rows["AwayScore"]
    away_rows["points_against"] = away_rows["HomeScore"]

    home_rows = game_scores.rename(columns={"home_team": "team_id"}).copy()
    home_rows["points_for"] = home_rows["HomeScore"]
    home_rows["points_against"] = home_rows["AwayScore"]

    outcomes = pd.concat([away_rows, home_rows], ignore_index=True)
    outcomes["point_diff"] = outcomes["points_for"] - outcomes["points_against"]
    outcomes["total_points"] = outcomes["points_for"] + outcomes["points_against"]
    outcomes["win"] = (outcomes["point_diff"] > 0).astype(int)

    return outcomes[
        [
            "season",
            "season_start_year",
            "team_id",
            "game_id",
            "points_for",
            "points_against",
            "point_diff",
            "total_points",
            "win",
        ]
    ]
