# stars.py
import pandas as pd

def compute_player_usage(events_long: pd.DataFrame) -> pd.DataFrame:
    """
    Crude usage metric: count of high-impact events per (season, team, player).
    We’ll use roles that clearly belong to the team with the ball:
      Shooter, Assister, Rebounder, TurnoverPlayer, FreeThrowShooter.
    """
    impact_roles = ["Shooter", "Assister", "Rebounder", "TurnoverPlayer", "FreeThrowShooter"]
    df = events_long[events_long["role"].isin(impact_roles)].copy()

    usage = (
        df.assign(event_count=1)
          .groupby(["season", "season_start_year", "team_id", "player_id"], as_index=False)["event_count"]
          .sum()
    )
    return usage

def flag_team_stars(usage: pd.DataFrame, star_quantile: float = 0.9) -> pd.DataFrame:
    """
    Label top 'star_quantile' players in usage within each (season, team) as stars.
    """
    usage = usage.copy()

    def mark_stars(group):
        q = group["event_count"].quantile(star_quantile)
        group["is_star"] = group["event_count"] >= q
        return group

    usage = usage.groupby(["season", "season_start_year", "team_id"], group_keys=False).apply(mark_stars)
    return usage
