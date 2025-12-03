import pandas as pd
import numpy as np
import networkx as nx

from player_events import extract_player_id


def build_team_passing_edges(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build directed passer->shooter assist edges for each (season, team, game).

    Returns a DataFrame with columns:
      season, season_start_year, team_id, game_id,
      passer_id, shooter_id, weight
    where weight is the count of assists from passer to shooter in that game.
    """
    if "Assister" not in pbp.columns or "Shooter" not in pbp.columns:
        raise KeyError("Expected 'Assister' and 'Shooter' columns in pbp data.")

    pbp_local = pbp.copy()

    # Assisted baskets: rows where Assister is non-empty
    assist_mask = pbp_local["Assister"].notna() & (
        pbp_local["Assister"].astype(str).str.strip() != ""
    )

    edges = pbp_local.loc[
        assist_mask,
        [
            "season",
            "season_start_year",
            "game_id",
            "event_team",
            "Assister",
            "Shooter",
        ],
    ].copy()

    edges = edges.rename(columns={"event_team": "team_id"})

    # Convert raw player strings like "A. Drummond - drumman01" to stable IDs
    edges["passer_id"] = edges["Assister"].apply(extract_player_id)
    edges["shooter_id"] = edges["Shooter"].apply(extract_player_id)

    edges = edges[
        edges["passer_id"].notna()
        & (edges["passer_id"].astype(str).str.strip() != "")
        & edges["shooter_id"].notna()
        & (edges["shooter_id"].astype(str).str.strip() != "")
    ].copy()

    grouped = (
        edges.groupby(
            [
                "season",
                "season_start_year",
                "team_id",
                "game_id",
                "passer_id",
                "shooter_id",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "weight"})
    )

    return grouped


def compute_passing_network_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Given passer->shooter edges, compute simple network cohesion metrics
    for each (season, team, game):
      - net_n_players: number of distinct players in the network
      - net_n_edges: number of directed edges
      - net_density: edge density (directed)
      - net_clustering: average clustering (on undirected version)
      - net_reciprocity: fraction of edges that are reciprocated
    """
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "season_start_year",
                "team_id",
                "game_id",
                "net_n_players",
                "net_n_edges",
                "net_density",
                "net_clustering",
                "net_reciprocity",
            ]
        )

    records = []
    group_cols = ["season", "season_start_year", "team_id", "game_id"]

    for key, g in edges.groupby(group_cols):
        season, season_start_year, team_id, game_id = key

        G = nx.DiGraph()

        for _, row in g.iterrows():
            passer = row["passer_id"]
            shooter = row["shooter_id"]
            weight = row["weight"]
            if G.has_edge(passer, shooter):
                G[passer][shooter]["weight"] += weight
            else:
                G.add_edge(passer, shooter, weight=weight)

        n_players = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_players >= 2 and n_edges > 0:
            density = nx.density(G)
        else:
            density = np.nan

        if n_players >= 3 and n_edges > 0:
            # Use undirected version for a simple notion of clustering
            undirected = G.to_undirected()
            clustering_vals = nx.clustering(undirected, weight="weight")
            if clustering_vals:
                clustering = float(np.mean(list(clustering_vals.values())))
            else:
                clustering = np.nan
        else:
            clustering = np.nan

        reciprocity = nx.reciprocity(G) if n_edges > 0 else np.nan

        records.append(
            {
                "season": season,
                "season_start_year": season_start_year,
                "team_id": team_id,
                "game_id": game_id,
                "net_n_players": n_players,
                "net_n_edges": n_edges,
                "net_density": density,
                "net_clustering": clustering,
                "net_reciprocity": reciprocity,
            }
        )

    return pd.DataFrame.from_records(records)

