"""
Perform hyperparameter tuning on the edge threshold
"""

from functools import partial

import hyperopt
import joblib
import networkx as nx
import numpy as np

from StructuredRag.evaluation import graph_scoring
from StructuredRag.processing import graph_construction


def tune_edgethresh(adj_mat, *args, **kwargs) -> float:
    """
    Tune the edge threshold to produce a graph where 20% of the original edges are kept.

    Parameters:
    - adj_mat (dict): The adjacency matrix representing the graph.

    Returns:
    - float: The optimal edge threshold.

    """
    edge_weights = [d2["weight"] for d in adj_mat.values() for d2 in d.values()]

    search_space = np.linspace(np.min(edge_weights), np.max(edge_weights), 50)

    results = []
    for n in search_space:
        proportion_above_thresh = len([x for x in edge_weights if x > n]) / len(
            edge_weights
        )
        results.append((n, proportion_above_thresh))

    # Find nearest where proportion above threshold is 0.2
    nearest = min(results, key=lambda x: abs(x[1] - 0.2))
    print(f"Optimal edge threshold, edge_thresh: {nearest[0]}, proportion {nearest[1]}")

    return nearest[0]


def tune_edgethresh_multicrit(adj_mat, embedded_index) -> float:
    """Select an edge threshold based on the tradeoff
    of a few different variables.

    Args:
        adj_mat (_type_): _description_
        embedded_index (_type_): _description_

    Returns:
        float: _description_
    """

    print("Optimising the edge threshold")

    search_space = {"edge_thresh": hyperopt.hp.uniform("edge_thresh", -1, 2)}

    trials = hyperopt.Trials()

    with joblib.parallel_backend("loky", n_jobs=1):
        best_params = hyperopt.fmin(
            fn=partial(objective, adj_mat, embedded_index),
            space=search_space,
            algo=hyperopt.atpe.suggest,
            max_evals=25,
            trials=trials,
        )

    return best_params["edge_thresh"]


def objective(adj_mat, embedded_index, params):
    """This is our function to return the score to minimise based on our input
    Hyperopt minimises the objective function by default.

    Args:
        params (_type_): _description_
    """
    print(f"Trying edge threshold: {params['edge_thresh']}")

    graph = graph_construction.construct_graph_from_adj_dict(
        adj_dict=adj_mat,
        edge_thresh=params["edge_thresh"],
        embedded_index=embedded_index,
    )

    try:
        graph_scores = graph_scoring.quick_stats(graph)

        # I want average degree to be around 10. Square to penalise large differences much more
        # Expect this to be around 10 max once it hits equilibrium
        avg_deg_penalty = (10 - graph_scores["average_degree"]) ** 2

        # Need to make sure every node is connected
        if not nx.is_connected(graph):
            connected_penalty = 1000
        else:
            connected_penalty = 0

        # Try to encourage our graph having some decent modularity
        graph_modularity = graph_scoring.community_detection(graph)

        # Higher is better, but not super emphasized. So make it negative and *10.
        modularity_penalty = -(graph_modularity["community_modularity"]) * 10

        return avg_deg_penalty + connected_penalty + modularity_penalty

    except ZeroDivisionError:
        return 10000
