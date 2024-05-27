"""
Functions for the evaluation of the retrieval algorithms.
"""

#! @TODO CURRENTLY NOT SURE HOW TO IMPLEMENT THIS...


# Eval 1: 20 or so test cases where we compare the similarity of the retrieval output with what I'd expect to return.
# Eval 2: Construct a graph using the similarity between nodes. Then evaluate the suitability of the graphs.
# Eval 3: Compare the results of the retrieval algorithms with the results of different algorithms.
# Eval 4:

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from typing import List, Dict, Any, AnyStr
from sklearn.metrics import f1_score, precision_score, recall_score
from llama_index.core.schema import Document


def construct_graph_top_k(
    embedded_index: List[Document],
    retrieval_algorithm: callable,
    edge_threshold: float = 0.5,
    **kwargs,
) -> nx.Graph:
    """
    Constructs a graph based on the similarity of the documents, using a top k retrieval algorithm.
    The top k algorithm is pretty inneficent, but it's a good starting point for the evaluation.

    Parameters:
    - embedded_index (List[Document]): A list of embedded documents.
    - retrieval_algorithm (callable): A callable function for retrieving similar documents.
    - **kwargs: Additional keyword arguments for the retrieval_algorithm.

    Returns:
    - graph (nx.Graph): A networkx graph representing the connections between documents.
    """

    # Create an empty graph
    graph = nx.Graph()

    # Create a dictionary to store the connections
    connections = {}

    # Iterate over each document in the embedded_index with tqdm progress bar
    for doc in tqdm.tqdm(embedded_index, desc="Constructing Graph"):
        # Retrieve the top k documents using retrieval_algorithm
        retrieved_docs = retrieval_algorithm(
            np.array(doc.embedding), embedded_index, **kwargs
        )

        # Add the document as a node in the graph
        graph.add_node(
            doc.id_,
            **doc.metadata,
            date_num=int(pd.Timestamp(doc.metadata["Date"]).timestamp()),
        )

        # Add the connections to the dictionary
        connections[doc.id_] = [
            (retrieved_doc[0], retrieved_doc[1])
            for retrieved_doc in retrieved_docs
            if retrieved_doc[0] != doc.id_
        ]

    # Add the connections to the graph
    for node, connected_nodes in connections.items():
        graph.add_weighted_edges_from(
            [
                (node, c_nd[0], c_nd[1][0][0])
                for c_nd in connected_nodes
                if c_nd[1][0][0] > edge_threshold
            ]
        )

    return graph


def construct_graph_adjacency_matrix(
    embedded_index: List[Document],
    distance_metric: callable,
    edge_threshold: float = 0.5,
) -> nx.Graph:
    """Construct a graph based on using an adjacency matrix approach. More efficient than using the retrieval algorithm directly."""

    return


def quick_stats(graph: nx.Graph) -> Dict[AnyStr, Any]:
    """
    Calculate various statistics for a given graph.

    Parameters:
        graph (nx.Graph): The input graph.

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    # print("Calculating quick stats")

    stats = {}

    stats["number_of_nodes"] = nx.number_of_nodes(graph)
    stats["number_of_edges"] = nx.number_of_edges(graph)
    stats["average_degree"] = sum(dict(graph.degree()).values()) / float(
        nx.number_of_nodes(graph)
    )
    stats["density"] = nx.density(graph)
    stats["clustering_coefficient"] = nx.average_clustering(graph)
    # stats["average_node_connectivity"] = nx.average_node_connectivity(graph)
    # stats["edge_connectivity"] = nx.edge_connectivity(graph)
    # stats["node_connectivity"] = nx.node_connectivity(graph)

    return stats


def simple_stats(graph: nx.Graph) -> Dict[AnyStr, Any]:
    """
    Calculate simple statistics of a given graph.

    Parameters:
    graph (nx.Graph): The input graph.

    Returns:
    dict: A dictionary containing the following statistics:
        - number_of_nodes: The number of nodes in the graph.
        - number_of_edges: The number of edges in the graph.
        - average_degree: The average degree of the nodes in the graph.
        - density: The density of the graph.
        - clustering_coefficient: The average clustering coefficient of the graph.
        - average_shortest_path_length: The average shortest path length in the graph (if the graph is connected).
        - diameter: The diameter of the graph (if the graph is connected).
        - average_node_connectivity: The average node connectivity of the graph.
        - edge_connectivity: The edge connectivity of the graph.
        - node_connectivity: The node connectivity of the graph.
    """
    # print("Calculating simple stats")
    stats = {}

    stats["number_of_nodes"] = nx.number_of_nodes(graph)
    stats["number_of_edges"] = nx.number_of_edges(graph)
    stats["average_degree"] = sum(dict(graph.degree()).values()) / float(
        nx.number_of_nodes(graph)
    )
    stats["density"] = nx.density(graph)
    stats["clustering_coefficient"] = nx.average_clustering(graph)
    stats["average_node_connectivity"] = nx.average_node_connectivity(graph)
    stats["edge_connectivity"] = nx.edge_connectivity(graph)
    stats["node_connectivity"] = nx.node_connectivity(graph)

    if nx.is_connected(graph):
        stats["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
        stats["diameter"] = nx.diameter(graph)
    else:
        stats["average_shortest_path_length"] = None
        stats["diameter"] = None

    return stats


def attribute_stats(graph: nx.Graph) -> Dict[AnyStr, Any]:
    """Calculate statistics based on some attributes

    Args:
        graph (nx.Graph): The input graph to calculate statistics on.

    Returns:
        dict: A dictionary containing various statistics calculated based on the attributes of the graph.
    """
    # print("Calculating attribute stats")
    stats = {}

    stats["degree_assortativity"] = nx.degree_assortativity_coefficient(graph)
    stats["source_assortativity"] = nx.attribute_assortativity_coefficient(
        graph, "file_name"
    )
    stats["page_assortativity"] = nx.attribute_assortativity_coefficient(
        graph, "page_label"
    )
    stats["date_assortativity"] = nx.numeric_assortativity_coefficient(
        graph, "date_num"
    )

    return stats


def community_detection(graph: nx.Graph) -> Dict[AnyStr, Any]:
    """Detect communities in the graph.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        Dict[str, Any]: A dictionary containing the following statistics:
            - 'num_communities': The number of communities detected.
            - 'communities': The list of communities detected as sets of node ids.
            - 'community_modularity': The modularity of our communities
    """
    # print("Calculating community detection")
    stats = {}

    communities = nx.community.greedy_modularity_communities(graph)

    stats["num_communities"] = len(communities)
    stats["communities"] = communities
    stats["community_modularity"] = nx.community.modularity(graph, communities)

    return stats


def node_classification(
    graph: nx.Graph, label_to_predict: AnyStr = "file_name"
) -> Dict[AnyStr, Any]:
    """
    Perform node classification on a graph using the specified label attribute.

    Args:
        graph (nx.Graph): The input graph.
        label_to_predict (AnyStr, optional): The label attribute to predict for each node. Defaults to 'file_name'.

    Returns:
        Dict[AnyStr, Any]: A dictionary containing the evaluation statistics for the node classification task.
    """
    # print("Calculating node classification")
    stats = {}

    node_class_preds_lclglbl = (
        nx.algorithms.node_classification.local_and_global_consistency(
            graph, label_name=label_to_predict
        )
    )
    node_class_preds_harmonic = nx.algorithms.node_classification.harmonic_function(
        graph, label_name=label_to_predict
    )

    node_class_true = [graph.nodes[node][label_to_predict] for node in graph.nodes]

    # Evaluate the predictions
    stats["precision_lclglbl"] = precision_score(
        node_class_true, node_class_preds_lclglbl, average="weighted"
    )
    stats["recall_lclglbl"] = recall_score(
        node_class_true, node_class_preds_lclglbl, average="weighted"
    )
    stats["f1_lclglbl"] = f1_score(
        node_class_true, node_class_preds_lclglbl, average="weighted"
    )

    stats["precision_harmonic"] = precision_score(
        node_class_true, node_class_preds_harmonic, average="weighted"
    )
    stats["recall_harmonic"] = recall_score(
        node_class_true, node_class_preds_harmonic, average="weighted"
    )
    stats["f1_harmonic"] = f1_score(
        node_class_true, node_class_preds_harmonic, average="weighted"
    )

    return stats


def evaluate_graph(graph: nx.Graph) -> Dict[AnyStr, Any]:
    """
    Wrapper function for all the other evaluations.

    Parameters:
        graph (nx.Graph): The input graph to be evaluated.

    Returns:
        dict: A dictionary containing the evaluation results, including simple statistics,
              attribute statistics, community detection results, and node classification results.
    """

    graph_stats = {}

    try:
        graph_stats.update(simple_stats(graph))
    except Exception as e:
        print(f"Error calculating simple stats: {str(e)}")

    try:
        graph_stats.update(attribute_stats(graph))
    except Exception as e:
        print(f"Error calculating attribute stats: {str(e)}")

    try:
        graph_stats.update(community_detection(graph))
    except Exception as e:
        print(f"Error calculating community detection: {str(e)}")

    try:
        graph_stats.update(node_classification(graph))
    except Exception as e:
        print(f"Error calculating node classification: {str(e)}")

    return graph_stats
