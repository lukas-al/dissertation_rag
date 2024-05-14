"""
Script to do parallel processing for constructing the adjacency dictionary for the graph.
"""

import multiprocessing
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import networkx as nx
import inspect

def calc_graph_chunk(chunk, index, retrieval_class):
    """
    Calculate the adjacency dictionary for a given chunk of documents.

    Args:
        chunk (list): A list of documents to process.
        index (list): A list of documents to compare against.
        retrieval_class (class): The retrieval algorithm class to use.

    Returns:
        dict: The adjacency dictionary representing the graph.
    """
    # Instantiate the class within the function
    retrieval_algo = retrieval_class()

    adjacency_dict = {}
    for doc in tqdm(chunk, desc=f"Processing chunk {chunk[0].id_}"):
        adjacency_dict[doc.id_] = {}
        for doc2 in index:
            adjacency_dict[doc.id_][doc2.id_] = {
                "weight": retrieval_algo.calculate_distance(
                    doc,
                    doc2,
                )
            }

    return adjacency_dict


def calc_doc_adjacency(doc, index, retrieval_class):
    """
    Calculate the adjacency dict for a given document.

    Args:
        doc (Document): A document to process.
        index (list): A list of documents to compare against.
        retrieval_class (class): The retrieval algorithm class to use.

    Returns:
        dict: The adjacency dictionary representing the graph.
    """
    
    if inspect.isclass(retrieval_class):
        retrieval_class = retrieval_class()

    
    adjacency_dict = {}
    adjacency_dict[doc.id_] = {}
    for doc2 in index:
        adjacency_dict[doc.id_][doc2.id_] = {
            "weight": retrieval_class.calculate_distance(
                doc,
                doc2,
            )
        }

    return adjacency_dict


def calc_doc_vectors(doc, index, retrieval_class):
    """
    Calculate document vectors for a given document using a retrieval class.

    Args:
        doc: The document for which to calculate the vectors.
        index: The index of documents to compare against.
        retrieval_class: The retrieval class used to calculate the distance vector.

    Returns:
        A dictionary containing the document vectors.
    """
    if inspect.isclass(retrieval_class):
        retrieval_class = retrieval_class()
        
    vectors_dict = {}
    vectors_dict[doc.id_] = {}
    for doc2 in index:
        vectors_dict[doc.id_][doc2.id_] = {
            "vector": retrieval_class.calculate_distance_vector(
                doc,
                doc2,
            )
        }
        
    return vectors_dict
    
def chunkify(lst, n):
    """
    Splits a list into approximately equal-sized chunks.

    Args:
        lst (list): The list to be split.
        n (int): The number of chunks to create.

    Returns:
        list: A list of chunks, where each chunk is a sublist of the original list.

    Example:
        >>> lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> chunkify(lst, 3)
        [[1, 4, 7, 10], [2, 5, 8], [3, 6, 9]]
    """
    return [lst[i::n] for i in range(n)]


def construct_adjacency_dict_parallel(embedded_index, retrieval_class):
    """
    Constructs and returns an adjacency dictionary based on the given embedded index and retrieval class.
    Uses parallel processing to speed up the construction.
    ! Not recommended for now - current algorithms are not worth parallelizing
    ! @Todo - explore threading

    Args:
        embedded_index (dict): The embedded index used for constructing the adjacency dictionary.
        retrieval_class (class): The retrieval class used for calculating the adjacency between nodes.

    Returns:
        dict: The constructed adjacency dictionary.

    """
    num_processes = multiprocessing.cpu_count() - 1
    chunks = chunkify(embedded_index.copy(), num_processes)

    with multiprocessing.Pool(num_processes) as pool:
        try:
            results = list(
                pool.map(
                    partial(
                        calc_graph_chunk,
                        index=embedded_index.copy(),
                        retrieval_class=retrieval_class,
                    ),
                    chunks,
                ),
            )
        finally:
            pool.close()
            pool.join()

    adjacency_dict = {k: v for result in results for k, v in result.items()}

    return adjacency_dict


def construct_adjacency_dict(embedded_index, retrieval_class):
    """
    Constructs an adjacency dictionary based on the given embedded index and retrieval class.
    Single process version of the function.
    ! This is actually the recommended version for the current algorithms

    Args:
        embedded_index (list): A list of documents with embedded representations.
        retrieval_class (object): An instance of the retrieval class used to calculate distances.

    Returns:
        dict: An adjacency dictionary where each document ID is mapped to a dictionary of neighboring document IDs and their weights.
    """
    job_list = []
    adjacency_dict = defaultdict(dict)
    
    for i, doc0 in enumerate(embedded_index):
        for doc1 in embedded_index[i+1:]:
            job_list.append((doc0, doc1))
        
    for job in tqdm(job_list, 'Processing documents'):
        doc, doc2 = job
        adjacency_dict[doc.id_][doc2.id_] = {
            "weight": retrieval_class.calculate_distance(
                doc,
                doc2,
            )
        }
    
    return adjacency_dict


def construct_adjacency_dict_multithread(embedded_index, retrieval_class):
    pass

def construct_graph_from_adj_dict(adj_dict, edge_thresh, embedded_index) -> nx.Graph:
    """
    Constructs a graph from an adjacency dictionary.

    Parameters:
        adj_dict (dict): The adjacency dictionary representing the graph.
        edge_thresh (float): The threshold value for edge weights.
        embedded_index (list): The list of embedded documents.

    Returns:
        nx.Graph: The constructed graph.
    """

    # Copy the adjacency dict
    adj_dict_local = adj_dict.copy()

    # Iterate over the adjacency dict to remove every edge which links a node to the same node
    for node0, edge_dict in adj_dict_local.items():
        for node1, weight in list(edge_dict.items()):
            # Remove matching nodes
            if node0 == node1:
                del adj_dict_local[node0][node1]

    # Construct a graph
    graph = nx.Graph()

    # Add the nodes
    for node in adj_dict_local.keys():
        matching_doc = [doc for doc in embedded_index if doc.id_ == node][0]
        graph.add_node(
            node,
            **matching_doc.metadata,
            date_num=int(pd.Timestamp(matching_doc.metadata["Date"]).timestamp()),
        )

    # Add the edges
    for node0, edge_bunch in adj_dict.items():
        for node1, edge_weight in edge_bunch.items():
            if edge_weight["weight"] > edge_thresh:
                graph.add_edge(node0, node1, weight=edge_weight["weight"])

    return graph
