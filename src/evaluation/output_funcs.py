"""
Functions to structure and serve the output of our retrieval algorithms and other results stuff.
"""

from typing import List, Dict, Tuple
from llama_index.core.schema import Document
from datetime import datetime
from pathlib import Path

import pickle


def format_output(
    top_k_docs: List[Tuple[str, float]],
    query: str,
    retrieval_algorithm: str,
    document_index: List[Document],
) -> Dict[str, str]:
    """A function to format the results of the retrieval algorithm into a neat dictionary.
    This dictionary will be used for evaluation, presentation, and logging

    To make these results packages manageable, we constrain what we log. For example, the document index only
    stores metadata - making it 6 times smaller (min) than the full document objects.

    Args:
        top_k_docs (List[Document]): The result from the retrieval algorithm
        query (str): The query input into the retrieval algorithm. Can also be
        retrieval_algorithm (str): The name of the retrieval algorithm used
        document_index (List[Document]): The document index used for retrieval

    Returns:
        Dict[str, str]: Package of outputs for evaluation, presentation, and logging
    """

    # Store contextual information about the process
    results_dict = {
        "query": query,
        "retrieval algo": retrieval_algorithm,
        "document index": {
            doc.metadata["file_name"]: doc.metadata for doc in document_index
        },
    }

    # Store the results of the process itself
    retrieval_output = {}

    # Iterate over the top_k_docs object
    for i, doc in enumerate(top_k_docs):
        doc_id, score = doc
        document = [doc for doc in document_index if doc.id_ == doc_id][0]

        # Store the document in an intermediate dictionary
        retrieval_output["result_" + str(i)] = {
            "score": score,
            "document": document,
        }

    # Add the results to the results_dict
    results_dict["retrieval output"] = retrieval_output

    return results_dict


def save_results(
    results_dict: Dict[str, str],
) -> None:
    """A function to save the results of the retrieval algorithm to a pickle file.

    Args:
        results_dict (Dict[str, str]): The dictionary containing the results of the retrieval algorithm
    """
    # Get the path of the current file
    current_path = Path(__file__)

    # Construct the path to the directory
    output_path = current_path.parent.parent.parent / "data/03_output"

    # Get the current date and time
    now = datetime.now()

    # Format the date and time to create a unique identifier (assuming we're operating a <1 a second)
    timestamp_str = now.strftime("%Y%m%d%H%M%S")

    # Use the timestamp in the output file path
    output_path = output_path / f"output_{timestamp_str}.pickle"

    with open(output_path, "wb") as f:
        pickle.dump(results_dict, f)
