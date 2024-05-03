"""
V1 of the retrieval algorithm - using a weighted combination of embeddings and metadata
"""

#! @TODO TO BE IMPLEMENTED
from typing import List, Tuple
from llama_index.core.schema import Document
from sentence_transformers import SentenceTransformer
from .distance_metrics import calculate_distance_vector


def retrieve_top_k(
    query_doc: Document,
    doc_index: List[Document],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Retrieve the top k most similar documents from the document index based on a query document.

    Args:
        query_doc (Document): The query document for which to retrieve similar documents.
        doc_index (List[Document]): The list of documents in the index.
        k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the document ID and similarity score of the top k similar documents.
    """

    # Instantiate embedding model to be used @TODO: Move this to a config file
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a list to store the similarity scores and document ids
    similarity_scores = []

    # Iterate over the documents in the index
    for doc in doc_index:
        # Calculate the distance vector between the documents
        distance_vector = calculate_distance_vector(
            query_doc, doc, embed_model, fuzz_thresh=80
        )

        # Combine the distance vector into a single score.
        sim_score = sum(distance_vector) / len(distance_vector)

        # Append the similarity score and document id to the list
        similarity_scores.append((doc.id_, sim_score))

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    top_k = similarity_scores[:k]

    return top_k
