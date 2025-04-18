"""Version 0 of the retrieval algorithm - simply using the embeddings"""

from typing import List, Tuple
from llama_index.core.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util
from StructuredRag.etl.embedding_funcs import embed_query
from numpy import array


def retrieve_top_k_query(
    query: str, doc_index: List[Document], k: int = 5
) -> List[Tuple[str, float]]:
    """Retrieve the top k most similar documents to the query.
    This algorithm is expected to return a tuple of document id and similarity score.
    This algorithm will embed the query on the fly.

    Args:
        query (str): the query string
        doc_index (List[Document]): the list of documents
        k (int): the number of documents to retrieve

    Returns:
        List[Tuple[str, float]]: the top k most similar documents
    """
    # Get the query embedding
    query_embedding = embed_query(query)

    # Create a list to store the similarity scores and document ids
    similarity_scores = []

    # Iterate over the documents in the index
    for doc in doc_index:
        # Calculate the cosine similarity between the query embedding and document embedding
        # doc_embed_rshp = array(doc.embedding).reshape(1, -1)
        # similarity_score = cosine_similarity(
        #     query_embedding.reshape(1, -1), doc_embed_rshp
        # )

        # For dot score models
        similarity_score = float(util.dot_score(
            query_embedding, doc.embedding
        ))

        # Append the similarity score and document id to the list
        similarity_scores.append((doc.id_, similarity_score))

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the top k documents
    top_k = similarity_scores[:k]

    return top_k


def retrieve_top_k(
    query_embedding: str, doc_index: List[Document], k: int = 5
) -> List[Tuple[str, float]]:
    """Retrieve the top k most similar documents to the query.
    This algorithm is expected to return a tuple of document id and similarity score.
    This algorithm expects an embedded document to compare against.

    Args:
        query_embedding (str): the query embedding (pre-computed)
        doc_index (List[Document]): the list of documents
        k (int): the number of documents to retrieve

    Returns:
        List[Tuple[str, float]]: the top k most similar documents
    """

    # Create a list to store the similarity scores and document ids
    similarity_scores = []

    # Iterate over the documents in the index
    for doc in doc_index:
        # Calculate the cosine similarity between the query embedding and document embedding
        doc_embed_rshp = array(doc.embedding).reshape(1, -1)
        similarity_score = float(util.dot_product(
            query_embedding, doc_embed_rshp
        ))
        # Append the similarity score and document id to the list
        similarity_scores.append((doc.id_, similarity_score))

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the top k documents
    top_k = similarity_scores[:k]

    return top_k
