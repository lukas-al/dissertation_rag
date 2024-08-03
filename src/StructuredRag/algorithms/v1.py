"""
v1 of the retrieval algorithm - using a weighted combination of embeddings and metadata
"""

from llama_index.core.schema import Document
from .abstract_retriever import AbstractRetriever
from typing import List, Tuple

from ..processing.distance_metrics import calculate_distance_vector

class V1Retriever(AbstractRetriever):
    """
    V1 of the retrieval algorithm, scoring a linear average of the embeddings and metadata distances.

    This class implements the retrieval algorithm for retrieving similar documents from an embedded index.
    It calculates the similarity scores between a given document and the documents in the embedded index
    based on a linear average of the embeddings and metadata distances.

    Attributes:
        None

    Methods:
        retrieve_top_k_doc: Retrieve the top k documents from the embedded index based on their similarity scores.
        calculate_distance: Calculate the distance between two documents using a weighted combination of embeddings and metadata.
    """

    # @Override
    def retrieve_top_k_doc(
        self,
        doc1: Document,
        embedded_index: List[Document],
        k: int = 5,
        fuzzy_thresh: int = 80,
    ) -> List[Tuple[str, float]]:
        """Retrieve the top k documents from the embedded index based on their similarity scores.

        Args:
            doc1 (Document): The document for which to retrieve similar documents.
            embedded_index (List[Document]): The list of documents in the embedded index.
            k (int, optional): The number of top documents to retrieve. Defaults to 5.
            fuzzy_thresh (int, optional): The fuzzy threshold for similarity comparison. Defaults to 80.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing the document ID and its similarity score.
        """

        sim_scores = []
        for doc in embedded_index:
            print("Calcing the score for: ", doc.id_)
            sim_scores.append((doc.id_, self.calculate_distance(doc1, doc)))

        sim_scores.sort(key=lambda x: x[1], reverse=True)

        return sim_scores[:k]

    # @Override
    def calculate_distance(
        self,
        doc1: Document,
        doc2: Document,
    ) -> float:
        """
        Calculate the distance between two documents using a
        weighted combination of embeddings and metadata.

        Parameters:
        - doc1 (Document): The first document.
        - doc2 (Document): The second document.

        Returns:
        - float: The calculated distance between the two documents. Higher is better.
        """

        distance_vector = calculate_distance_vector(
            # doc1, doc2, self.embed_model
            doc1,
            doc2,
        )

        # Simple linear combination of the metrics
        sim_score = sum(distance_vector) / len(distance_vector)

        return sim_score
