"""
_summary_
"""

from .abstract_retriever import AbstractRetriever
from typing import List, Tuple
from llama_index.core.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from numpy import array


class V0Retriever(AbstractRetriever):
    """
    V0 of the retrieval algorithm, using a simple cosine similarity between embeddings
    of the document's text.
    """

    # @Override
    def retrieve_top_k_doc(
        self,
        doc1: Document,
        embedded_index: List[Document],
        k: int = 5,
        fuzzy_thresh: int = 80,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the top k most similar documents to a given document.

        Args:
            doc1 (Document): The document for which to find similar documents.
            embedded_index (List[Document]): The list of documents to compare against.
            k (int, optional): The number of top documents to retrieve. Defaults to 5.
            fuzzy_thresh (int, optional): The fuzzy threshold for document similarity. Defaults to 80.

        Returns:
            List[Tuple[str | float]]: A list of tuples containing the document ID and similarity score.
        """
        sim_scores = []
        for doc in embedded_index:
            sim_scores.append((doc.id_, self.calculate_distance(doc1, doc)))

        sim_scores.sort(key=lambda x: x[1], reverse=True)

        return sim_scores[:k]

    # @Override
    def calculate_distance(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate the distance between two documents using cosine similarity.

        Parameters:
        doc1 (Document): The first document.
        doc2 (Document): The second document.

        Returns:
        float: The similarity score between the two documents.
        """
        sim_score = cosine_similarity(
            array(doc1.embedding).reshape(1, -1), array(doc2.embedding).reshape(1, -1)
        )

        return sim_score
