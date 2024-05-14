"""
Defines a class for our generic retrieval algorithm. The template is extended
by all the specific retrieval algorithms.
"""

from typing import List, Tuple
from llama_index.core.schema import Document


class AbstractRetriever:
    def __init__(self, name: str = 'default', version: str = "0", embed_model=None):
        self.name = name
        self.version = version
        self.embed_model = embed_model
        
        print(f"Initialised {self.name} v{self.version}")

    def retrieve_top_k_doc(
        self,
        doc1: Document,
        embedded_index: List[Document],
        k: int = 5,
        fuzzy_thresh: int = 80,
    ) -> List[Tuple[str, float]]:
        """Retrieve the top k most similar documents from the document index based on a query document.

        This function takes a query document `doc1` and a list of embedded documents `embedded_index`.
        It calculates the similarity between `doc1` and each document in `embedded_index` using a specific retrieval algorithm.
        The function returns a list of tuples containing the document ID and similarity score of the top k similar documents.

        Args:
            doc1 (Document): The query document.
            embedded_index (List[Document]): The list of embedded documents.
            k (int, optional): The number of top similar documents to retrieve. Defaults to 5.
            fuzzy_thresh (int, optional): The fuzzy threshold for similarity calculation. Defaults to 80.

        Raises:
            NotImplementedError: This function should be implemented by the specific retrieval algorithm.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing the document ID and similarity score of the top k similar documents.
        """

        raise NotImplementedError

    def calculate_distance(
        doc1: Document,
        doc2: Document,
    ) -> float:
        """Calculate the similarity between two documents.

        This function calculates the similarity between two documents `doc1` and `doc2` using a specific retrieval algorithm.
        The function returns a similarity score between 0 and 1.

        Args:
            doc1 (Document): The first document.
            doc2 (Document): The second document.

        Raises:
            NotImplementedError: This function should be implemented by the specific retrieval algorithm.

        Returns:
            float: The similarity score between the two documents.
        """

        raise NotImplementedError

    def __repr__(self):
        return f"{self.name} v{self.version}"
