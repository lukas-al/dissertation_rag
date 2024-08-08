from llama_index.core.schema import Document
from .abstract_retriever import AbstractRetriever
from sklearn.decomposition import PCA
import warnings

from ..processing.distance_metrics import calculate_distance_vector

class V3Retriever(AbstractRetriever):
    """
    A retriever class that calculates the distance between two documents using a weighted combination
    of embeddings and metadata across a range of distance metrics.

    Args:
        AbstractRetriever (type): The abstract retriever class.

    Returns:
        type: The calculated distance between the two documents.
    """

    # @Override
    def __init__(self, name: str = "default", version: str = "0", embed_model=None):
        super().__init__(name, version, embed_model)
        # warnings.warn(
        #     "This class will return an adjacency dict with vectors, not a similarity score. Apply .pca_vector_dict to get the sim scores."
        # )

    # @Override
    def calculate_distance(
        self,
        doc1: Document,
        doc2: Document,
    ) -> float:
        """
        Calculate the distance between two documents using a
        weighted combination of embeddings and metadata across a range of distance metrics.

        Parameters:
        - doc1 (Document): The first document.
        - doc2 (Document): The second document.

        Returns:
        - float: The calculated distance between the two documents.
        """

        distance_vector = calculate_distance_vector(
            doc1,
            doc2,
        )

        # A bit kludgey - similar to the cloudstrike OOB error on the channel file :)
        sim_score = (
            distance_vector[0] * 0.05 + # Name distance
            distance_vector[1] * 0.3 + # Text distance
            distance_vector[2] * 0.1 + # Description distance
            distance_vector[3] * 0.05 + # Type distance
            distance_vector[4] * 0.1 + # Author distance
            distance_vector[5] * 0.1 + # Topic distance
            distance_vector[6] * 0.05 + # Brand distance
            distance_vector[7] * 0.05 + # Division distance
            distance_vector[8] * 0.05 + # MPC round distance
            distance_vector[9] * 0.05 + # Forecast round distance
            distance_vector[10] * 0.1   # Date distance
        ) / len(distance_vector)

        return sim_score