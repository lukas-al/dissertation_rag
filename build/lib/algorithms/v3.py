"""
V3 of the retrieval algorithm - using a PCA to reduce the dimensionality of the metadata vector
"""

from llama_index.core.schema import Document
from .abstract_retriever import AbstractRetriever
from sklearn.decomposition import PCA
import warnings

from ..processing.distance_metrics import calculate_distance_vector


class V3Retriever(AbstractRetriever):
    """
    V1 of the retrieval algorithm, scoring a linear average of the embeddings and metadata distances.
    The aggregation calculation could definitely be made more sophisticated...
    """

    # @Override
    def __init__(self, name: str = "default", version: str = "0", embed_model=None):
        super().__init__(name, version, embed_model)
        warnings.warn(
            "This class will return an adjacency dict with vectors, not a similarity score. Apply .pca_vector_dict to get the sim scores."
        )

    # def retrieve_top_k_doc(
    #     self,
    #     doc1: Document,
    #     embedded_index: List[Document],
    #     k: int = 5,
    #     fuzzy_thresh: int = 80,
    # ) -> List[Tuple[str, float]]:
    #     """Retrieve the top k documents from the embedded index based on their similarity scores.

    #     Args:
    #         doc1 (Document): The document for which to retrieve similar documents.
    #         embedded_index (List[Document]): The list of documents in the embedded index.
    #         k (int, optional): The number of top documents to retrieve. Defaults to 5.
    #         fuzzy_thresh (int, optional): The fuzzy threshold for similarity comparison. Defaults to 80.

    #     Returns:
    #         List[Tuple[str, float]]: A list of tuples containing the document ID and its similarity score.
    #     """

    #     sim_scores = []
    #     for doc in embedded_index:
    #         print("Calcing the score for: ", doc.id_)
    #         sim_scores.append((doc.id_, self.calculate_distance(doc1, doc)))

    #     sim_scores.sort(key=lambda x: x[1], reverse=True)

    #     return sim_scores[:k]

    # # @Override
    # def calculate_distance(
    #     self,
    #     doc1: Document,
    #     doc2: Document,
    # ) -> float:
    #     """
    #     Calculates the distance between two documents using the PCA result.

    #     Args:
    #         doc1 (Document): The first document.
    #         doc2 (Document): The second document.

    #     Returns:
    #         float: The distance between the two documents.
    #     Raises:
    #         AttributeError: If PCA has not been calculated yet. Please run calculate_pca first.
    #     """
    #     try:
    #         self.adj_dict
    #     except AttributeError:
    #         raise AttributeError(
    #             "Adj dict not calculated yet."
    #         )

    #     if not self.is_pca_applied:
    #         # do pca
    #         pass

    #     return self.adj_dict[doc1.id_][doc2.id_]["weight"]

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
        - float: The calculated distance between the two documents.
        """

        distance_vector = calculate_distance_vector(
            # doc1, doc2, self.embed_model
            doc1,
            doc2,
        )

        return distance_vector

    def pca_vector_dict(self, vector_dict):
        """Take the dict with vectors of weights calcd elsewhere and apply a pca weighting to them.

        Args:
            adj_dict (_type_): _description_
        """
        # Fit the PCA on the data
        distance_vectors = []

        for key, nested_dict in vector_dict.items():
            for key2, weight_dict in nested_dict.items():
                distance_vectors.append(weight_dict["weight"])

        # Fit the pca
        pca_mod = PCA(n_components=3).fit(distance_vectors)

        # Apply the pca to each item
        pca_adj_dict = vector_dict.copy()

        for doc1, nested_dict in vector_dict.items():
            for doc2, weight_dict in nested_dict.items():
                pca_adj_dict[doc1][doc2]["weight"] = pca_mod.transform(
                    [weight_dict["weight"]]
                ).mean(axis=1)[0]

        return pca_adj_dict

    # def calculate_adj_dict(
    #     self,
    #     embed_index: List[Document],
    # ) -> float:
    #     """
    #     Calculates the Principal Component Analysis (PCA) for the given embed_index,
    #     Applying it over the calculate distance vector function.

    #     Args:
    #         embed_index (List[Document]): A list of Document objects representing the embeddings.

    #     Returns:
    #         float: The result of the PCA calculation.

    #     Raises:
    #         AttributeError: If self.distance_vectors is not defined.

    #     """
    #     try:
    #         self.distance_vectors

    #     except AttributeError:
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             futures = {}
    #             for doc0 in tqdm.tqdm(embed_index, desc="Calculating distance vectors"):
    #                 for doc1 in embed_index:
    #                     future = executor.submit(calculate_distance_vector, doc0, doc1)
    #                     futures[future] = (doc0.id_, doc1.id_)
    #                     # futures.append((doc0.id_, doc1.id_, executor.submit(calculate_distance_vector, doc0, doc1)))

    #             results = []
    #             for future in concurrent.futures.as_completed(futures):
    #                 doc0_id, doc1_id = futures[future]
    #                 results.append((doc0_id, doc1_id, future.result()))

    #         self.distance_vectors = results

    #     # Calculate the PCA
    #     pca_res = (
    #         PCA(n_components=3).fit_transform([x[2] for x in self.distance_vectors]).mean(axis=1)
    #     )

    #     scaled_pca = RobustScaler().fit_transform(
    #         pca_res.reshape(-1, 1)
    #     )

    #     # Put the results of the PCA into an adj_dict_format so our 'calculate_distance' function can use it
    #     adj_dict = defaultdict(dict)

    #     for res, val in zip(self.distance_vectors, scaled_pca):
    #         adj_dict[res[0]][res[1]] = {
    #             "weight": val
    #         }

    #     # Store the adjacency dict
    #     self.adj_dict = adj_dict

    #     return self.adj_dict

    # def return_sim_score(self, doc0, doc1):
    #     """
    #     Returns the similarity score between two documents.

    #     Parameters:
    #         doc0 (Document): The first document.
    #         doc1 (Document): The second document.

    #     Returns:
    #         float: The similarity score between the two documents.
    #     """
    #     return self.adj_dict[doc0.id_][doc1.id_]["weight"]
