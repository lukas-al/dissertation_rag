"""
Implement BM25 for text similarity as a more sophisticated bag-of-words algorithm
"""

import numpy as np
import copy
from fastbm25 import fastbm25
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from llama_index.core.schema import Document
from .abstract_retriever import AbstractRetriever
from typing import List, Tuple

import warnings

class V4Retriever(AbstractRetriever):
    """
    V4 of the retrieval algorithm, scoring a BM25 similarity between the documents.
    """

    # @Override
    def __init__(
        self, name: str = "V4Retriever", version: str = "0.1", embedded_index=None
    ):
        super().__init__(name, version)

        if embedded_index is None:
            warnings.warn("The embedded index is not set. This algorithm will not work without one.")
            # raise ValueError()
        else:
            # Initialise the model
            tokenised_corpus = []
            for doc in embedded_index:
                tokenised_text = doc.text.split()
                tokenised_corpus.append(tokenised_text)

            model = fastbm25(tokenised_corpus)
            self.model = model

        print(f"Initialised {self.name} v{self.version}")
        
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
        Wrapper function to calc the distance between two documents.

        Args:
            doc1 (Document): The first document.
            doc2 (Document): The second document.

        Returns:
            float: The distance between the two documents.
        """
        return self.calc_sim_score(doc1, doc2)

    # def calc_sim_score(self, doc1, doc2):
    #     """
    #     Calculates the similarity score between two documents using the BM25 algorithm.

    #     Args:
    #         doc1 (Document): The first document.
    #         doc2 (Document): The second document.

    #     Returns:
    #         float: The similarity score between the two documents.
    #     """
    #     assert self.embedded_index is not None, "The embedded index is not set"

    #     embedded_index = self.embedded_index

    #     tokenised_corpus = []
    #     for doc in embedded_index:
    #         tokenised_text = doc.text.split()
    #         tokenised_corpus.append(tokenised_text)

    #     model = fastbm25(tokenised_corpus)

    #     tokenised_doc1 = doc1.text.split()
    #     tokenised_doc2 = doc2.text.split()

    #     return model.similarity_bm25(tokenised_doc1, tokenised_doc2)

    def calc_sim_score(self, doc1, doc2):
        """
        Calculates the similarity score between two documents using the BM25 algorithm.

        Args:
            doc1 (Document): The first document.
            doc2 (Document): The second document.

        Returns:
            float: The similarity score between the two documents.
        """

        tokenised_doc1 = doc1.text.split()
        tokenised_doc2 = doc2.text.split()

        return self.model.similarity_bm25(tokenised_doc1, tokenised_doc2)
        
        
        
    def normalise_adj_dict(self, glbl_adj_dict):
        """
        Normalise the adjacency dictionary.

        Args:
            adj_dict (dict): The adjacency dictionary.

        Returns:
            dict: The normalised adjacency dictionary.
        """
        adj_dict = copy.deepcopy(glbl_adj_dict)
        
        total_weights = []
        for doc in adj_dict.keys():
            for doc2 in adj_dict[doc].keys():
                total_weights.append(adj_dict[doc][doc2]["weight"])

        # Fit a minmax scaler
        scaler = RobustScaler()
        scaler.fit(np.array(total_weights).reshape(-1, 1))
        
        # Scale the weights
        for doc in adj_dict.keys():
            for doc2 in adj_dict[doc].keys():
                adj_dict[doc][doc2]["weight"] = scaler.transform(
                    np.array(adj_dict[doc][doc2]["weight"]).reshape(-1, 1)
                )[0][0]

        return adj_dict
