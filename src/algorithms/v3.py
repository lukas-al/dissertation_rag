"""
V3 of the retrieval algorithm - using a PCA to reduce the dimensionality of the metadata vector
"""

from llama_index.core.schema import Document
from .abstract_retriever import AbstractRetriever
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import tqdm

from ..processing.distance_metrics import calculate_distance_vector


class V3Retriever(AbstractRetriever):
    """
    V1 of the retrieval algorithm, scoring a linear average of the embeddings and metadata distances.
    The aggregation calculation could definitely be made more sophisticated...
    """
     
    # @Override
    def retrieve_top_k_doc(
        self,
        doc1: Document,
        embedded_index: List[Document],
        k: int = 5,
        fuzzy_thresh: int = 80
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
            sim_scores.append(
                (doc.id_, self.calculate_distance(doc1, doc))
            )
        
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        
        return sim_scores[:k]
    
    
    # @Override
    def calculate_distance(
        self,
        doc1: Document,
        doc2: Document,
    ) -> float:
        """
        Calculates the distance between two documents using the PCA result.

        Args:
            doc1 (Document): The first document.
            doc2 (Document): The second document.

        Returns:
            float: The distance between the two documents.
        Raises:
            AttributeError: If PCA has not been calculated yet. Please run calculate_pca first.
        """
        try:
            self.pca_result
        except AttributeError:
            raise AttributeError("PCA not calculated yet. Please run calculate_pca first.")
            
        return self.return_sim_score(doc1, doc2)
    

    def construct_adjacency_dict(self, embed_index):
        """
        Constructs and returns an adjacency dictionary based on the given embed_index.

        Parameters:
        embed_index (int): The index of the embedding.

        Returns:
        dict: The adjacency dictionary representing the graph.
        """
        self.calculate_pca(embed_index)
        return self.adj_dict

    
    def calculate_pca(
        self,
        embed_index: List[Document],
    ) -> float:
        """
        Calculates the Principal Component Analysis (PCA) for the given embed_index,
        Applying it over the calculate distance vector function.

        Args:
            embed_index (List[Document]): A list of Document objects representing the embeddings.

        Returns:
            float: The result of the PCA calculation.

        Raises:
            AttributeError: If self.distance_vectors is not defined.

        """
        try:
            self.distance_vectors
        except AttributeError:
            distance_vectors = []
            for doc0 in tqdm.tqdm(embed_index, desc='Calculating distance vectors'):
                for doc1 in embed_index:
                    distance_vectors.append(
                        calculate_distance_vector(doc0, doc1)
                    )
            self.distance_vectors = distance_vectors
        
        # Calculate the PCA
        self.pca_result = PCA(n_components=3).fit_transform(self.distance_vectors).mean(axis=1)
        
        self.scaled_pca = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.pca_result.reshape(-1, 1))
        
        # Put the results of the PCA into an adj_dict_format so our 'calculate_distance' function can use it
        adj_dict = {}
        
        # Need to enum the ordered list of results to fit it into the adj_dict format
        for i, doc0 in enumerate(embed_index):
            adj_dict[doc0.id_] = {}
            for j, doc1 in enumerate(embed_index):
                adj_dict[doc0.id_][doc1.id_] = {
                    'weight': self.scaled_pca[i*len(embed_index) + j][0]
                }
        
        # Store the adjacency dict
        self.adj_dict = adj_dict
    
    
    def return_sim_score(self, doc0, doc1):
        """
        Returns the similarity score between two documents.

        Parameters:
            doc0 (Document): The first document.
            doc1 (Document): The second document.

        Returns:
            float: The similarity score between the two documents.
        """
        return self.adj_dict[doc0.id_][doc1.id_]['weight']
