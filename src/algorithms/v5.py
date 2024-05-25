"""
Structural Metadata (from the DB)
Implicit Information (extracted from the doc) (Stanza?)
    - Named entity recognition (capital letters, acronyms?)
    - Numbers (matching)
    - Bag-of-words
    - Links
Embedding
Engineered Embeddings
Combinations

----------------

v5 employs regex to do named entity recognition, number matching, and link extraction.
"""

from llama_index.core.schema import Document
from .abstract_retriever import AbstractRetriever
from typing import List, Tuple
from sklearn.preprocessing import RobustScaler

class V5Retriever(AbstractRetriever):
    """
    V5 of the retrieval algorithm, doing named entity recognition, number matching, and link extraction.
    """

    # @Override
    def __init__(
        self, 
        name: str = "V5Retriever", 
        version: str = "0.1", 
        embedded_index=None,
        spacy_model=None
    ):
        super().__init__(name, version)
        self.spacy_model = spacy_model
        
        if self.spacy_model is None:
            raise ValueError(
                "The spacy nlp model is not set. This algorithm will not work without one."
            )
            
        # print(f"Initialised {self.name} v{self.version}")
    
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

    def calculate_distance(self, doc1: Document, doc2: Document) -> float:
        
        # Extract the named entities
        doc1_named_entities = self.extract_named_entities(doc1)
        doc2_named_entities = self.extract_named_entities(doc2)

        # Extract the links
        doc1_links = doc1.metadata['links']
        doc2_links = doc2.metadata['links']
        
        ner_score = self.calculate_ner_distance(doc1_named_entities, doc2_named_entities)
        link_score = self.calculate_link_similarity(doc1_links, doc2_links)
        
        return (ner_score + link_score) / 2
    

    def extract_named_entities(self, doc: Document) -> List[str]:
        """Extract named entities from the document.

        Args:
            doc (Document): The document object containing the text.

        Returns:
            List[str]: A list of named entities extracted from the document.
        """
        
        # Ensure the document text is a string
        if not isinstance(doc.text, str):
            raise ValueError("Document text must be a string.")

        doc_text = doc.text
        doc_text = doc_text.replace("\n", " ")
        
        # Use the Spacy model to process the text
        proc_doc = self.spacy_model(doc_text)

        # Extract the named entities
        named_entities = [ent.text for ent in proc_doc.ents]

        return named_entities
    
    
    def calculate_ner_distance(self, named_entities1: List[str], named_entities2: List[str]) -> float:
        """Calculate the similarity score between two lists of named entities.

        Args:
            named_entities1 (List[str]): The list of named entities from the first document.
            named_entities2 (List[str]): The list of named entities from the second document.

        Returns:
            float: The similarity score between the two lists of named entities.
        """

        return self.jaccard_sim(set(named_entities1), set(named_entities2))
    
    
    def calculate_link_similarity(self, links1: List[str], links2: List[str]) -> float:
        """Calculate the similarity score between two lists of links.

        Args:
            links1 (List[str]): The list of links from the first document.
            links2 (List[str]): The list of links from the second document.

        Returns:
            float: The similarity score between the two lists of links.
        """

        return self.jaccard_sim(set(links1), set(links2))
    
    
    def jaccard_sim(self, set1: set, set2: set) -> float:
        """Calculate the Jaccard similarity between two sets.

        Args:
            set1 (set): The first set.
            set2 (set): The second set.

        Returns:
            float: The Jaccard similarity score.
        """

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    
    def scale_adj_matrix(self, vector_dict):
        """Take the dict with vectors of weights calcd elsewhere and apply a pca weighting to them.

        Args:
            adj_dict (_type_): _description_
        """
        # Fit the PCA on the data
        distance_vectors = []

        for key, nested_dict in vector_dict.items():
            for key2, weight_dict in nested_dict.items():
                distance_vectors.append(weight_dict["weight"])

        # Fit the scaler
        scaler = RobustScaler().fit(distance_vectors)

        # Apply the pca to each item
        scaled_vector_dict = vector_dict.copy()

        for doc1, nested_dict in vector_dict.items():
            for doc2, weight_dict in nested_dict.items():
                scaled_vector_dict[doc1][doc2]["weight"] = scaler.transform([weight_dict["weight"]])[0]

                
        return scaled_vector_dict