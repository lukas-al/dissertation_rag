"""
Module containing the embedding functions for the ETL pipeline and other activities.
"""

from typing import List
from llama_index.core.schema import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embed_index(document_index: List[Document]) -> List[Document]:
    """
    Utility function to accept an index constructed by load_documents,
    embed all the documents, and return the embedded index.

    This function is model agnostic and can be used with any embedding model.

    Args:
        document_index (List[Document]): A list of Document objects representing the index to be embedded.

    Returns:
        List[Document]: A list of Document objects with embedded representations included.
    """

    #! Eventually replace the model selection with an env variable specified in .env or config flag
    #! Or perhaps a

    # Load the SentenceTransformer model
    # model = SentenceTransformer("/Users/lukasalemu/Documents/00. Bank of England/00. Degree/Dissertation/structured-rag/models/all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with tqdm(total=len(document_index), desc="Embedding Documents") as pbar:
        for doc in document_index:
            # Embed the document text
            doc.embedding = model.encode(doc.text).tolist()

            # Deal with the doc name
            doc_name = doc.metadata["file_name"].split("/")[-1].split(".")[0]
            doc.metadata["embedded_name"] = model.encode(doc_name)

            # Deal with the doc description
            doc_desc = doc.metadata["Description"]
            doc.metadata["embedded_description"] = model.encode(doc_desc)

            pbar.update(1)

    return document_index


def embed_query(query: str) -> List[float]:
    """
    Utility function to embed a query using the SentenceTransformer model.

    Args:
        query (str): The query string to be embedded.

    Returns:
        List[float]: A list of floats representing the embedded query.
    """

    #! Eventually replace the model selection with an env variable specified in .env or config flag

    # Load the SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed the query
    query_embedding = model.encode(query)

    return query_embedding
