"""
This module contains the functions used to extract, transform, and load the documents and data
for the RAG project.
"""

import pandas as pd
from typing import List, Tuple, Dict
import re
from fuzzywuzzy import fuzz
import pymupdf

from llama_index.core.schema import BaseNode
from llama_index.core import SimpleDirectoryReader

# from llama_index.core.node_parser import TokenTextSplitter
from StructuredRag.utils.custom_TokenTextSplitter import TokenTextSplitter
from pathlib import Path


def load_document_metadata() -> pd.DataFrame:
    """
    Loads the document metadata from an Excel file and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The document metadata as a pandas DataFrame.
    """
    # Get the path of the current file
    current_path = Path(__file__)

    # Construct the path to the documents directory
    path_to_file = (
        current_path.parent.parent.parent.parent / "config" / "data_organisation.xlsx"
    )

    doc_metadata_df = pd.read_excel(path_to_file, index_col=0)

    return doc_metadata_df


def match_name(
    name: str, metadata_df: pd.DataFrame, min_score: float = 0.0
) -> Tuple[str, float]:
    """
    Matches a given name with names in the metadata DataFrame using a range of string similarity metrics.

    Args:
        name (str): The name to be matched.
        metadata_df (pd.DataFrame): The DataFrame containing the metadata with names (in the index) to be compared against.
        min_score (float, optional): The minimum score threshold for a match to be considered. Defaults to 0.

    Returns:
        Tuple[str, float]: A tuple containing the matched name and the corresponding similarity score.
    """
    # Initialize variables to store the maximum score and corresponding name
    max_score = -1
    max_name = ""

    # Preprocess the input name by removing non-alphanumeric characters
    processed_name = re.sub(r"[^a-zA-Z0-9]", "", name)

    # Iterate over the names in the metadata DataFrame (n)
    for n in metadata_df.index.tolist():
        processed_n = re.sub(r"[^a-zA-Z0-9]", "", n)

        # Calculate different string similarity metrics
        ratio_score = fuzz.ratio(processed_name, processed_n)
        partial_ratio_score = fuzz.partial_ratio(processed_name, processed_n)
        token_sort_ratio_score = fuzz.token_sort_ratio(processed_name, processed_n)
        token_set_ratio_score = fuzz.token_set_ratio(processed_name, processed_n)

        # Calculate weighted average of the scores
        weighted_score = (
            0.4 * ratio_score
            + 0.3 * partial_ratio_score
            + 0.2 * token_sort_ratio_score
            + 0.1 * token_set_ratio_score
        ) / 100

        # Check if the weighted score is above the minimum threshold
        if weighted_score > min_score:
            # Update the maximum score and corresponding name if the current score is higher
            if weighted_score > max_score:
                max_name = n
                max_score = weighted_score
            # If the scores are equal, choose the name with the shorter length
            elif weighted_score == max_score and len(n) < len(max_name):
                max_name = n
                max_score = weighted_score

    # Return the matched name and the corresponding similarity score
    return (max_name, max_score)


def match_notes_metadata(file_path: str, metadata_df: pd.DataFrame) -> Dict[str, str]:
    """Match the metadata using the file name and the manual extracts
    I pasted into the 'data organisation' spreadsheet.

    Args:
        file_path (str): absolute file path to the pdf to match
        metadata_df (pd.DataFrame): dataframe of the data organisation spreadsheet

    Returns:
        dict: dictionary containing the matched metadata
    """
    file_name = file_path.split("\\")[-1]
    idx_nm, _ = match_name(file_name, metadata_df)
    matched_metadata = metadata_df.loc[idx_nm].to_dict()

    return matched_metadata


# def get_random_metadata(file_path: str) -> Dict[str, Union[int, Dict[str, int]]]:
#     """Dummy function to demonstrate how we could extract extra
#     metadata from the text. We're going to need to make this
#     much more sophisticated.

#     Args:
#         file_path (str): absolute file path to the pdf

#     Returns:
#         dict: collection of random metadata
#     """
#     random_metadata = {}
#     # Read the document and extract metadata as a dictionary
#     with open(file_path, "rb") as file:
#         pdf_reader = PyPDF2.PdfReader(file)

#         # Get the number of characters in the pdf
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         random_metadata["num_characters"] = len(text)

#         # Get the number of words in the pdf
#         words = text.split()
#         random_metadata["num_words"] = len(words)

#         # Get the most common 5 words in the pdf
#         word_counts = Counter(words)
#         random_metadata["most_common_words"] = dict(word_counts.most_common(5))

#     return random_metadata


def extract_links_from_pdf(file_path) -> List[str]:
    """
    Extracts all the links from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of extracted links from the PDF file.
    """
    # Open the PDF file
    doc = pymupdf.open(file_path)

    links = []

    # Iterate over PDF pages
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract links
        links_on_page = page.get_links()

        # Iterate over links and append to list
        for link in links_on_page:
            if "uri" in link.keys():
                links.append(link["uri"])

    return links


def get_metadata(file_path: str) -> Dict[str, str]:
    """Gather all the metadata into one spot

    Args:
        file_path (str): absolute file path to pdf

    Returns:
        dict: collection of our metadata
    """
    # Load the document metadata which I manually extracted from the Notes Portal
    metadata_df = load_document_metadata()

    # Match and extract some other metadata
    matched_metadata = match_notes_metadata(file_path, metadata_df)
    # random_metadata = get_random_metadata(file_path)
    link_list = extract_links_from_pdf(file_path)

    # Combine into a single dictionary
    total_metadata = {
        **matched_metadata,
        # **random_metadata,
        "links": link_list,
    }

    # Convert all metadata to strings for consistency
    total_metadata = {k: str(v) for k, v in total_metadata.items()}

    return total_metadata


def load_documents(num_files_limit=None, chunk_size: int = 256) -> List[BaseNode]:
    """Wrapper function to read in the documents and metadata,
    and return the output index list object.

    Returns:
        List[Nodes]: Our index of documents, as a list.
    """
    # Get the path of the current file
    current_path = Path(__file__)

    # Construct the path to the documents directory
    path_to_docs = current_path.parent.parent.parent.parent / "data/01_raw"

    doc_list = SimpleDirectoryReader(
        path_to_docs, file_metadata=get_metadata, num_files_limit=num_files_limit
    ).load_data()

    
    # Split the documents into appropriately sized nodes
    node_list = TokenTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=25,
        backup_separators=["\n"],
    ).get_nodes_from_documents(doc_list)

    return node_list


def doc_node_limiter(doc_list, chunk_size: int = 1024):
    text_splitter = TokenTextSplitter(
        seperator=" ",
        chunk_size=chunk_size,
        chunk_overlap="20",
        backup_separators=["\n"],
    )

    for doc in doc_list:
        nodes = text_splitter(doc.text)

        if len(nodes) > 1:
            pass

    # node_parse = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

    raise NotImplementedError
