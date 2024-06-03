"""
Contains functions to calculate the distance between two documents.
"""

from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio
from pandas import to_datetime
from fuzzywuzzy import fuzz, process
from sentence_transformers import util
import re

# from sentence_transformers import SentenceTransformer
import numpy as np


def node_name_distance(doc0, doc1) -> float:
    """
    Calculates the distance between the names of two documents using both semantic and syntactic similarity measures.

    Args:
        doc0 (Document): The first document.
        doc1 (Document): The second document.
        embed_model (EmbeddingModel): The embedding model used to calculate semantic similarity.

    Returns:
        float: The distance between the names of the two documents.
    """
    doc0_name = doc0.metadata["file_name"].split("/")[-1].split(".")[0]
    doc1_name = doc1.metadata["file_name"].split("/")[-1].split(".")[0]

    # semantic_similarity:
    doc0_emb = doc0.metadata["embedded_name"]
    doc1_emb = doc1.metadata["embedded_name"]

    # doc0_emb = embed_model.encode(doc0_name)
    # doc1_emb = embed_model.encode(doc1_name)

    # For cosine similarity
    # semantic_sim = cosine_similarity(
    #     np.array(doc0_emb).reshape(1, -1), np.array(doc1_emb).reshape(1, -1)
    # )[0][0]

    semantic_sim = float(util.dot_score(
        doc0_emb, doc1_emb
    ))

    # syntactic_similarity:
    syntactic_sim = ratio(doc0_name, doc1_name)
    syntactic_sim = 2 * (syntactic_sim - 0) / (1 - 0) - 1

    return round(0.3 * syntactic_sim + 0.7 * semantic_sim, 3)  # Random weights


def node_text_distance(doc0, doc1) -> float:
    """Calculate the distance between two nodes based on their text content.

    Args:
        doc1 (Document): the first document
        doc2 (Document): the second document

    Returns:
        float: the distance between the two nodes
    """

    # Cosine sim models
    # sim_score = cosine_similarity(
    #     np.array(doc0.embedding).reshape(1, -1), np.array(doc1.embedding).reshape(1, -1)
    # )[0][0]

    # Dot product models
    sim_score = float(util.dot_score(
        doc0.embedding, doc1.embedding
    ))

    return round(sim_score, 3)


def description_distance_metric(doc0, doc1):
    """Use the cosine distance between the description embeddings to
    calculate the distance between the two documents

    Args:
        doc0 (_type_): _description_
        doc1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # doc0_desc = doc0.metadata["Description"]
    # doc1_desc = doc1.metadata["Description"]

    # doc0_emb = embed_model.encode(doc0_desc)
    # doc1_emb = embed_model.encode(doc1_desc)

    doc0_emb = doc0.metadata["embedded_description"]
    doc1_emb = doc1.metadata["embedded_description"]

    # Cosine sim
    # dist_score = util.dot_product(
    #     np.array(doc0_emb).reshape(1, -1), np.array(doc1_emb).reshape(1, -1)
    # )[0][0]

    # Dot product
    dist_score = float(util.dot_score(
        doc0_emb, doc1_emb
    ))

    return round(dist_score, 3)


def doctype_distance_metric(doc0, doc1):
    """
    Calculates the distance metric between two document types based on a knowledge graph.

    #! @TODO: THE KNOWLEDGE GRAPH NEEDS UPDATING

    Parameters:
    - doc0 (Document): The first document.
    - doc1 (Document): The second document.

    Returns:
    - float: The distance metric between the document types. The value is scaled between -1 and 1.

    Raises:
    - ValueError: If the document types are not found in the knowledge graph.
    - ValueError: If no path is found between the document types in the knowledge graph.
    """

    doc_type1 = doc0.metadata["Type"]
    doc_type2 = doc1.metadata["Type"]

    # ! When testing, there are only actually 3 types of document - background reading, essential reading, and recommended reading.
    # ! This means we don't need this to be so complicated at all.

    if doc_type1 == doc_type2:
        return 1

    else:
        return -1

    # This is hard coded in currently - could use a config file to define it.
    doc_knowledge_graph = {
        "MPR": {"press release": 0.0, "research paper": 0.9, "speech": 0.6},
        "press release": {"research paper": 0.9, "speech": 0.7},
        "research paper": {"speech": 0.7},
        "speech": {},
    }

    # If the document types are the same, return 1
    if doc_type1 == doc_type2:
        return 1

    # Check if the document types exist in the knowledge graph
    if doc_type1 not in doc_knowledge_graph or doc_type2 not in doc_knowledge_graph:
        raise ValueError("Document types not found in the knowledge graph")

    # Get the distance from the knowledge graph
    distance = doc_knowledge_graph[doc_type1].get(
        doc_type2, doc_knowledge_graph[doc_type2].get(doc_type1)
    )

    if distance is None:
        raise ValueError(
            "No path found between the document types in the knowledge graph"
        )

    # Scale the distance to be between -1 and 1
    scaled_distance = round(2 * (distance - 0.5), 3)

    return scaled_distance


def scaled_date_difference(doc0, doc1):
    """
    Scales the difference between two dates and returns a value between -1 and 1.
    The function I've created is a bit odd.

    Parameters:
    date0 (str or datetime): The base date.
    date1 (str or datetime): The date to compare with the base date.

    Returns:
    float: The scaled difference between the two dates.

    """
    # Try to convert the dates to datetime objects
    date0 = to_datetime(doc0.metadata["Date"])
    date1 = to_datetime(doc1.metadata["Date"])

    diff = (date0 - date1).days

    # Intuition - if date 1 > date 0 by a lot then it likely isn't related at all
    if diff < -90:
        return -1

    # Intuition - if date 1 < date 0 by a lot then it likely isn't related at all
    elif diff > 365:
        return -1

    # Intuition - if the difference is within a week, return 1
    elif -15 < diff < 15:
        return 1

    # Scale the diff to be between -1 and 1, with the min and max values being -365 and +365
    else:
        scaled_diff = 2 * (diff + 365) / (365 + 365) - 1
        # Return the scaled diff as the output of a smoothing 1/x function
        return round(1 / (0.5 + scaled_diff**2) - 1, 3)


def author_distance_metric(doc0, doc1, fuzz_thresh=80) -> float:
    """
    Calculates the author distance metric between two documents.

    Parameters:
    - doc0 (Document): The first document.
    - doc1 (Document): The second document.
    - fuzz_thresh (int): The threshold for fuzzy matching similarity. Default is 80.

    Returns:
    - float: The scaled similarity between the authors of the two documents.

    """
    authors0 = doc0.metadata["Authors"]
    authors1 = doc1.metadata["Authors"]

    if isinstance(authors0, str):
        authors0 = [name.strip() for name in authors0.split(",")]

    if isinstance(authors1, str):
        authors1 = [name.strip() for name in authors1.split(",")]

    # Calculate the Jaccard similarity between the two author lists
    jacc_sim = fuzzy_jaccard_similarity(
        set(authors0), set(authors1), threshold=fuzz_thresh
    )

    # Scale the Jaccard similarity to be between -1 and 1
    scaled_similarity = round(2 * jacc_sim - 1, 3)

    return scaled_similarity


def fuzzy_jaccard_similarity(set1, set2, threshold):
    """
    Calculates the fuzzy Jaccard similarity between two sets of strings.
    Used in the author_distance_metric function to help with where
    I or anyone else might have made a typo in the author's name.

    Parameters:
    set1 (str or list): The first set of strings.
    set2 (str or list): The second set of strings.
    threshold (int): The minimum similarity threshold for a match.

    Returns:
    float: The fuzzy Jaccard similarity between the two sets.
    """
    # Deal with the case where either one of the inputs are single item strings
    if isinstance(set1, str) | isinstance(set2, str):
        return fuzz.ratio(set1, set2) / 100

    # Deal with the case where both inputs are empty
    if len(set1) == 0 and len(set2) == 0:
        return 0.0

    # Otherwise calculate the fuzzy Jaccard similarity
    intersection = set()
    for el1 in set1:
        if el1.strip():  # Check if el1 is not an empty string
            _, similarity = process.extractOne(el1, set2)
            if similarity >= threshold:
                intersection.add(el1)

    union = len(set1) + len(set2) - len(intersection)

    # if union != 0:
    #     if len(intersection) / union > 1:
    #         print('---')
    #         print(len(intersection) / union)
    #         print(set1, '||' ,set2, '||', intersection)

    #! Add a dumb check for outliers. Not sure why they're occuring rn and cba to fix it.
    if len(intersection) / union > 1:
        return 1

    return round(
        len(intersection) / union if union != 0 else 0, 3
    )  # Add a check to prevent division by zero


def topic_distance_metric(doc0, doc1, fuzz_thresh=80):
    """
    Calculates the topic distance metric between two documents.

    #! @TODO: There are nested topics and I'm not sure how to ex-ante construct
    #! a knowledge graph for this. I'll just use fuzzy matching for now.

    Parameters:
        doc0 (Document): The first document.
        doc1 (Document): The second document.
        fuzz_thresh (int, optional): The threshold for fuzzy matching. Defaults to 80.

    Returns:
        float: The scaled similarity between the topics of the two documents.
    """

    # Get the topics from the metadata
    # topics0 = doc0.metadata["Topics"].split('>').split(",")
    # topics1 = doc1.metadata["Topics"].split(",")

    topics0 = [topic.strip() for topic in re.split(r"[>|,]", doc0.metadata["Topics"])]
    topics1 = [topic.strip() for topic in re.split(r"[>|,]", doc1.metadata["Topics"])]

    # Calculate the Jaccard similarity between the two topic lists
    jacc_sim = fuzzy_jaccard_similarity(topics0, topics1, threshold=fuzz_thresh)

    # Scale the Jaccard similarity to be between -1 and 1
    scaled_similarity = round(2 * jacc_sim - 1, 3)

    return scaled_similarity


def brand_distance_metric(doc0, doc1, fuzz_thresh=80):
    """
    Calculates the brand distance metric between two documents based on their brand metadata.

    #! @TODO: There are nested topics and I'm not sure how to ex-ante construct
    #! a knowledge graph for this. I'll just use fuzzy matching for now.

    Parameters:
    - doc0: The first document.
    - doc1: The second document.
    - fuzz_thresh: The threshold for fuzzy matching similarity. Default is 80.

    Returns:
    - scaled_similarity: The scaled similarity between the two documents' brand metadata.
    """
    # Get the brands from the metadata
    brands0 = [brand.strip() for brand in re.split(r"[>|,]", doc0.metadata["Brands"])]
    brands1 = [brand.strip() for brand in re.split(r"[>|,]", doc1.metadata["Brands"])]

    for el in brands0:
        if el.lower() == "nan":
            return 0

    for el in brands1:
        if el.lower() == "nan":
            return 0

    # Calculate the Jaccard similarity between the two brand lists
    jacc_sim = fuzzy_jaccard_similarity(brands0, brands1, threshold=fuzz_thresh)

    # Scale the Jaccard similarity to be between -1 and 1
    scaled_similarity = round(2 * jacc_sim - 1, 3)

    return scaled_similarity


def division_distance_metric(doc0, doc1, fuzz_thresh=80):
    """
    Calculates the division distance metric between two documents based on their divisions metadata.

    Parameters:
    - doc0: The first document.
    - doc1: The second document.
    - fuzz_thresh: The threshold for fuzzy matching similarity. Default is 80.

    Returns:
    - The scaled similarity between the divisions of the two documents, ranging from -1 to 1.
    """
    # Get the divisions from the metadata
    divisions0 = [
        division.strip() for division in re.split(r"[>|,]", doc0.metadata["Divisions"])
    ]
    divisions1 = [
        division.strip() for division in re.split(r"[>|,]", doc1.metadata["Divisions"])
    ]

    # Calculate the Jaccard similarity between the two division lists
    jacc_sim = fuzzy_jaccard_similarity(divisions0, divisions1, threshold=fuzz_thresh)

    # Scale the Jaccard similarity to be between -1 and 1
    scaled_similarity = round(2 * jacc_sim - 1, 3)

    # if scaled_similarity > 1:
    #     return 1

    return scaled_similarity


def mpc_round_distance_metric(doc0, doc1) -> int:
    """
    Calculates the distance metric between two documents based on their MPC Round metadata.

    Parameters:
    doc0 (Document): The first document.
    doc1 (Document): The second document.

    Returns:
    int: The distance metric between the two documents.
        Returns 1 if the MPC Round metadata is the same, -1 otherwise.
    """

    # Get the MPC round from the metadata
    mpc_round0 = doc0.metadata["MPC Round"]
    mpc_round1 = doc1.metadata["MPC Round"]

    # If the MPC round is the same, return 1
    if mpc_round0 == mpc_round1:
        return 1

    # If the MPC round is different, return -1
    else:
        return -1


def forecast_round_distance_metric(doc0, doc1) -> float:
    """
    Calculates the distance metric between two documents based on their forecast round.

    Parameters:
    - doc0: The first document.
    - doc1: The second document.

    Returns:
    - int: The distance metric between the two documents.
        Returns 1 if the forecast round is the same, -1 otherwise.
    """
    # Get the forecast round from the metadata
    forecast_round0 = doc0.metadata["Forecast Round"]
    forecast_round1 = doc1.metadata["Forecast Round"]

    # If the forecast round is the same, return 1
    if forecast_round0 == forecast_round1:
        return 1

    # If the forecast round is different, return -1
    else:
        return -1


def calculate_distance_vector(doc0, doc1, fuzz_thresh=80):
    """
    Calculate the distance vector between two documents using a combination of semantic and syntactic similarity measures.

    Args:
        doc0 (Document): The first document.
        doc1 (Document): The second document.
        embed_model (EmbeddingModel): The embedding model used to calculate semantic similarity.
        fuzz_thresh (int): The threshold for fuzzy matching similarity. Default is 80.

    Returns:
        List[float]: The distance vector between the two documents
                    All scores are normalised between -1 and 1.
    """
    name_distance = node_name_distance(doc0, doc1)
    text_distance = node_text_distance(doc0, doc1)
    desc_distance = description_distance_metric(doc0, doc1)
    type_distance = doctype_distance_metric(doc0, doc1)
    author_distance = author_distance_metric(doc0, doc1, fuzz_thresh)
    topic_distance = topic_distance_metric(doc0, doc1, fuzz_thresh)
    brand_distance = brand_distance_metric(doc0, doc1, fuzz_thresh)
    division_distance = division_distance_metric(doc0, doc1, fuzz_thresh)
    mpc_round_distance = mpc_round_distance_metric(doc0, doc1)
    forecast_round_distance = forecast_round_distance_metric(doc0, doc1)
    date_distance = scaled_date_difference(doc0, doc1)

    return [
        name_distance,
        text_distance,
        desc_distance,
        type_distance,
        author_distance,
        topic_distance,
        brand_distance,
        division_distance,
        mpc_round_distance,
        forecast_round_distance,
        date_distance,
    ]
