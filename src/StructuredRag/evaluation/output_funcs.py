"""
Functions to structure and serve the output of our retrieval algorithms and other results stuff.
"""

from typing import List, Dict, Tuple
from llama_index.core.schema import Document
from datetime import datetime
from pathlib import Path
from StructuredRag.algorithms.inquirer import StructRAGInquirer

import tqdm
import pickle
import yaml
import pandas as pd


def format_output(
    top_k_docs: List[Tuple[str, float]],
    query: str,
    retrieval_algorithm: str,
    document_index: List[Document],
) -> Dict[str, str]:
    """A function to format the results of the retrieval algorithm into a neat dictionary.
    This dictionary will be used for evaluation, presentation, and logging

    To make these results packages manageable, we constrain what we log. For example, the document index only
    stores metadata - making it 6 times smaller (min) than the full document objects.

    Args:
        top_k_docs (List[Document]): The result from the retrieval algorithm
        query (str): The query input into the retrieval algorithm. Can also be
        retrieval_algorithm (str): The name of the retrieval algorithm used
        document_index (List[Document]): The document index used for retrieval

    Returns:
        Dict[str, str]: Package of outputs for evaluation, presentation, and logging
    """

    # Store contextual information about the process
    results_dict = {
        "query": query,
        "retrieval algo": retrieval_algorithm,
        "document index": {
            doc.metadata["file_name"]: doc.metadata for doc in document_index
        },
    }

    # Store the results of the process itself
    retrieval_output = {}

    # Iterate over the top_k_docs object
    for i, doc in enumerate(top_k_docs):
        doc_id, score = doc
        document = [doc for doc in document_index if doc.id_ == doc_id][0]

        # Store the document in an intermediate dictionary
        retrieval_output["result_" + str(i)] = {
            "score": score,
            "document": document,
        }

    # Add the results to the results_dict
    results_dict["retrieval output"] = retrieval_output

    return results_dict


def save_results(
    results_dict: Dict[str, str],
) -> None:
    """A function to save the results of the retrieval algorithm to a pickle file.

    Args:
        results_dict (Dict[str, str]): The dictionary containing the results of the retrieval algorithm
    """
    # Get the path of the current file
    current_path = Path(__file__)

    # Construct the path to the directory
    output_path = current_path.parent.parent.parent / "data/03_output"

    # Get the current date and time
    now = datetime.now()

    # Format the date and time to create a unique identifier (assuming we're operating a <1 a second)
    timestamp_str = now.strftime("%Y%m%d%H%M%S")

    # Use the timestamp in the output file path
    output_path = output_path / f"output_{timestamp_str}.pickle"

    with open(output_path, "wb") as f:
        pickle.dump(results_dict, f)


def run_eval(test_cases, inquirer: StructRAGInquirer) -> List[Dict[str, str]]:
    """
    Run evaluation on a list of test cases using the StructRAGInquirer.

    Args:
        test_cases (dict): A dictionary containing the test cases.
        inquirer (StructRAGInquirer): An instance of the StructRAGInquirer class.

    Returns:
        list: A list of dictionaries containing the evaluation results.
    """
    results = []
    for tc in test_cases["test_cases"]:
        query = tc["question"]

        res = inquirer.run_inquirer(
            query=query,
            source_document_name=tc["anchor_document"],
            k_context=3,
        )

        results.append(res)

    return results


def format_results(
    test_cases, anchor_doc_flag: bool, results: List[Dict[str, str]]
) -> pd.DataFrame:
    """
    Formats the results of the evaluation, into a df so downstream analysis can be completed

    Args:
        test_cases (dict): A dictionary containing the test cases.
        anchor_doc_flag (bool): A flag indicating whether to include the anchor document column.
        results (list): A list of dictionaries containing the evaluation results.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the formatted results.
    """
    df = pd.DataFrame(results)

    # Split the input documents into new columns
    df = pd.concat(
        [df.drop(["input_documents"], axis=1), df["input_documents"].apply(pd.Series)],
        axis=1,
    )
    df.insert(1, "output_score", None)

    for i, col in enumerate(df.columns[2:]):
        # Rename
        df.rename(columns={col: f"input_document_{col}"}, inplace=True)

        # Add a subsequent column named context_{i}_score
        df.insert(i + 3 + int(col), f"context_{col}_score", None)

    # Add the query column
    df["query"] = [x["question"] for x in test_cases["test_cases"]]

    # Add the anchor_document column
    if anchor_doc_flag:
        df["anchor_document"] = [x["anchor_document"] for x in test_cases["test_cases"]]
    else:
        df["anchor_document"] = None

    return df


def create_human_eval_spreadsheet(
    anchor_doc_flag: bool,
    graph_paths: List[str],
    wb_name: str,
) -> List[pd.DataFrame]:
    """
    Creates a human evaluation spreadsheet for the inquirer, to allow a user
    to evaluate the different inquirer's performance, as well as the retrievers.

    Args:
        anchor_doc_flag (bool): Flag indicating whether to use an anchor document.
        graph_paths (List[str]): List of paths to the graphs, persisted from the rest of the code.
        sheet_name (str): Name of the spreadsheet to write to the evaluation folder.

    Returns:
        List[pd.DataFrame]: List of pandas DataFrames containing the evaluation spreadsheets.
    """

    project_root = str(Path(__file__).parent.parent.parent.parent)

    with open(project_root + "\\evaluation" + "\\test_cases.yml", "r") as f:
        test_cases = yaml.safe_load(f)

    anchor_doc_flag = False
    total_results_holder = {}
    for gp in tqdm.tqdm(graph_paths, desc="Running Inquirer for each graph / algo"):
        gp_path = project_root + "\\results" + gp
        inquirer = StructRAGInquirer(
            path_to_experiment=gp_path,
            llm_name="google/flan-t5-large",
            llm_max_tokens=512,
            use_anchor_document=False,
        )

        results = run_eval(test_cases=test_cases, inquirer=inquirer)

        total_results_holder[gp] = results

    df_holder = []
    for gp, results in total_results_holder.items():
        df = format_results(test_cases, anchor_doc_flag, results)

        with pd.ExcelWriter(project_root + "\\evaluation" + f"\\{wb_name}") as writer:
            gp_name = gp.replace("\\", "-")[-1]
            df.to_excel(writer, sheet_name=f"human_eval_{gp_name}")

        df_holder.append(df)

    return df


if __name__ == "__main__":
    create_human_eval_spreadsheet(
        wb_name="human_eval_test_bgemodel.xlsx",
        anchor_doc_flag=False,
        graph_paths=[
            "\\v0\\2024-06-15",
            "\\v1\\2024-06-16",
            "\\v3\\2024-05-28",
            "\\v4\\2024-05-28",
            "\\v5\\2024-05-19",
        ],
    )
