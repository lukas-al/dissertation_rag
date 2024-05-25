# Main entry point to application
import spacy

from src.etl import embedding_funcs, etl_funcs
from src.processing import graph_construction
from src.algorithms import v0, v1, v3, v4, v5
from src.utils import persist_results

from datetime import datetime

def main():
    """
    Executes the main process of the application.
    """
    
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    # load and embed the documents
    document_index = etl_funcs.load_documents(num_files_limit=2)
    embedded_index = embedding_funcs.embed_index(document_index)
    
    # v0 algorithm
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index,
        v0.V0Retriever
    )

    # persist the results
    persist_results.save_results(
        experiment_type="v0",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            'notes': "",
            # "algorithm": v0.V0Retriever()
        }
    )
    
    # v1 algorithm
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index,
        v1.V1Retriever
    )

    # persist the results
    persist_results.save_results(
        experiment_type="v1",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            'notes': "",
            # "algorithm": v1.V1Retriever()
            
        }
    )

    # v3 algorithm
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index,
        v3.V3Retriever
    )
    
    # persist the results
    persist_results.save_results(
        experiment_type="v3",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            'notes': "",
            # "algorithm": v3.V3Retriever()
        }
    )
    
    # v4 algorithm
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index, 
        v4.V4Retriever, 
        algo_type="v4"
    )

    # persist the results
    persist_results.save_results(
        experiment_type="v4",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            'notes': """
            Operation only performed on 1/3 of the documents to speed up.
            Needs significant performance optimisation to be useable.
            """,
            # "algorithm": v4.V4Retriever()
        }
    )
    
    
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index, 
        v5.V5Retriever, 
        algo_type="v5",
        spacy_model_name="en_core_web_lg"
    )
    
    # persist the results
    persist_results.save_results(
        experiment_type="v5",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            'notes': """""",
        }
    )
    
    
if __name__ == "__main__":
    main()
