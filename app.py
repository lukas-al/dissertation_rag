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
    document_index = etl_funcs.load_documents()
    embedded_index = embedding_funcs.embed_index(document_index)
    
    # v0 algorithm
    adj_matrix = graph_construction.construct_adjacency_dict(
        embedded_index,
        v0.V0Retriever(name="v0_algo", version="0.1")
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
    adj_matrix = graph_construction.construct_adjacency_dict(
        embedded_index,
        v1.V1Retriever(name='v1_algo')
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
    v3_retriever = v3.V3Retriever(name='v3_algo')
    adj_matrix = v3_retriever.construct_adjacency_dict(embed_index=embedded_index)
    
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
    adj_matrix = graph_construction.construct_adjacency_dict(
        embedded_index,
        v4.V4Retriever(name='v4_algo', embedded_index=embedded_index[::3])
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
    
    # v5 algorithm
    adj_matrix = graph_construction.construct_adjacency_dict(
        embedded_index,
        v5.V5Retriever(
            name='v5_algo_spacy_lg', 
            embedded_index=embedded_index,
            spacy_model=spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            )
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
