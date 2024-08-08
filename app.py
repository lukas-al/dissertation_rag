# Main entry point to application
import pickle

from StructuredRag.etl import embedding_funcs, etl_funcs
from StructuredRag.processing import graph_construction, param_tuning
from StructuredRag.algorithms import v0, v1, v3, v4, v5
from StructuredRag.utils import persist_results

from datetime import datetime

#! TODO: 
#! Testing
#! 1. Add some needle in haystack queries

def main():
    """
    Executes the main process of the application.
    """
    print("Loading Documents")
    curr_date = datetime.now().strftime("%Y-%m-%d")
    # load and embed the documents
    document_index = etl_funcs.load_documents(chunk_size=256)
    print("Embedding documents")
    embedded_index = embedding_funcs.embed_index(document_index)

    # # Load the embedded index from the following path
    # with open(r"results\v0\2024-07-19\embedded_index.pickle", "rb") as f:
    #     embedded_index = pickle.load(f)
    
    """
    # -------------------------------- # v0 algorithm # -------------------------------- #
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index, v0.V0Retriever
    )

    # Calculate the optimal edge_thresh
    edge_thresh = param_tuning.tune_edgethresh(adj_matrix, embedded_index)

    # persist the results
    persist_results.save_results(
        experiment_type="v0",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            "edge_thresh": edge_thresh,
            'notes': "using bge-large-en-v1.5 model for embedding",
            # "algorithm": v0.V0Retriever()
        },
    )
    

    # -------------------------------- # v1 algorithm # -------------------------------- #
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index, v1.V1Retriever
    )

    # Calculate the optimal edge_thresh
    edge_thresh = param_tuning.tune_edgethresh(adj_matrix, embedded_index)

    # persist the results
    persist_results.save_results(
        experiment_type="v1",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            "edge_thresh": edge_thresh,
            # 'notes': "",
            # "algorithm": v1.V1Retriever()
        },
    )
    
    """
    # -------------------------------- # v3 algorithm # -------------------------------- #
    adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index, v3.V3Retriever
    )

    # pca_adj_mat = v3.V3Retriever().pca_vector_dict(adj_vectors)

    # Calculate the optimal edge_thresh
    edge_thresh = param_tuning.tune_edgethresh(adj_matrix, embedded_index)

    # persist the results
    persist_results.save_results(
        experiment_type="v3",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": adj_matrix,
            # "adj_vectors": adj_vectors,
            "edge_thresh": edge_thresh,
            # 'notes': "",
            # "algorithm": v3.V3Retriever()
        },
    )
    """
    # -------------------------------- # v4 algorithm # -------------------------------- #
    unscaled_adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index, v4.V4Retriever, algo_type="v4"
    )

    scaled_adj_matrix = v4.V4Retriever().normalise_adj_dict(unscaled_adj_matrix)

    # Calculate the optimal edge_thresh
    edge_thresh = param_tuning.tune_edgethresh(scaled_adj_matrix, embedded_index)

    # persist the results
    persist_results.save_results(
        experiment_type="v4",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": scaled_adj_matrix,
            "unscaled_adj_matrix": unscaled_adj_matrix,
            "edge_thresh": edge_thresh,
            # 'notes': """""",
            # "algorithm": v4.V4Retriever()
        },
    )
    """
    # -------------------------------- # v5 algorithm # -------------------------------- #
    unscaled_adj_matrix = graph_construction.construct_adjacency_dict_parallel(
        embedded_index,
        v5.V5Retriever,
        algo_type="v5",
        spacy_model_name="en_core_web_sm",
    )

    v5_instance = v5.V5Retriever(spacy_model="en_core_web_sm")
    scaled_adj_matrix = v5_instance.scale_adj_matrix(unscaled_adj_matrix)

    # Calculate the optimal edge_thresh
    edge_thresh = param_tuning.tune_edgethresh(scaled_adj_matrix, embedded_index)

    # persist the results
    persist_results.save_results(
        experiment_type="v5",
        uuid=curr_date,
        persist_objects={
            "embedded_index": embedded_index,
            "adj_matrix": scaled_adj_matrix,
            "unscaled_adj_matrix": unscaled_adj_matrix,
            "edge_thresh": edge_thresh,
            "notes": "spacy model: en_core_web_sm",
        },
    )


if __name__ == "__main__":
    main()
