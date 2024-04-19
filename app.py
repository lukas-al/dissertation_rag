# Main entry point to application

from src.etl_module import etl_functions, embedding_functions
from src.processing_module import v0_retrieval
from src.evaluation_module import output_functions, graph_scoring_functions


def main():
    """
    Executes the main process of the application.

    This function loads the document index, embeds the index, retrieves the top k documents
    based on a dummy query, formats the results, saves the results, and evaluates the process.

    !@TODO Currently has some hard-coded dependencies
    """
    
    # Initially assume 
    # Load the document index
    document_index = etl_functions.load_documents()

    # Embed the document index
    embedded_index = embedding_functions.embed_index(document_index)

    # Present a dummy query
    query = "What is inflation currently?"

    # Retrieve the top k documents
    top_k_results = v0_retrieval.retrieve_top_k_query(query, embedded_index, k=2)

    # Format the results
    results = output_functions.format_output(
        top_k_results,
        query,
        "v0_retrieval",
        document_index,
    )

    #! @TODO: !!!
    # Evaluate the results of the process
    # evaluation = evaluation_functions.evaluate_results(results)
    
    # Print the results and evaluation
    # evaluation_functions.present_results(results, evaluation)
    graph = graph_scoring_functions.construct_graph(
        embedded_index,
        v0_retrieval.retrieve_top_k,
        edge_threshold=0.5,
        k=5
    )
    graph_evaluation = graph_scoring_functions.evaluate_graph(graph)
    
    # Save the results
    output_functions.save_results(
        results,
        # evaluation,
    )


if __name__ == "__main__":
    main()
