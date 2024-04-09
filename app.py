# Main entry point to application

from src.etl_module import etl_functions, embedding_functions
from src.processing_module import v0_retrieval
from src.evaluation_module import output_functions


def main():
    """
    Executes the main process of the application.

    This function loads the document index, embeds the index, retrieves the top k documents
    based on a dummy query, formats the results, saves the results, and evaluates the process.

    !@TODO Currently has some hard-coded dependencies
    """
    # Load the document index
    document_index = etl_functions.load_documents()

    # Embed the document index
    embedded_index = embedding_functions.embed_index(document_index)

    # Present a dummy query
    query = "What is inflation currently?"

    # Retrieve the top k documents
    top_k_documents = v0_retrieval.retrieve_top_k(query, embedded_index, k=2)

    # Format the results
    results = output_functions.format_output(
        top_k_documents,
        query,
        "v0_retrieval",
        document_index,
    )

    # Save the results
    output_functions.save_results(
        results,
    )

    # Evaluate the results of the process
    # evaluation_functions.evaluate_results(results)


if __name__ == "__main__":
    main()
