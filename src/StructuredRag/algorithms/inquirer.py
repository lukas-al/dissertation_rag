import os
import pickle
from datetime import date

import networkx as nx
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from llama_cpp import Llama

from sentence_transformers import util
from StructuredRag.etl import embedding_funcs
from StructuredRag.processing import graph_construction


class StructRAGInquirer:
    """
    Wraps all the logic for the llm inquirer and structured RAG system
    """

    def __init__(
        self,
        path_to_experiment: str,
        llm_name: str = "google/flan-t5-large",
        llm_max_tokens: int = 1024,
        use_anchor_document: bool = True,
        llm_type: str = "huggingface",
        **kwargs,
    ):
        # Read the data for the specified experiment
        data = {}
        for item in os.listdir(path_to_experiment):
            # If item is a pickle
            if item.split(".")[-1] == "pickle":
                print("Loading item:", item.split(".")[0])

                with open(path_to_experiment + "/" + item, "rb") as f:
                    data[item.split(".")[0]] = pickle.load(f)

        # Instantiate the class variables and llm
        self.embedded_index = data["embedded_index"]
        self.edge_thresh = data["edge_thresh"]
        self.adj_matrix = data["adj_matrix"]
        self.llm_type = llm_type

        if llm_type == "huggingface":
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=llm_name,
                task="text2text-generation",
                model_kwargs={
                    "max_length": llm_max_tokens,
                },
            )

        elif llm_type == "llamacpp":
            try:
                model_path = kwargs.pop("model_path", None)
                verbose = kwargs.pop("verbose", False)
                n_gpu_layers = kwargs.pop("n_gpu_layers", -1)

                self.llm = Llama(
                    model_path=model_path,
                    verbose=verbose,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=llm_max_tokens,
                    **kwargs,  # Unpack additional keyword arguments
                )
            except ValueError as e:
                raise ValueError(
                    f"Error loading Llama model: {e} \n Double check to include required arguments"
                )

        self.use_anchor_document = use_anchor_document
        self.graph = graph_construction.construct_graph_from_adj_dict(
            self.adj_matrix, self.edge_thresh, self.embedded_index
        )

    def run_inquirer(
        self,
        query: str,
        source_document_name: str,
        k_context: int = 3,
    ):
        """
        Runs the inquirer system to generate an answer to a query.

        Args:
            query (str): The query to be answered.
            source_document_name (str): The name of the document to use as the anchor.
            k_context (int): The number of similar nodes to consider for the context. Default is 3.

        Returns:
            response (dict): The response from the inquirer system.
        """
        # Embed the query
        embedded_query = embedding_funcs.embed_query(query)

        # Find the most similar chunk of the document to load
        most_similar_node_id, similarity_score = self._doc_similar_nodes(
            embedded_query, source_document_name
        )

        # Search through the graph to find the most similar nodes
        nearest_nodes = self._graph_similar_nodes(most_similar_node_id, k_context)

        if self.llm_type == "huggingface":
            top_matches = self._reshape_documents_for_llm(nearest_nodes)
            extractive_prompt, stuff_document_prompt = self._construct_prompt()

            # Query chain
            query_chain = load_qa_with_sources_chain(
                self.llm,
                chain_type="stuff",
                prompt=extractive_prompt,
                document_prompt=stuff_document_prompt,
            )

            # Response
            response = query_chain.invoke(
                {"input_documents": top_matches, "question": query},
                return_only_outputs=True,
            )
            response["input_documents"] = top_matches

            return response

        elif self.llm_type == "llamacpp":
            context_string = self.build_context_for_QA_gen(
                doc_id=most_similar_node_id,
                k_context=k_context,
            )

            prompt = self.create_chatML_prompt(context_string, query)

            response = self.llm.create_chat_completion(messages=prompt)

            return response

    def create_chatML_prompt(self, context, query):
        """
        Generates a chatML prompt for an AI assistant with a focus on helping to answer economists' search questions over particular documents.

        Parameters:
        - context (str): The context to be used in answering the question.
        - query (str): The question asked by the user.

        Returns:
        - list: A list of dictionaries representing the chatML prompt. Each dictionary has two keys: 'role' and 'content'. The 'role' can be either 'system' or 'user', and the 'content' contains the corresponding message.

        Example:
        prompt = create_chatML_prompt("Context information", "What is the inflation rate?")
        print(prompt)
        """
        return [
            {
                "role": "system",
                "content": """
                    You are an AI assistant with a focus on helping to answer economists' search questions over particular documents. 
                    Respond only to the question asked, the response should be concise and relevant, and use the context provided to give a comprehensive answer.
                    It is important to maintain impartiality and non-partisanship. If you are unable to answer a question based on the given instructions and context, please indicate so.
                    Your responses should be well-structured and professional, using British English.
                """,
            },
            {
                "role": "user",
                "content": f"""
                {query} Use the following context to answer the question:
                Context: {context}
                """,
            },
        ]

    def get_document_name_list(self):
        """
        Returns a list of document names extracted from the embedded index.

        Returns:
            list: A list of document names.
        """
        doc_list = set()
        for doc in self.embedded_index:
            doc_list.add(doc.metadata["file_name"].split("/")[-1])

        return list(doc_list)

    def _doc_similar_nodes(self, embedded_query, source_document):
        """
        Calculate the similarity scores between the embedded query and the documents in the embedded index.

        Args:
            embedded_query (numpy.ndarray): The embedded representation of the query.
            source_document (str): The name of the source document.

        Returns:
            tuple: A tuple containing the ID of the most similar node and its corresponding similarity score.

        Raises:
            ValueError: If no data is returned from similar nodes.
        """
        sim_scores = {}

        if self.use_anchor_document:
            for doc in self.embedded_index:
                if doc.metadata["file_name"].split("/")[-1] == source_document:
                    # sim_scores[doc.id_] = cosine_similarity(embedded_query.reshape(1, -1), np.array(doc.embedding).reshape(1, -1))[0][0]
                    sim_scores[doc.id_] = float(
                        util.dot_score(embedded_query, doc.embedding)
                    )
        else:
            for doc in self.embedded_index:
                # sim_scores[doc.id_] = cosine_similarity(embedded_query.reshape(1, -1), np.array(doc.embedding).reshape(1, -1))[0][0]
                sim_scores[doc.id_] = float(
                    util.dot_score(embedded_query, doc.embedding)
                )

        if len(sim_scores) == 0:
            raise ValueError("No data returned from similar nodes")

        doc_similarity = dict(
            sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
        )

        most_similar_node_id = list(doc_similarity.keys())[0]

        return most_similar_node_id, doc_similarity[most_similar_node_id]

    def _graph_similar_nodes(self, most_similar_node_id, k_context):
        """
        Finds the k_context most similar nodes to the given most_similar_node_id in the graph.

        Parameters:
            most_similar_node_id (int): The ID of the node to find similar nodes for.
            k_context (int): The number of similar nodes to retrieve.

        Returns:
            list: A list of tuples containing the IDs and weights of the k_context most similar nodes.
        """
        node_paths = nx.single_source_dijkstra(
            G=self.graph, source=most_similar_node_id, weight="weight"
        )

        # Get the nodes with the longest path - highest weight means most similar
        nearest_node_ids = list(node_paths[0].items())[-k_context:]

        return nearest_node_ids

    def _reshape_documents_for_llm(self, nearest_node_ids):
        """
        Reshapes the documents for the LLM to consume.

        Args:
            nearest_node_ids (list): A list of nearest node IDs.

        Returns:
            list: A list of Document objects representing the reshaped documents.
        """
        # Extract the info from the nodes
        nearest_docs = []
        for doc in self.embedded_index:
            for node_id in nearest_node_ids:
                if node_id[0] == doc.id_:
                    nearest_docs.append((doc, node_id[1]))

        # Pack into the Document class
        top_matches = [
            Document(
                page_content=doc[0].text,
                metadata={
                    "doc_num": i + 1,
                    "doc_description": doc[0].metadata["Description"],
                    "doc_date": doc[0].metadata["Date"],
                    "doc_difference": doc[1],
                    "title": doc[0].metadata["file_name"].split("/")[-1],
                },
            )
            for i, doc in enumerate(nearest_docs)
        ]

        # Re-order top matches to be lowest to highest by doc_difference
        top_matches = sorted(top_matches, key=lambda x: x.metadata["doc_difference"])

        return top_matches

    def _construct_prompt(self):
        """
        Constructs and returns two prompt templates for generating AI assistant responses.

        The first template, EXTRACTIVE_PROMPT_PYDANTIC, combines a core prompt with an extractive task prompt.
        It includes instructions for the AI assistant to answer questions based on provided contexts, ensuring
        impartiality and non-partisanship, and using British English. The template also includes the current date.

        The second template, STUFF_DOCUMENT_PROMPT, is a simple template for formatting document content.

        Returns:
            tuple: A tuple containing the EXTRACTIVE_PROMPT_PYDANTIC and STUFF_DOCUMENT_PROMPT templates.
        """
        _core_prompt = """
            ==Background==
            You are an AI assistant with a focus on helping to answer economists' search questions over particular documents. 
            Your responses should use the context included to provide contextual information. 
            It is important to maintain impartiality and non-partisanship. If you are unable to answer a question based on the given instructions, please indicate so.
            Your responses should be well-structured and professional, using British English.
        """

        _extractive_prompt = """
            ==TASK==
            Your task is to extract and write an answer for the question based on the provided contexts. 
            If the question cannot be answered from the information in the context, please do not provide an answer.
            If the context is not related to the question, please do not provide an answer.

            Question: {question}
            
            Contexts: {summaries}
        """

        EXTRACTIVE_PROMPT_PYDANTIC = PromptTemplate.from_template(
            template=_core_prompt + _extractive_prompt,
            partial_variables={
                "current_datetime": str(date.today()),
            },
        )

        _stuff_document_template = """
            {page_content}
        """

        STUFF_DOCUMENT_PROMPT = PromptTemplate.from_template(_stuff_document_template)
        return EXTRACTIVE_PROMPT_PYDANTIC, STUFF_DOCUMENT_PROMPT

    def build_context_for_QA_gen(self, doc_id, k_context: int = 3):
        """
        Builds the context string for generating synthetic question-answering pairs.

        Args:
            doc: The document for which the context is being built. Actual doc object
            rag_agent: The RAG agent used for retrieving similar nodes.
            k_context (int): The number of similar nodes to consider for building the context. Default is 3.

        Returns:
            context_string (str): The generated context string containing the clean text of similar nodes.
        """
        similar_nodes = self._graph_similar_nodes(doc_id, k_context)

        context_string = """ """
        for i, (node_id, _) in enumerate(similar_nodes):
            # Yes its unoptimised... find the document who's id matches the node_id
            node = next((x for x in self.embedded_index if x.id_ == node_id), None)

            clean_text = (
                node.text.replace("\n", " ")
                .replace("\t", " ")
                .replace("  ", " ")
                .strip()
            )
            context_string += f"Context {i}: {clean_text} \n"

        return context_string
