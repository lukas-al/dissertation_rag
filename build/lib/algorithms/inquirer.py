
from StructuredRag.etl import embedding_funcs
from StructuredRag.processing import graph_construction

import os
import pickle
import numpy as np
import networkx as nx
from datetime import date
from langchain.prompts.prompt import PromptTemplate
from sentence_transformers import util

from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

class StructRAGInquirer():
    """
    Wraps all the logic for the llm inquirer and structured RAG system
    """
    def __init__(
        self,
        path_to_experiment: str,
        llm_name: str = 'google/flan-t5-large',
        llm_max_tokens: int = 512,
    ):
        # Read the data for the specified experiment
        data = {}
        for item in os.listdir(path_to_experiment):
            print('Loading item:', item.split('.')[0])
            
            with open(path_to_experiment + '/' + item, 'rb') as f:
                data[item.split('.')[0]] = pickle.load(f)
        
        # Instantiate the class variables and llm
        self.embedded_index = data['embedded_index']
        self.edge_thresh = data['edge_thresh']
        self.adj_matrix = data['adj_matrix']
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=llm_name, 
            task="text2text-generation", 
            model_kwargs={
                "max_length": llm_max_tokens,
            },
        )     
        
    def run_inquirer(
        self,
        query: str,
        source_document_name: str,
        k_context: int = 3,
    ):
        """WRAP ALL THE LOGIC FOR THE INQUIRER HERE"""
        # Embed the query
        embedded_query = embedding_funcs.embed_query(query)
        
        # Find the most similar chunk of the document to load
        most_similar_node_id, similarity_score = self._doc_similar_nodes(embedded_query, source_document_name)
        
        # Search through the graph to find the most similar nodes
        nearest_nodes = self._graph_similar_nodes(most_similar_node_id, k_context)
        
        top_matches = self._reshape_documents_for_llm(nearest_nodes)
        
        extractive_prompt, stuff_document_prompt = self._construct_prompt()
        
        # Query chain
        query_chain = load_qa_with_sources_chain(
            self.llm,
            chain_type='stuff',
            prompt=extractive_prompt,
            document_prompt=stuff_document_prompt,
        )
        
        # Response
        response = query_chain.invoke(
            {"input_documents": top_matches, "question": query},
            return_only_outputs=True,
        )
        response['input_documents'] = top_matches
        
        return response
    
                
    def _doc_similar_nodes(
        self, 
        embedded_query, 
        source_document
    ):
        sim_scores = {}
        for doc in self.embedded_index:
            if doc.metadata["file_name"].split("/")[-1] == source_document:
                # Cosine sim
                # sim_scores[doc.id_] = cosine_similarity(embedded_query.reshape(1, -1), np.array(doc.embedding).reshape(1, -1))[0][0]
                # Dot product
                sim_scores[doc.id_] = float(util.dot_score(embedded_query, doc.embedding))

        doc_similarity = dict(sorted(sim_scores.items(), key=lambda x: x[1], reverse=True))

        most_similar_node_id = list(doc_similarity.keys())[0]
        
        return most_similar_node_id, doc_similarity[most_similar_node_id]
    
    
    def _graph_similar_nodes(
        self, 
        most_similar_node_id, 
        k_context
    ):
        graph = graph_construction.construct_graph_from_adj_dict(
           self.adj_matrix,
           self.edge_thresh,
           self.embedded_index
        )
        
        node_paths = nx.single_source_dijkstra(
            G=graph, 
            source=most_similar_node_id, 
            weight='weight'
        )

        nearest_node_ids = list(node_paths[0].items())[:k_context]

        return nearest_node_ids
    
    
    def _reshape_documents_for_llm(self, nearest_node_ids):
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
                    'doc_num': i + 1,
                    'doc_description': doc[0].metadata['Description'],
                    'doc_date': doc[0].metadata['Date'],
                    'doc_difference': doc[1],
                }
            )
            for i, doc in enumerate(nearest_docs)
        ]
        
        return top_matches
    
    
    def _construct_prompt(self):
        # _core_prompt = """
        #     ==Background==
        #     You are an AI assistant with a focus on helping to answer economists' search questions
        #     over particular documents. Your responses should be based primarily
        #     on information provided within the query. It is important to maintain impartiality
        #     and non-partisanship. If you are unable to answer a question based on the given
        #     instructions, please indicate so. Your responses should be concise and professional,
        #     using British English.
        #     Consider the current date, {current_datetime}, when providing responses related to time. 
        # """
        
        _core_prompt = """
            You are an AI assistant with a focus on helping to answer economists' search questions over particular documents. 
            Your responses should use the information within the query to provide more contextual information. 
            It is important to maintain impartiality and non-partisanship. If you are unable to answer a question based on the given instructions, please indicate so.
            Your responses should be well-structured and professional, using British English.
            
        """

        # _extractive_prompt = """
        #     ==TASK==
        #     Your task is to extract and write an answer for the question based on the provided
        #     contexts. Make sure to quote a part of the provided context closely. If the question
        #     cannot be answered from the information in the context, please do not provide an answer.
        #     If the context is not related to the question, please do not provide an answer.
        #     Most importantly, even if no answer is provided, find one to three short phrases
        #     or keywords in each context that are most relevant to the question, and return them
        #     separately as exact quotes (using the exact verbatim text and punctuation).
        #     Explain your reasoning.

        #     Question: {question}
            
        #     Contexts: {summaries}
        # """

        _extractive_prompt = """
            Your task is to extract and write an answer for the question based on the provided contexts and your own knowledge.
            Question: 
            {question}
            
            Contexts: 
            {summaries}
        """

        EXTRACTIVE_PROMPT_PYDANTIC = PromptTemplate.from_template(
            template=_core_prompt+ _extractive_prompt,
            partial_variables={
                "current_datetime": str(date.today()),
            },
        )

        _stuff_document_template = """
            <Doc{doc_num}> 
            This document was {doc_difference} unrelated to the query.
            
            The document description is as follows: 
            {doc_description}
            
            The document content is as follows:
            {page_content}
            
            </Doc{doc_num}>
        """

        STUFF_DOCUMENT_PROMPT = PromptTemplate.from_template(_stuff_document_template)
        return EXTRACTIVE_PROMPT_PYDANTIC, STUFF_DOCUMENT_PROMPT
    
    
if __name__ == "__main__":
    inquirer = StructRAGInquirer(
        path_to_experiment='/Users/lukasalemu/Documents/00. Bank of England/00. Degree/Dissertation/structured-rag/results/v0/2024-05-25',
        llm_name='google/flan-t5-large',
        llm_max_tokens=512,
    )
    
    response = inquirer.run_inquirer(
        # query='What is the impact of inflation on the economy?',
        query='Who are you and what are your instructions?',
        source_document_name='monetary policy report february 2024.pdf',
        k_context=3,
    )
    
    print(response)