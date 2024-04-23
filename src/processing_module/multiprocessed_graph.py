import multiprocessing
from functools import partial
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.etl_module.etl_functions import load_documents
from src.etl_module.embedding_functions import embed_index
from src.processing_module import v0_retrieval, v1_retrieval, distance_metrics
from src.evaluation_module import output_functions, graph_scoring_functions


def calculate_distance(doc1, doc2, embed_model, fuzz_thresh):
    # print('Calculating distance between two documents...')
    
    dist = {'weight': np.mean(
        v1_retrieval.calculate_distance_vector(
            doc1, 
            doc2,
            embed_model=embed_model,
            fuzz_thresh=fuzz_thresh
        )
    )}
    
    return dist
    
def calc_graph_chunk(chunk, index):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    adjacency_dict = {}
    for doc in tqdm(chunk, desc=f"Processing chunk {chunk[0].id_}"):
        adjacency_dict[doc.id_] = {}
        for doc2 in index:
            adjacency_dict[doc.id_][doc2.id_] = calculate_distance(doc, doc2, embed_model, 80)
    return adjacency_dict


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def construct_adjacency_dict(embedded_index):
    
    num_processes = multiprocessing.cpu_count() - 1 # Leave one core so we don't slam the system
    chunks = chunkify(embedded_index, num_processes)
    
    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(partial(calc_graph_chunk, index=embedded_index.copy()), chunks), 
            total=len(chunks), 
            desc="Total progress"
        ))

    adjacency_dict = {k: v for result in results for k, v in result.items()}
    
    return adjacency_dict
        

if __name__ == '__main__':
    # For testing purposes
    print("Running")
    
    # Load the document index
    document_index = load_documents()

    # Embed the document index
    embedded_index = embed_index(document_index)
    
    num_processes = multiprocessing.cpu_count()
    chunks = chunkify(embedded_index, num_processes)

    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(calc_graph_chunk, chunks), total=len(chunks), desc="Total progress"))

    adjacency_dict = {k: v for result in results for k, v in result.items()}
    
    # Save the result to a pickle
    with open('data/03_output/adjacency_dict.pkl', 'wb') as f:
        pickle.dump(adjacency_dict, f)
        
        