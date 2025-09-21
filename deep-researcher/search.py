# Search Pipeline for Deep Researcher
# This script implements the core search functionality

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class SearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='document.index', chunks_file='chunks.json'):
        """
        Initialize the search engine with model and index files
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            index_file (str): Path to the FAISS index file
            chunks_file (str): Path to the chunks JSON file
        """
        self.model_name = model_name
        self.index_file = index_file
        self.chunks_file = chunks_file
        
        # Load model, index, and chunks
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_file)
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)
        
        print(f"Search engine initialized with model: {model_name}")
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} text chunks")
    
    def search(self, query_text, k=5):
        """
        Searches the index for the top k most relevant chunks.
        
        Args:
            query_text (str): The user's search query
            k (int): Number of top results to return
            
        Returns:
            list: List of text chunks that are most relevant to the query
        """
        # Encode the query
        query_embedding = self.model.encode([query_text])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the actual text chunks
        results = [self.chunks[i] for i in indices[0]]
        return results
    
    def search_with_scores(self, query_text, k=5):
        """
        Searches the index and returns results with their similarity scores.
        
        Args:
            query_text (str): The user's search query
            k (int): Number of top results to return
            
        Returns:
            list: List of tuples (chunk, score) sorted by relevance
        """
        # Encode the query
        query_embedding = self.model.encode([query_text])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results with scores
        results = [(self.chunks[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
        return results

def main():
    """Example usage of the search functionality"""
    try:
        # Initialize search engine
        search_engine = SearchEngine()
        
        # Example search
        query = "What is local embedding generation?"
        print(f"Searching for: '{query}'")
        print()
        
        # Get search results
        retrieved_chunks = search_engine.search(query, k=3)
        
        print("Retrieved context:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"{i}. {chunk}")
            print()
            
        # Example with scores
        print("=== Search with Scores ===")
        results_with_scores = search_engine.search_with_scores(query, k=3)
        for i, (chunk, score) in enumerate(results_with_scores, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Chunk: {chunk}")
            print()
            
    except Exception as e:
        print(f"Error in main function: {e}")
        print("Make sure you have built an index first by running build_index.py")

if __name__ == "__main__":
    main()
