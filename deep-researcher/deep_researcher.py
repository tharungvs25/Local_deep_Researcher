# Deep Researcher Application
# This application uses sentence transformers for embeddings and FAISS for vector search

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json

# Import data loader
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from data_loader import load_all_text_files
except ImportError:
    # Fallback if data_loader is not available
    def load_all_text_files(data_directory, chunk_size=512):
        return []

class DeepResearcher:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='document.index', chunks_file='chunks.json'):
        """
        Initialize the Deep Researcher with a sentence transformer model
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            index_file (str): Path to the FAISS index file
            chunks_file (str): Path to the chunks JSON file
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.index_file = index_file
        self.chunks_file = chunks_file
        self.load_existing_index()
        
    def load_existing_index(self):
        """
        Load existing FAISS index and chunks if they exist
        """
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
                print("Loading existing index and chunks...")
                self.index = faiss.read_index(self.index_file)
                with open(self.chunks_file, 'r') as f:
                    self.documents = json.load(f)
                print(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} chunks")
            else:
                print("No existing index found. Will create new index when needed.")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            self.index = None
            self.documents = []
    
    def create_embeddings(self, texts):
        """
        Create embeddings for a list of texts
        """
        return self.model.encode(texts)
    
    def setup_vector_index(self, embeddings):
        """
        Setup FAISS index for similarity search
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype(np.float32))
    
    def build_index_from_chunks(self, chunks):
        """
        Build FAISS index from text chunks
        
        Args:
            chunks (list): List of text chunks to index
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.create_embeddings(chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        # Store chunks
        self.documents = chunks
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def save_index(self):
        """
        Save the FAISS index and chunks to disk
        """
        if self.index is not None and self.documents:
            try:
                # Save the index
                faiss.write_index(self.index, self.index_file)
                print(f"Index saved to {self.index_file}")
                
                # Save chunks as JSON
                with open(self.chunks_file, 'w') as f:
                    json.dump(self.documents, f)
                print(f"Chunks saved to {self.chunks_file}")
            except Exception as e:
                print(f"Error saving index: {e}")
        else:
            print("No index or documents to save")
    
    def add_documents(self, documents):
        """
        Add documents to the research database
        """
        self.documents.extend(documents)
        # Create embeddings for all documents
        embeddings = self.create_embeddings(documents)
        # Setup or update the vector index
        if self.index is None:
            self.setup_vector_index(embeddings)
        else:
            # For simplicity, we'll recreate the index with new documents
            # In production, you'd want to use a more sophisticated approach
            all_embeddings = self.create_embeddings(self.documents)
            self.setup_vector_index(all_embeddings)
    
    def search(self, query, k=5):
        """
        Search for most similar documents to the query
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            list: List of dictionaries with 'document' and 'score' keys
        """
        if self.index is None:
            raise ValueError("No documents indexed yet. Please build an index first.")
            
        query_embedding = self.create_embeddings([query])
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'document': self.documents[idx],
                'score': float(score)
            })
        
        return results
    
    def search_simple(self, query, k=5):
        """
        Simple search that returns just the text chunks (backward compatibility)
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            list: List of text chunks
        """
        if self.index is None:
            raise ValueError("No documents indexed yet. Please build an index first.")
            
        query_embedding = self.create_embeddings([query])
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])
            
        return results
    
    def rag_search(self, query: str, k: int = 5) -> dict:
        """
        Perform a complete RAG (Retrieval-Augmented Generation) search
        
        Args:
            query (str): The user query
            k (int): Number of chunks to retrieve
            
        Returns:
            dict: Dictionary containing query, retrieved chunks, and a placeholder for the final answer
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.search_simple(query, k)
        
        # Return the results
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks
        }
    
    def enhanced_rag_search(self, query: str, k: int = 5, 
                           export_formats: list = None) -> dict:
        """
        Perform an enhanced RAG search with conversation history and export capabilities
        
        Args:
            query (str): The user query
            k (int): Number of chunks to retrieve
            export_formats (list): List of export formats ('pdf', 'markdown')
            
        Returns:
            dict: Dictionary containing all results and metadata
        """
        # This method would typically integrate with the enhanced reasoning engine
        # For now, we'll just return the basic RAG results
        retrieved_chunks = self.search_simple(query, k)
        
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "timestamp": json.dumps(datetime.datetime.now().isoformat())
        }

def main():
    # Example usage
    researcher = DeepResearcher()
    
    # Try to load documents from data directory
    try:
        print("Loading documents from data directory...")
        documents = load_all_text_files('data', chunk_size=200)
        
        if documents:
            print(f"Loaded {len(documents)} chunks from data files")
            
            # Check if we need to build a new index
            if researcher.index is None:
                print("Building new index from loaded documents...")
                researcher.build_index_from_chunks(documents)
                researcher.save_index()
            else:
                print("Using existing index")
        else:
            # Fallback to sample documents if no data files found
            print("No data files found, using sample documents...")
            sample_docs = [
                "Artificial intelligence is transforming industries by automating repetitive tasks.",
                "Machine learning algorithms can process vast amounts of data to find patterns.",
                "Deep learning neural networks are inspired by the structure of the human brain.",
                "Natural language processing enables computers to understand human language.",
                "Computer vision allows machines to interpret and understand visual information."
            ]
            if researcher.index is None:
                print("Building new index from sample documents...")
                researcher.build_index_from_chunks(sample_docs)
                researcher.save_index()
            else:
                researcher.add_documents(sample_docs)
    except Exception as e:
        print(f"Error loading documents: {e}")
        # Use sample documents as fallback
        sample_docs = [
            "Artificial intelligence is transforming industries by automating repetitive tasks.",
            "Machine learning algorithms can process vast amounts of data to find patterns.",
            "Deep learning neural networks are inspired by the structure of the human brain.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and understand visual information."
        ]
        if researcher.index is None:
            print("Building new index from sample documents...")
            researcher.build_index_from_chunks(sample_docs)
            researcher.save_index()
        else:
            researcher.add_documents(sample_docs)
    
    # Perform a search
    query = "How do computers understand language?"
    print(f"\nSearching for: '{query}'")
    
    results = researcher.search(query, k=3)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Document: {result['document']}\n")

if __name__ == "__main__":
    main()
