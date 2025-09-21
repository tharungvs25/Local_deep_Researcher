# Build FAISS Index for Deep Researcher
# This script generates embeddings and builds a FAISS index from text chunks

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

def build_index_from_chunks(all_chunks, model_name='all-MiniLM-L6-v2'):
    """
    Build a FAISS index from text chunks
    
    Args:
        all_chunks (list): List of text chunks to index
        model_name (str): Name of the sentence transformer model to use
    
    Returns:
        tuple: (faiss index, chunks list)
    """
    if not all_chunks:
        raise ValueError("No chunks provided for indexing")
    
    # 1. Initialize model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 2. Generate embeddings
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # 3. Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors")
    
    return index, all_chunks

def save_index_and_chunks(index, chunks, index_filename='document.index', chunks_filename='chunks.json'):
    """
    Save the FAISS index and chunks to disk
    
    Args:
        index: FAISS index object
        chunks (list): List of text chunks
        index_filename (str): Filename for the FAISS index
        chunks_filename (str): Filename for the chunks JSON file
    """
    # Save the index
    faiss.write_index(index, index_filename)
    print(f"Index saved to {index_filename}")
    
    # Save chunks as JSON
    with open(chunks_filename, 'w') as f:
        json.dump(chunks, f)
    print(f"Chunks saved to {chunks_filename}")

def load_index_and_chunks(index_filename='document.index', chunks_filename='chunks.json'):
    """
    Load the FAISS index and chunks from disk
    
    Args:
        index_filename (str): Filename for the FAISS index
        chunks_filename (str): Filename for the chunks JSON file
    
    Returns:
        tuple: (faiss index, chunks list)
    """
    # Load the index
    index = faiss.read_index(index_filename)
    print(f"Index loaded from {index_filename}")
    
    # Load chunks
    with open(chunks_filename, 'r') as f:
        chunks = json.load(f)
    print(f"Chunks loaded from {chunks_filename}")
    
    return index, chunks

def main():
    """Main function to demonstrate building an index from data"""
    # Load data from the data directory
    try:
        from data_loader import load_all_text_files
        print("Loading documents from data directory...")
        chunks = load_all_text_files('data', chunk_size=200)
        
        if not chunks:
            print("No data files found, using sample chunks...")
            # Create some sample chunks for demonstration
            chunks = [
                "Artificial intelligence is transforming industries by automating repetitive tasks.",
                "Machine learning algorithms can process vast amounts of data to find patterns.",
                "Deep learning neural networks are inspired by the structure of the human brain.",
                "Natural language processing enables computers to understand human language.",
                "Computer vision allows machines to interpret and understand visual information."
            ]
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Build index
        index, processed_chunks = build_index_from_chunks(chunks)
        
        # Save index and chunks
        save_index_and_chunks(index, processed_chunks)
        
        print("\nIndex building completed successfully!")
        print(f"Created index with {index.ntotal} vectors")
        print("Files created:")
        print("- document.index (FAISS index)")
        print("- chunks.json (original text chunks)")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
