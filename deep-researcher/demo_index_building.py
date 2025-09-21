# Demo script to show the index building process

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our functions
from build_index import build_index_from_chunks, save_index_and_chunks
from data_loader import load_all_text_files

def demo_index_building():
    """Demonstrate building an index from data"""
    print("=== Deep Researcher Index Building Demo ===\n")
    
    # Load data from the data directory
    print("1. Loading documents from data directory...")
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
    
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Build index
    print("2. Building FAISS index...")
    index, processed_chunks = build_index_from_chunks(chunks)
    
    # Save index and chunks
    print("3. Saving index and chunks...")
    save_index_and_chunks(index, processed_chunks)
    
    print("\n=== Demo Complete ===")
    print(f"Created index with {index.ntotal} vectors")
    print("Files created:")
    print("- document.index (FAISS index)")
    print("- chunks.json (original text chunks)")

if __name__ == "__main__":
    demo_index_building()
