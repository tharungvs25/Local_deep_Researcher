# Comprehensive Demo of Deep Researcher
# Demonstrates the complete workflow from data ingestion to reasoning and synthesis

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from build_index import build_index_from_chunks, save_index_and_chunks
from data_loader import load_all_text_files
from search import SearchEngine
from reasoning_synthesis import ReasoningSynthesisEngine

def comprehensive_demo():
    """Demonstrate the complete deep researcher workflow"""
    print("=== Comprehensive Deep Researcher Demo ===\n")
    
    # Step 1: Load data
    print("1. Loading data from data directory...")
    chunks = load_all_text_files('data', chunk_size=200)
    
    if not chunks:
        print("No data files found, using sample chunks...")
        # Create some sample chunks for demonstration
        chunks = [
            "Artificial intelligence is transforming industries by automating repetitive tasks.",
            "Machine learning algorithms can process vast amounts of data to find patterns.",
            "Deep learning neural networks are inspired by the structure of the human brain.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Ethical considerations are important in AI development to prevent bias and protect privacy.",
            "The future of AI holds tremendous promise but also significant challenges.",
            "AI in healthcare is revolutionizing diagnostics and personalized treatment plans."
        ]
    
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Step 2: Build index
    print("2. Building FAISS index...")
    index, processed_chunks = build_index_from_chunks(chunks)
    save_index_and_chunks(index, processed_chunks)
    print("Index built and saved successfully!\n")
    
    # Step 3: Initialize search engine
    print("3. Initializing search engine...")
    search_engine = SearchEngine()
    print("Search engine ready!\n")
    
    # Step 4: Initialize reasoning and synthesis engine
    print("4. Initializing reasoning and synthesis engine...")
    reasoning_engine = ReasoningSynthesisEngine(search_engine)
    print("Reasoning engine ready!\n")
    
    # Step 5: Demonstrate complete RAG pipeline
    print("5. Demonstrating complete RAG pipeline...")
    queries = [
        "What are the key areas of artificial intelligence?",
        "How does AI impact healthcare?",
        "What are the ethical considerations in AI?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        # Run the complete RAG pipeline
        result = reasoning_engine.rag_pipeline(query, k=3)
        
        print(f"\nRetrieved chunks:")
        for j, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"  {j}. {chunk}")
        
        print(f"\nSimulated LLM Response:")
        # Just show the beginning of the response to keep output readable
        response_preview = result['final_answer'][:300] + "..." if len(result['final_answer']) > 300 else result['final_answer']
        print(response_preview)
    
    print(f"\n{'='*60}")
    print("Demo Complete!")
    print("The deep researcher can now:")
    print("- Load and process local text data")
    print("- Generate embeddings and build FAISS indexes")
    print("- Perform semantic search on local data")
    print("- Apply multi-step reasoning and synthesis")
    print("- Generate comprehensive answers using RAG")
    print('='*60)

if __name__ == "__main__":
    comprehensive_demo()
