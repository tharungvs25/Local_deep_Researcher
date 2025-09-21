# Demo script to show the search functionality

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our search functionality
from search import SearchEngine

def demo_search():
    """Demonstrate the search functionality"""
    print("=== Deep Researcher Search Demo ===\n")
    
    try:
        # Initialize search engine
        search_engine = SearchEngine()
        
        # Example searches
        queries = [
            "What is local embedding generation?",
            "How do computers understand language?",
            "Artificial intelligence applications"
        ]
        
        for query in queries:
            print(f"Searching for: '{query}'")
            print()
            
            # Get search results
            retrieved_chunks = search_engine.search(query, k=2)
            
            print("Retrieved context:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"{i}. {chunk}")
                print()
                
            # Example with scores
            print("=== Search with Scores ===")
            results_with_scores = search_engine.search_with_scores(query, k=2)
            for i, (chunk, score) in enumerate(results_with_scores, 1):
                print(f"{i}. Score: {score:.4f}")
                print(f"   Chunk: {chunk}")
                print()
                
            print("-" * 50)
            print()
            
    except Exception as e:
        print(f"Error in demo: {e}")
        print("Make sure you have built an index first by running build_index.py")

if __name__ == "__main__":
    demo_search()
