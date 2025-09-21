# Test script to verify the deep researcher functionality

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the data loading functionality
try:
    from data_loader import load_all_text_files
    print("Successfully imported data_loader")
    
    # Test loading data
    chunks = load_all_text_files('data', chunk_size=200)
    print(f"Successfully loaded {len(chunks)} chunks from data files")
    
    # Print first few chunks to verify
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i+1}: {chunk[:100]}...")
        
except Exception as e:
    print(f"Error with data loading: {e}")

# Test the main researcher functionality
try:
    from deep_researcher import DeepResearcher
    print("Successfully imported DeepResearcher")
    
    researcher = DeepResearcher()
    print("Created DeepResearcher instance successfully")
    
except Exception as e:
    print(f"Error with DeepResearcher: {e}")
