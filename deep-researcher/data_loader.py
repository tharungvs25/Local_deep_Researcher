# Data Loader for Deep Researcher
# This script handles loading and chunking text data for the research application

import os
from typing import List

def load_and_chunk_data(file_path: str, chunk_size: int = 512) -> List[str]:
    """
    Load a text file and split it into manageable chunks.
    
    Args:
        file_path (str): Path to the text file
        chunk_size (int): Number of words per chunk (default: 512)
    
    Returns:
        List[str]: List of text chunks
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into words
        words = text.split()
        chunks = []
        
        # Create chunks of specified size
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def load_all_text_files(data_directory: str, chunk_size: int = 512) -> List[str]:
    """
    Load and chunk all text files in a directory.
    
    Args:
        data_directory (str): Path to the directory containing text files
        chunk_size (int): Number of words per chunk (default: 512)
    
    Returns:
        List[str]: List of all text chunks from all files
    """
    all_chunks = []
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Directory {data_directory} does not exist.")
        return all_chunks
    
    # Process all .txt files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)
            print(f"Processing {filename}...")
            chunks = load_and_chunk_data(file_path, chunk_size)
            all_chunks.extend(chunks)
    
    return all_chunks

def main():
    """Main function to demonstrate data loading and chunking"""
    # Load and chunk the sample document
    sample_file = 'data/sample_document.txt'
    chunks = load_and_chunk_data(sample_file, chunk_size=200)
    
    print(f"Loaded and chunked {len(chunks)} chunks from {sample_file}")
    
    # Display first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} (length: {len(chunk)} characters):")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

if __name__ == "__main__":
    main()
