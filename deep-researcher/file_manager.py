# File Manager for Deep Researcher
# Handles uploading, managing, and indexing of text files

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import json
from data_loader import load_and_chunk_data, load_all_text_files
from build_index import build_index_from_chunks, save_index_and_chunks
from deep_researcher import DeepResearcher

class FileManager:
    """Manages file uploads, storage, and indexing for the Deep Researcher"""
    
    def __init__(self, data_directory: str = 'data', index_file: str = 'document.index', 
                 chunks_file: str = 'chunks.json'):
        """
        Initialize the file manager
        
        Args:
            data_directory (str): Directory to store uploaded files
            index_file (str): Path to the FAISS index file
            chunks_file (str): Path to the chunks JSON file
        """
        self.data_directory = Path(data_directory)
        self.index_file = index_file
        self.chunks_file = chunks_file
        
        # Create data directory if it doesn't exist
        self.data_directory.mkdir(exist_ok=True)
        
    def upload_file(self, file_path: str, chunk_size: int = 200) -> bool:
        """
        Upload a text file to the data directory
        
        Args:
            file_path (str): Path to the file to upload
            chunk_size (int): Number of words per chunk
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist")
                return False
                
            # Get file name
            file_name = os.path.basename(file_path)
            
            # Copy file to data directory
            destination_path = self.data_directory / file_name
            shutil.copy2(file_path, destination_path)
            
            print(f"✅ File '{file_name}' uploaded successfully to {self.data_directory}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading file: {e}")
            return False
    
    def list_uploaded_files(self) -> List[str]:
        """
        List all uploaded text files
        
        Returns:
            List[str]: List of file names
        """
        try:
            files = []
            for file_path in self.data_directory.glob('*.txt'):
                files.append(file_path.name)
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def remove_file(self, file_name: str) -> bool:
        """
        Remove a file from the data directory
        
        Args:
            file_name (str): Name of the file to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = self.data_directory / file_name
            if file_path.exists():
                file_path.unlink()
                print(f"✅ File '{file_name}' removed successfully")
                return True
            else:
                print(f"❌ File '{file_name}' not found")
                return False
        except Exception as e:
            print(f"❌ Error removing file: {e}")
            return False
    
    def get_file_info(self, file_name: str) -> Optional[Dict]:
        """
        Get information about a specific file
        
        Args:
            file_name (str): Name of the file
            
        Returns:
            Optional[Dict]: File information or None if not found
        """
        try:
            file_path = self.data_directory / file_name
            if not file_path.exists():
                return None
                
            # Get file stats
            stat = file_path.stat()
            
            # Load and chunk the file to get chunk count
            chunks = load_and_chunk_data(str(file_path), chunk_size=200)
            
            return {
                "name": file_name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "chunks": len(chunks),
                "path": str(file_path)
            }
        except Exception as e:
            print(f"Error getting file info: {e}")
            return None
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from all uploaded text files
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("🔄 Rebuilding index from uploaded files...")
            
            # Load all text files from data directory
            chunks = load_all_text_files(str(self.data_directory), chunk_size=200)
            
            if not chunks:
                print("⚠️  No text files found to index")
                return False
            
            print(f"Loaded {len(chunks)} chunks from {len(self.list_uploaded_files())} files")
            
            # Build new index
            index, processed_chunks = build_index_from_chunks(chunks)
            
            # Save index and chunks
            save_index_and_chunks(index, processed_chunks, self.index_file, self.chunks_file)
            
            print("✅ Index rebuilt successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            return False
    
    def add_file_to_index(self, file_name: str) -> bool:
        """
        Add a single file to the existing index without rebuilding the entire index
        
        Args:
            file_name (str): Name of the file to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the specific file
            file_path = self.data_directory / file_name
            if not file_path.exists():
                print(f"❌ File '{file_name}' not found")
                return False
                
            chunks = load_and_chunk_data(str(file_path), chunk_size=200)
            
            if not chunks:
                print(f"⚠️  No chunks found in file '{file_name}'")
                return False
            
            # Load existing index and chunks
            researcher = DeepResearcher(self.index_file, self.chunks_file)
            
            # Add new chunks to existing documents
            researcher.add_documents(chunks)
            
            # Save updated index
            researcher.save_index()
            
            print(f"✅ Added {len(chunks)} chunks from '{file_name}' to index")
            return True
            
        except Exception as e:
            print(f"❌ Error adding file to index: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index
        
        Returns:
            Dict: Index statistics
        """
        try:
            # Check if index files exist
            index_exists = os.path.exists(self.index_file)
            chunks_exists = os.path.exists(self.chunks_file)
            
            stats = {
                "index_exists": index_exists,
                "chunks_exists": chunks_exists,
                "total_vectors": 0,
                "total_chunks": 0,
                "uploaded_files": len(self.list_uploaded_files())
            }
            
            if index_exists and chunks_exists:
                # Load existing chunks to get count
                with open(self.chunks_file, 'r') as f:
                    chunks = json.load(f)
                stats["total_chunks"] = len(chunks)
                
                # Load index to get vector count
                import faiss
                index = faiss.read_index(self.index_file)
                stats["total_vectors"] = index.ntotal
                
            return stats
            
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}

def main():
    """Demonstration of file management functionality"""
    print("=== File Manager Demo ===")
    
    # Initialize file manager
    fm = FileManager()
    
    # Show current files
    print("\n1. Uploaded files:")
    files = fm.list_uploaded_files()
    if files:
        for file in files:
            info = fm.get_file_info(file)
            if info:
                print(f"   - {file} ({info['chunks']} chunks)")
    else:
        print("   No files uploaded yet")
    
    # Show index stats
    print("\n2. Index Statistics:")
    stats = fm.get_index_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
