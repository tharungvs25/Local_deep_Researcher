#!/usr/bin/env python3
"""
Test script for the file manager functionality
"""

import os
import tempfile
import shutil
from pathlib import Path

# Add the project directory to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_manager import FileManager

def test_file_manager():
    """Test the file manager functionality"""
    print("=== Testing File Manager ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize file manager with temp directory
        test_data_dir = Path(temp_dir) / "test_data"
        fm = FileManager(str(test_data_dir))
        
        # Test 1: List files when none exist
        print("\n1. Testing empty file list:")
        files = fm.list_uploaded_files()
        print(f"   Files: {files}")
        assert len(files) == 0, "Should have no files initially"
        
        # Test 2: Create a test file
        print("\n2. Creating test file:")
        test_file_path = Path(temp_dir) / "test_document.txt"
        test_content = "This is a test document for file management.\nIt contains multiple lines.\nAnd should be properly chunked."
        
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        print(f"   Created test file: {test_file_path}")
        
        # Test 3: Upload file
        print("\n3. Uploading file:")
        success = fm.upload_file(str(test_file_path))
        assert success, "File upload should succeed"
        
        # Test 4: List files after upload
        print("\n4. Listing files after upload:")
        files = fm.list_uploaded_files()
        print(f"   Files: {files}")
        assert len(files) == 1, "Should have one file"
        assert files[0] == "test_document.txt", "File name should match"
        
        # Test 5: Get file info
        print("\n5. Getting file info:")
        info = fm.get_file_info("test_document.txt")
        print(f"   File info: {info}")
        assert info is not None, "File info should be available"
        assert info["name"] == "test_document.txt", "File name should match"
        assert info["chunks"] > 0, "File should have chunks"
        
        # Test 6: Rebuild index
        print("\n6. Rebuilding index:")
        success = fm.rebuild_index()
        assert success, "Index rebuild should succeed"
        
        # Test 7: Get index stats
        print("\n7. Getting index stats:")
        stats = fm.get_index_stats()
        print(f"   Index stats: {stats}")
        assert stats["index_exists"] == True, "Index should exist"
        assert stats["chunks_exists"] == True, "Chunks file should exist"
        assert stats["total_chunks"] > 0, "Should have chunks in index"
        
        # Test 8: Remove file
        print("\n8. Removing file:")
        success = fm.remove_file("test_document.txt")
        assert success, "File removal should succeed"
        
        # Test 9: Verify file removal
        print("\n9. Verifying file removal:")
        files = fm.list_uploaded_files()
        print(f"   Files after removal: {files}")
        assert len(files) == 0, "Should have no files after removal"
        
        print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_file_manager()
