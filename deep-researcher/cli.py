# CLI Interface for Deep Researcher
# Command-line interface for the deep research application

import argparse
import sys
import os
import json
from pathlib import Path

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher
from search import SearchEngine
from reasoning_synthesis import ReasoningSynthesisEngine
from enhancements import ConversationHistory, ResultExporter
from file_manager import FileManager

def setup_parser():
    """Setup argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Deep Researcher - Local AI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py search "What are the key areas of AI?"
  python cli.py rag "Explain machine learning in simple terms"
  python cli.py history
  python cli.py export --format pdf --query "AI applications"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Perform a simple search')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--k', type=int, default=5, help='Number of results (default: 5)')
    
    # RAG command
    rag_parser = subparsers.add_parser('rag', help='Perform RAG search with reasoning')
    rag_parser.add_argument('query', help='Research query')
    rag_parser.add_argument('--k', type=int, default=5, help='Number of results (default: 5)')
    rag_parser.add_argument('--export', choices=['pdf', 'markdown'], 
                           help='Export results to format')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show conversation history')
    history_parser.add_argument('--limit', type=int, default=5, help='Number of entries to show')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export previous results')
    export_parser.add_argument('--format', choices=['pdf', 'markdown'], required=True, 
                              help='Export format')
    export_parser.add_argument('--query', help='Query to export (uses latest if not specified)')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build or rebuild index')
    index_parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    
    # File management commands
    file_parser = subparsers.add_parser('file', help='Manage uploaded files')
    file_subparsers = file_parser.add_subparsers(dest='file_command', help='File management commands')
    
    # Upload command
    upload_parser = file_subparsers.add_parser('upload', help='Upload a text file')
    upload_parser.add_argument('file_path', help='Path to the text file to upload')
    upload_parser.add_argument('--chunk-size', type=int, default=200, help='Chunk size for text splitting')
    
    # List command
    list_parser = file_subparsers.add_parser('list', help='List uploaded files')
    
    # Remove command
    remove_parser = file_subparsers.add_parser('remove', help='Remove an uploaded file')
    remove_parser.add_argument('file_name', help='Name of the file to remove')
    
    # Stats command
    stats_parser = file_subparsers.add_parser('stats', help='Show file and index statistics')
    
    # Add to index command
    add_to_index_parser = file_subparsers.add_parser('add-to-index', help='Add a file to existing index')
    add_to_index_parser.add_argument('file_name', help='Name of the file to add to index')
    
    # Rebuild index command
    rebuild_index_parser = file_subparsers.add_parser('rebuild-index', help='Rebuild the entire index from all files')
    
    return parser

def run_search(query, k=5):
    """Run a simple search"""
    try:
        search_engine = SearchEngine()
        results = search_engine.search(query, k)
        
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 50)
        
        for i, chunk in enumerate(results, 1):
            print(f"\n{i}. {chunk}")
            
        return results
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return []

def run_rag(query, k=5, export_format=None):
    """Run RAG search with reasoning"""
    try:
        search_engine = SearchEngine()
        reasoning_engine = ReasoningSynthesisEngine(search_engine)
        
        print(f"\n🧠 Performing RAG search for: '{query}'")
        print("=" * 50)
        
        # Perform RAG search
        result = reasoning_engine.rag_pipeline_with_enhancements(
            query, 
            k=k,
            export_formats=[export_format] if export_format else []
        )
        
        # Display results
        print(f"\n📋 Query: {query}")
        print(f"\n📊 Retrieved {len(result['retrieved_chunks'])} chunks")
        
        print(f"\n📋 Retrieved Context:")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"\nChunk {i}:")
            print(chunk)
            print("-" * 30)
        
        print(f"\n🧠 Final Answer:")
        print(result['final_answer'])
        
        print(f"\n🔍 Reasoning Steps:")
        print("Generated sub-questions:")
        for i, sq in enumerate(result['sub_questions'], 1):
            print(f"  {i}. {sq}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error during RAG search: {e}")
        return {}

def show_history(limit=5):
    """Show conversation history"""
    try:
        history = ConversationHistory()
        entries = history.get_recent_conversation(limit)
        
        if not entries:
            print("\n📝 No conversation history found.")
            return
            
        print(f"\n📝 Recent Conversations (last {len(entries)}):")
        print("=" * 50)
        
        for i, entry in enumerate(entries, 1):
            print(f"\n{i}. Query: {entry['query'][:60]}...")
            print(f"   Timestamp: {entry['timestamp'][:19]}")
            print(f"   Retrieved {len(entry['retrieved_chunks'])} chunks")
            
    except Exception as e:
        print(f"❌ Error reading history: {e}")

def export_results(format_type, query=None):
    """Export previous results"""
    try:
        history = ConversationHistory()
        entries = history.get_recent_conversation(1)
        
        if not entries:
            print("❌ No previous results to export.")
            return
            
        # Use the most recent entry
        entry = entries[-1]
        
        if query and query != entry['query']:
            print(f"⚠️  Query '{query}' not found in recent history. Exporting latest result.")
        
        # Export
        if format_type == 'pdf':
            filename = ResultExporter.export_to_pdf(
                entry['query'], 
                entry['answer'], 
                entry['sub_questions'], 
                entry['retrieved_chunks']
            )
        else:  # markdown
            filename = ResultExporter.export_to_markdown(
                entry['query'], 
                entry['answer'], 
                entry['sub_questions'], 
                entry['retrieved_chunks']
            )
            
        if filename:
            print(f"✅ Results exported to: {filename}")
        else:
            print("❌ Failed to export results.")
            
    except Exception as e:
        print(f"❌ Error during export: {e}")

def build_index(data_dir='data'):
    """Build or rebuild the FAISS index"""
    try:
        from data_loader import load_all_text_files
        from build_index import build_index_from_chunks, save_index_and_chunks
        
        print(f"\n🔄 Building index from data in '{data_dir}'...")
        
        # Load data
        chunks = load_all_text_files(data_dir, chunk_size=200)
        
        if not chunks:
            print("⚠️  No data found. Using sample chunks.")
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
        save_index_and_chunks(index, processed_chunks)
        
        print("✅ Index built and saved successfully!")
        
    except Exception as e:
        print(f"❌ Error building index: {e}")

def handle_file_commands(args):
    """Handle file management commands"""
    file_manager = FileManager()
    
    if args.file_command == 'upload':
        success = file_manager.upload_file(args.file_path, args.chunk_size)
        if success:
            print("✅ File uploaded successfully")
        else:
            print("❌ Failed to upload file")
            
    elif args.file_command == 'list':
        files = file_manager.list_uploaded_files()
        if files:
            print("\n📁 Uploaded files:")
            for file in files:
                info = file_manager.get_file_info(file)
                if info:
                    print(f"   - {file} ({info['chunks']} chunks)")
        else:
            print("\n📭 No files uploaded yet")
            
    elif args.file_command == 'remove':
        success = file_manager.remove_file(args.file_name)
        if success:
            print("✅ File removed successfully")
        else:
            print("❌ Failed to remove file")
            
    elif args.file_command == 'stats':
        stats = file_manager.get_index_stats()
        print("\n📊 File and Index Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    elif args.file_command == 'add-to-index':
        success = file_manager.add_file_to_index(args.file_name)
        if success:
            print("✅ File added to index successfully")
        else:
            print("❌ Failed to add file to index")
            
    elif args.file_command == 'rebuild-index':
        success = file_manager.rebuild_index()
        if success:
            print("✅ Index rebuilt successfully")
        else:
            print("❌ Failed to rebuild index")
    else:
        print("❌ Unknown file command")
        return False
    return True

def main():
    """Main CLI function"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Handle commands
    if args.command == 'search':
        run_search(args.query, args.k)
    elif args.command == 'rag':
        run_rag(args.query, args.k, args.export)
    elif args.command == 'history':
        show_history(args.limit)
    elif args.command == 'export':
        export_results(args.format, args.query)
    elif args.command == 'index':
        build_index(args.data_dir)
    elif args.command == 'file':
        handle_file_commands(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
