# Deep Researcher

A Python application that uses sentence transformers for creating embeddings and FAISS for efficient vector similarity search.

## Features

- Text embedding generation using sentence transformers
- Vector similarity search with FAISS
- Easy-to-use API for adding documents and searching
- Interactive query refinement with conversation history
- Detailed reasoning step logging
- Export to Markdown and PDF formats
- Retrieval-Augmented Generation (RAG) pipeline
- Web-based UI with Streamlit
- Command-line interface (CLI)

## Installation

1. Make sure you have Python 3.7+ installed
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web UI (Streamlit)
```bash
streamlit run app.py
```

### Command Line Interface
```bash
python cli.py --help
```

Examples:
```bash
python cli.py search "What are the key areas of AI?"
python cli.py rag "Explain machine learning in simple terms"
python cli.py history
python cli.py export --format pdf --query "AI applications"
python cli.py file upload /path/to/my_document.txt
python cli.py file list
python cli.py file stats
python cli.py file rebuild-index
```

The web UI currently supports uploading text files (.txt) through the file uploader in the sidebar. For uploading other file types (PDF, DOC, etc.), please use the CLI commands above.

### Direct Python Usage
```bash
python deep_researcher.py
```

This will demonstrate the functionality with sample documents and a search query.

## Components

- `deep_researcher.py`: Main application with DeepResearcher class
- `data_loader.py`: Loads and chunks text data from the data/ directory
- `build_index.py`: Generates embeddings and builds FAISS indexes
- `search.py`: Implements the search functionality
- `reasoning_synthesis.py`: Implements multi-step reasoning and synthesis
- `enhancements.py`: Adds interactive query refinement, logging, and export capabilities
- `app.py`: Streamlit web interface
- `cli.py`: Command-line interface
- `requirements.txt`: List of required Python packages

## Dependencies

- [sentence-transformers](https://pypi.org/project/sentence-transformers/): For creating high-quality text embeddings
- [faiss-cpu](https://pypi.org/project/faiss-cpu/): For efficient similarity search
- [numpy](https://pypi.org/project/numpy/): For numerical operations
- [fpdf2](https://pypi.org/project/fpdf2/): For PDF export functionality
- [streamlit](https://pypi.org/project/streamlit/): For web UI
