# Streamlit UI for Deep Researcher
# A web interface for the deep research application

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher
from search import SearchEngine
from reasoning_synthesis import ReasoningSynthesisEngine
# Comment out the problematic import for now
# from enhancements import ConversationHistory, ResultExporter
from file_manager import FileManager
import json

# Set page config
st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #2c7744;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: #f0f8f0;
        border-left: 5px solid #2c7744;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .query-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #888;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'search_engine' not in st.session_state:
    try:
        st.session_state.search_engine = SearchEngine()
        st.session_state.reasoning_engine = ReasoningSynthesisEngine(st.session_state.search_engine)
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.error(f"Error initializing search engine: {e}")

# Header
st.markdown('<h1 class="main-header">🧠 Deep Researcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Your local AI-powered research assistant</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0
    )
    
    # Number of results
    k_results = st.slider("Number of results", 1, 10, 5)
    
    # Clear history button
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")
    
    # File Management Section
    st.header("📁 File Management")
    
    # Initialize file manager
    fm = FileManager()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        # Read the file content as text
        file_content = uploaded_file.read().decode("utf-8")
        # Save to data directory with the uploaded filename
        file_path = os.path.join(fm.data_directory, uploaded_file.name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    # List files button
    if st.button("📋 List Uploaded Files"):
        files = fm.list_uploaded_files()
        if files:
            st.subheader("Uploaded Files:")
            for file in files:
                info = fm.get_file_info(file)
                if info:
                    st.write(f"- {file} ({info['chunks']} chunks)")
        else:
            st.info("No files uploaded yet")
    
    # Remove file input
    remove_file_name = st.text_input("File name to remove")
    if st.button("🗑️ Remove File") and remove_file_name:
        success = fm.remove_file(remove_file_name)
        if success:
            st.success(f"File '{remove_file_name}' removed successfully")
        else:
            st.error(f"Failed to remove file '{remove_file_name}'")
    
    # Rebuild index button
    if st.button("🔄 Rebuild Index"):
        success = fm.rebuild_index()
        if success:
            st.success("Index rebuilt successfully")
        else:
            st.error("Failed to rebuild index")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Your Research Question")
    
    # Query input
    user_query = st.text_input(
        "Enter your research question:",
        placeholder="e.g., What are the key areas of artificial intelligence?",
        key="query_input"
    )
    
    # Search button
    if st.button("🔍 Research", type="primary"):
        if user_query.strip():
            with st.spinner("Searching and analyzing..."):
                try:
                    # Perform RAG search
                    result = st.session_state.reasoning_engine.rag_pipeline(
                        user_query, 
                        k=k_results
                    )
                    
                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        "query": user_query,
                        "result": result
                    })
                    
                    # Display results
                    st.markdown('<div class="query-box">', unsafe_allow_html=True)
                    st.markdown(f"**Query:** {user_query}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 📋 Retrieved Context")
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text_area("", chunk, height=100, key=f"chunk_{i}")
                    
                    st.markdown("### 🧠 Final Answer")
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(result['final_answer'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show reasoning steps
                    st.markdown("### 🔍 Reasoning Steps")
                    st.markdown("**Generated sub-questions:**")
                    for i, sq in enumerate(result['sub_questions'], 1):
                        st.markdown(f"{i}. {sq}")
                        
                except Exception as e:
                    st.error(f"Error during research: {e}")
        else:
            st.warning("Please enter a research question.")

with col2:
    st.subheader("📋 Recent Conversations")
    
    if st.session_state.conversation_history:
        for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
            with st.expander(f"Query {i}: {entry['query'][:50]}..."):
                st.markdown(f"**Query:** {entry['query']}")
                st.markdown(f"**Results:** {len(entry['result']['retrieved_chunks'])} chunks retrieved")
                st.markdown("### Preview:")
                st.text_area("", entry['result']['final_answer'][:200] + "...", height=100, key=f"preview_{i}")
    else:
        st.info("No conversations yet. Start by asking a research question!")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Deep Researcher - Local AI Research Assistant | All processing happens locally on your machine")
st.markdown('</div>', unsafe_allow_html=True)
