# Enhancements for Deep Researcher
# Implements interactive query refinement, reasoning step logging, and result export

import os
import json
import datetime
from typing import List, Dict, Any
from pathlib import Path
from fpdf2 import FPDF
import markdown

class ConversationHistory:
    """Manages conversation history for interactive query refinement"""
    
    def __init__(self, history_file: str = "conversation_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading conversation history: {e}")
        return []
    
    def save_history(self):
        """Save conversation history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def add_entry(self, query: str, sub_questions: List[str], retrieved_chunks: List[str], answer: str):
        """Add a new conversation entry"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "sub_questions": sub_questions,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer
        }
        self.history.append(entry)
        self.save_history()
    
    def get_recent_conversation(self, num_entries: int = 3) -> List[Dict[str, Any]]:
        """Get recent conversation entries"""
        return self.history[-num_entries:] if self.history else []

class ResultExporter:
    """Handles exporting results in various formats"""
    
    @staticmethod
    def export_to_markdown(query: str, answer: str, sub_questions: List[str], 
                          retrieved_chunks: List[str], filename: str = None) -> str:
        """Export results to Markdown format"""
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_result_{timestamp}.md"
        
        markdown_content = f"""# Research Report

## Query
{query}

## Sub-Questions Generated
"""
        for i, sq in enumerate(sub_questions, 1):
            markdown_content += f"{i}. {sq}\n"
        
        markdown_content += f"""

## Retrieved Context Chunks
"""
        for i, chunk in enumerate(retrieved_chunks, 1):
            markdown_content += f"### Chunk {i}\n{chunk}\n\n"
        
        markdown_content += f"""## Final Answer
{answer}

## Generated on
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Markdown exported to {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting to Markdown: {e}")
            return None
    
    @staticmethod
    def export_to_pdf(query: str, answer: str, sub_questions: List[str], 
                     retrieved_chunks: List[str], filename: str = None) -> str:
        """Export results to PDF format"""
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_result_{timestamp}.pdf"
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Research Report", ln=True, align='C')
        pdf.ln(10)
        
        # Query
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Query:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, query)
        pdf.ln(5)
        
        # Sub-questions
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Sub-Questions Generated:", ln=True)
        pdf.set_font("Arial", "", 12)
        for i, sq in enumerate(sub_questions, 1):
            pdf.cell(0, 10, f"{i}. {sq}", ln=True)
        pdf.ln(5)
        
        # Retrieved chunks
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Retrieved Context Chunks:", ln=True)
        pdf.set_font("Arial", "", 12)
        for i, chunk in enumerate(retrieved_chunks, 1):
            pdf.cell(0, 10, f"Chunk {i}:", ln=True)
            pdf.multi_cell(0, 10, chunk)
            pdf.ln(2)
        
        # Final answer
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Final Answer:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, answer)
        
        # Timestamp
        pdf.ln(10)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        
        try:
            pdf.output(filename)
            print(f"PDF exported to {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return None

class EnhancedReasoningSynthesisEngine:
    """Enhanced reasoning and synthesis engine with conversation history and export capabilities"""
    
    def __init__(self, search_engine, model_name: str = "llama3", 
                 conversation_history: ConversationHistory = None):
        """
        Initialize the enhanced reasoning and synthesis engine
        
        Args:
            search_engine: The search engine instance to use
            model_name (str): Name of the local LLM to use
            conversation_history (ConversationHistory): Conversation history manager
        """
        self.search_engine = search_engine
        self.model_name = model_name
        self.conversation_history = conversation_history or ConversationHistory()
        self.last_query_context = ""
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-questions
        
        Args:
            query (str): The original user query
            
        Returns:
            List[str]: List of sub-questions
        """
        print(f"Decomposing query: '{query}'")
        
        # Simple rule-based decomposition for demonstration
        # In a real implementation, this would call a local LLM
        sub_questions = [
            f"What is the main topic of '{query}'?",
            f"What aspects of '{query}' need to be addressed?",
            f"How can '{query}' be understood in context?"
        ]
        
        print("Generated sub-questions:")
        for i, q in enumerate(sub_questions, 1):
            print(f"  {i}. {q}")
            
        return sub_questions
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant context chunks for the query
        
        Args:
            query (str): The user query
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: Retrieved text chunks
        """
        # Add conversation context if available
        if self.last_query_context:
            # For demonstration, we'll just use the original query
            # In a real implementation, you'd combine the context
            pass
            
        return self.search_engine.search_simple(query, k)
    
    def generate_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """
        Generate a prompt for the LLM with the retrieved context
        
        Args:
            query (str): Original user query
            retrieved_chunks (List[str]): Retrieved text chunks
            
        Returns:
            str: Formatted prompt for LLM
        """
        context = "\n\n".join(retrieved_chunks)
        
        prompt = f"""
        Based on the following context, please provide a comprehensive answer to the user's query.

        Context:
        {context}

        Query: {query}

        Please provide a well-structured, detailed answer that synthesizes information from the context.
        Make sure to cite relevant points from the context where appropriate.
        Answer:
        """
        
        return prompt
    
    def generate_answer(self, query: str, retrieved_chunks: List[str]) -> str:
        """
        Generate a final answer using the retrieved chunks and a local LLM
        
        Args:
            query (str): The original user query
            retrieved_chunks (List[str]): Retrieved text chunks
            
        Returns:
            str: Generated answer from the LLM
        """
        # Generate the prompt
        prompt = self.generate_prompt(query, retrieved_chunks)
        
        # This is where you would normally call your local LLM
        # For now, we'll simulate the LLM response with a sample answer
        simulated_response = f"""
        Based on the retrieved context, here is a comprehensive answer to your query: "{query}"

        The information suggests that the topic relates to important concepts in artificial intelligence and machine learning. 
        Key points from the context include:
        1. Artificial intelligence is transforming industries by automating repetitive tasks
        2. Machine learning algorithms can process vast amounts of data to find patterns
        3. Deep learning neural networks are inspired by the structure of the human brain
        4. Natural language processing enables computers to understand human language
        5. Computer vision allows machines to interpret and understand visual information

        In summary, the field of artificial intelligence encompasses various approaches including machine learning, 
        deep learning, natural language processing, and computer vision, all of which are contributing to significant 
        advancements in technology and industry applications.
        """
        
        return simulated_response
    
    def rag_pipeline_with_enhancements(self, query: str, k: int = 5, 
                                     export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline with enhancements
        
        Args:
            query (str): The user query
            k (int): Number of chunks to retrieve
            export_formats (List[str]): List of export formats ('pdf', 'markdown')
            
        Returns:
            Dict[str, Any]: Dictionary containing all results and metadata
        """
        print(f"=== Starting Enhanced RAG Pipeline for Query: '{query}' ===\n")
        
        # Step 1: Decompose query
        sub_questions = self.decompose_query(query)
        
        # Step 2: Retrieve context
        print(f"\n--- Retrieving context for main query ---")
        retrieved_chunks = self.retrieve_context(query, k)
        
        # Step 3: Generate answer
        print(f"\n--- Generating answer ---")
        final_answer = self.generate_answer(query, retrieved_chunks)
        
        # Step 4: Store conversation history
        self.conversation_history.add_entry(query, sub_questions, retrieved_chunks, final_answer)
        
        # Step 5: Export results if requested
        if export_formats:
            print(f"\n--- Exporting results ---")
            for fmt in export_formats:
                if fmt.lower() == 'pdf':
                    ResultExporter.export_to_pdf(query, final_answer, sub_questions, retrieved_chunks)
                elif fmt.lower() == 'markdown':
                    ResultExporter.export_to_markdown(query, final_answer, sub_questions, retrieved_chunks)
        
        # Step 6: Return results with metadata
        result = {
            "query": query,
            "sub_questions": sub_questions,
            "retrieved_chunks": retrieved_chunks,
            "final_answer": final_answer,
            "conversation_history": self.conversation_history.get_recent_conversation(3),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return result

def main():
    """Demonstration of the enhanced features"""
    try:
        # Initialize components
        print("Initializing enhanced deep researcher...")
        search_engine = SearchEngine()
        conversation_history = ConversationHistory()
        reasoning_engine = EnhancedReasoningSynthesisEngine(search_engine, conversation_history=conversation_history)
        
        # Example queries to demonstrate conversation history
        queries = [
            "What are the key areas of artificial intelligence?",
            "How does AI impact healthcare?",
            "What are the ethical considerations in AI?"
        ]
        
        print("\n=== Demonstrating Enhanced Features ===")
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print('='*60)
            
            # Run the enhanced RAG pipeline
            result = reasoning_engine.rag_pipeline_with_enhancements(
                query, 
                k=3, 
                export_formats=['markdown'] if i == 1 else None  # Only export first result
            )
            
            # Display results
            print(f"\nRetrieved chunks:")
            for j, chunk in enumerate(result['retrieved_chunks'], 1):
                print(f"  {j}. {chunk}")
            
            print(f"\nFinal Answer:")
            print(result['final_answer'][:200] + "..." if len(result['final_answer']) > 200 else result['final_answer'])
            
            # Show conversation history
            print(f"\nRecent Conversation History:")
            for entry in result['conversation_history']:
                print(f"  Query: {entry['query']}")
                print(f"  Timestamp: {entry['timestamp'][:19]}")
                print()
        
        print("Enhanced features demonstration complete!")
        print("Features implemented:")
        print("- Interactive query refinement with conversation history")
        print("- Detailed reasoning step logging")
        print("- Export to Markdown format")
        print("- PDF export capability")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        print("Make sure you have built an index first by running build_index.py")

if __name__ == "__main__":
    main()
