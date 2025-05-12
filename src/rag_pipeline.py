import os
import logging
import requests
import re
from dotenv import load_dotenv
from preprocessing import DocumentPreprocessor
from retriever import Retriever
from typing import List, Dict, Set

# Load environment variables first, before any initialization
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, pinecone_api_key: str = None, pinecone_environment: str = None, 
                ai_api_url: str = None, index_name: str = "rag-index"):
        """Initialize the RAG pipeline with preprocessor, retriever, and custom LLM API."""
        logger.info("Initializing RAG Pipeline...")
        
        # Get API keys from environment variables if not provided
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = pinecone_environment or os.getenv("PINECONE_ENVIRONMENT")
        self.ai_api_url = ai_api_url or os.getenv("AI_API_URL")
        
        # Validate required parameters
        if not self.pinecone_api_key:
            logger.error("PINECONE_API_KEY is not set.")
            raise ValueError("PINECONE_API_KEY must be provided in .env or as a parameter.")
            
        if not self.pinecone_environment:
            logger.error("PINECONE_ENVIRONMENT is not set.")
            raise ValueError("PINECONE_ENVIRONMENT must be provided in .env or as a parameter.")
        
        if not self.ai_api_url:
            logger.error("AI_API_URL is not set.")
            raise ValueError("AI_API_URL must be provided in .env or as a parameter.")
        
        # Initialize preprocessor and retriever
        self.preprocessor = DocumentPreprocessor(self.pinecone_api_key, self.pinecone_environment, index_name)
        self.retriever = Retriever(self.pinecone_api_key, self.pinecone_environment, index_name)
        
        # Initialize BM25 with chunks from Pinecone
        logger.info("Fetching chunks for BM25 initialization...")
        index = self.preprocessor.pinecone_client.Index(index_name)
        query_results = index.query(vector=[0]*384, top_k=1000, include_metadata=True)["matches"]
        chunks = [match["metadata"]["text"] for match in query_results]
        self.retriever.initialize_bm25(chunks)
        logger.info(f"BM25 initialized with {len(chunks)} chunks.")
        
        # Initialize custom LLM API
        logger.info("Initializing custom LLM API...")
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Load prompt template
        logger.info("Loading prompt template...")
        with open("config/prompt_template.txt", "r") as f:
            self.prompt_template = f.read().strip()
        logger.info("RAG Pipeline initialized successfully.")

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context string."""
        logger.info(f"Formatting {len(retrieved_chunks)} retrieved chunks into context...")
        context = "\n\n".join([f"Chunk {i+1}: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        logger.info("Context formatting complete.")
        return context

    def extract_capacities_from_chunks(self, retrieved_chunks: List[Dict], project_name: str) -> Dict:
        """
        Extract capacity information from chunks, along with contextual details.
        Returns a dictionary mapping capacity values to their context.
        """
        logger.info(f"Extracting capacity information for {project_name}...")
        
        capacity_info = {}
        # Pattern to match capacity values (e.g., "600 MW", "83 MW", etc.)
        capacity_pattern = r'(\d+(?:\.\d+)?)\s*(?:mw|MW)'
        
        for i, chunk in enumerate(retrieved_chunks):
            text = chunk['text']
            # Check if the chunk mentions the project
            if project_name.lower() in text.lower():
                # Find all capacity mentions
                capacities = re.findall(capacity_pattern, text)
                for capacity in capacities:
                    # Extract a short context around the capacity mention
                    # First, find the position of this capacity in the text
                    capacity_pos = text.lower().find(f"{capacity} mw") 
                    if capacity_pos == -1:
                        capacity_pos = text.lower().find(f"{capacity} MW")
                    
                    if capacity_pos != -1:
                        # Extract context (up to 100 chars before and after)
                        start = max(0, capacity_pos - 100)
                        end = min(len(text), capacity_pos + 100)
                        context_text = text[start:end]
                        
                        # Clean up the context - ensure it starts and ends with complete words
                        if start > 0:
                            context_text = context_text[context_text.find(' ')+1:]
                        if end < len(text):
                            last_space = context_text.rfind(' ')
                            if last_space != -1:
                                context_text = context_text[:last_space]
                        
                        # Store with chunk reference
                        key = f"{capacity} MW"
                        if key not in capacity_info:
                            capacity_info[key] = []
                        
                        context_entry = {
                            "chunk_num": i+1,
                            "context": context_text
                        }
                        
                        # Only add if this exact context isn't already in the list
                        if not any(entry["context"] == context_text for entry in capacity_info[key]):
                            capacity_info[key].append(context_entry)
        
        logger.info(f"Extracted information about {len(capacity_info)} different capacity values.")
        return capacity_info

    def create_enhanced_prompt(self, query: str, capacity_info: Dict, general_context: str) -> str:
        """Create an enhanced prompt that specifically asks the LLM to reconcile different capacity values."""
        # First part - standard context
        prompt = f"{self.prompt_template}\n\nContext:\n{general_context}\n\n"
        
        # Add capacity-specific information
        prompt += "Extracted capacity information for the project:\n"
        for capacity, contexts in capacity_info.items():
            prompt += f"\nCapacity value: {capacity}\n"
            for context in contexts:
                prompt += f"- From Chunk {context['chunk_num']}: \"{context['context']}\"\n"
        
        # Special instructions for handling multiple values
        prompt += "\nSpecial instructions: The context contains multiple capacity values. Please analyze all the information carefully and provide a comprehensive answer that explains these different values. Consider whether different values might refer to different phases, units, or planning stages of the project. Cite the specific chunks that support your answer.\n\n"
        
        # Add the original query
        prompt += f"Query:\n{query}\n\nAnswer:"
        
        return prompt

    def generate_answer(self, query: str, top_k: int = 10) -> str:
        """Generate an answer using retrieved context and custom LLM API."""
        logger.info(f"Generating answer for query: {query}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        if not retrieved_chunks:
            logger.warning("No chunks retrieved for the query.")
            return "No relevant information found in the document."
        
        # Format standard context
        general_context = self.format_context(retrieved_chunks)
        
        # Handle specific capacity-related queries
        if "capacity" in query.lower() and any(project in query.lower() for project in ["omkareshwar", "dam", "solar", "power"]):
            # Extract project name - we assume it's "Omkareshwar Dam Floating Solar Power Project"
            project_name = "Omkareshwar Dam Floating Solar Power Project"
            if "omkareshwar" in query.lower():
                project_name = "Omkareshwar Dam Floating Solar Power Project"
            
            # Extract capacity information with context
            capacity_info = self.extract_capacities_from_chunks(retrieved_chunks, project_name)
            
            if capacity_info:
                # Create enhanced prompt for capacity questions
                prompt = self.create_enhanced_prompt(query, capacity_info, general_context)
            else:
                # Fallback to standard prompt
                prompt = f"{self.prompt_template}\n\nContext:\n{general_context}\n\nQuery:\n{query}\n\nAnswer:"
        else:
            # Standard prompt for non-capacity queries
            prompt = f"{self.prompt_template}\n\nContext:\n{general_context}\n\nQuery:\n{query}\n\nAnswer:"
        
        logger.info("Prompt constructed, invoking LLM API...")
        
        # Call custom LLM API
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(self.ai_api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            
            if not result.get("done"):
                logger.error("LLM API response not complete.")
                return "Error: LLM response not complete."
            
            answer = result.get("response", "").strip()
            logger.info("Answer generated successfully.")
            return answer if answer else "No answer provided by the LLM."
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return f"Error generating answer: {str(e)}"
        except ValueError as e:
            logger.error(f"Error parsing LLM API response: {str(e)}")
            return f"Error parsing LLM response: {str(e)}"

if __name__ == "__main__":
    # Example usage
    logger.info("Starting RAG Pipeline test...")
    
    # No need to explicitly pass environment variables here
    # They will be loaded from .env file
    pipeline = RAGPipeline()
    
    # Test query
    query = "What is the capacity of the Omkareshwar Dam Floating Solar Power Project?"
    answer = pipeline.generate_answer(query)
    print("Query:", query)
    print("Answer:", answer)