import os
import logging
from src.preprocessing import DocumentPreprocessor
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def process_documents():
    """Process all documents in the docs/ folder."""
    logger.info("Starting document processing...")
    preprocessor = DocumentPreprocessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    docs_folder = "docs"
    total_chunks = 0
    
    if not os.path.exists(docs_folder):
        logger.error(f"Directory '{docs_folder}' does not exist.")
        return 0
    
    for file_name in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
            logger.info(f"Processing {file_name}...")
            chunks_processed = preprocessor.process_document(file_path)
            total_chunks += chunks_processed
            logger.info(f"Processed {chunks_processed} chunks from {file_name}")
    
    logger.info(f"Total chunks processed: {total_chunks}")
    return total_chunks

def start_api():
    """Start the FastAPI server."""
    logger.info("Starting FastAPI server...")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # Process documents (optional, can be run separately)
    process_documents()
    
    # Start the API
    start_api()