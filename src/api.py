import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from collections import deque
from dotenv import load_dotenv
import json
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Pipeline API", description="API for querying renewable energy documents using RAG")

# Initialize RAG pipeline
pipeline = RAGPipeline(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
    ai_api_url=os.getenv("AI_API_URL")
)

# Load whitelist and blacklist
with open("config/whitelist.json", "r") as f:
    whitelist = set(json.load(f)["allowed_words"])
with open("config/blacklist.json", "r") as f:
    blacklist = set(json.load(f)["blocked_words"])

# In-memory chat history (max 5 messages per session)
chat_history = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    history: List[Dict[str, str]]

def validate_query(query: str) -> bool:
    """Validate query against whitelist and blacklist."""
    query_words = set(query.lower().split())
    if query_words.intersection(blacklist):
        logger.warning(f"Query contains blacklisted words: {query}")
        return False
    if not query_words.intersection(whitelist):
        logger.warning(f"Query lacks whitelisted words: {query}")
        return False
    return True

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Handle query requests and return RAG-generated answers."""
    logger.info(f"Received query: {request.query} (Session: {request.session_id})")
    
    # Validate query
    if not validate_query(request.query):
        logger.error("Invalid query detected.")
        raise HTTPException(status_code=400, detail="Query contains invalid or non-relevant words.")
    
    # Initialize chat history for session
    if request.session_id not in chat_history:
        chat_history[request.session_id] = deque(maxlen=5)
    
    # Generate answer
    try:
        answer = pipeline.generate_answer(request.query)
        chat_history[request.session_id].append({"query": request.query, "answer": answer})
        logger.info(f"Answer generated successfully for query: {request.query}")
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
    
    # Prepare response
    response = QueryResponse(
        query=request.query,
        answer=answer,
        history=list(chat_history[request.session_id])
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)