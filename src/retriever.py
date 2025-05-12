import os
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Tuple, Set
from scipy.spatial.distance import cosine
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str = "rag-index"):
        """Initialize the retriever with Pinecone, embedding model, and reranker."""
        load_dotenv()
        logger.info("Initializing Retriever...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.pinecone_client = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
        self.index = self.pinecone_client.Index(index_name)
        self.bm25 = None
        self.chunks = None
        logger.info("Retriever initialized successfully.")

    def initialize_bm25(self, chunks: List[str]):
        """Initialize BM25 with tokenized chunks."""
        logger.info(f"Initializing BM25 with {len(chunks)} chunks...")
        self.chunks = chunks
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.info("BM25 initialized successfully.")

    def dense_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        """Perform dense retrieval using Pinecone."""
        logger.info(f"Performing dense retrieval for query: {query}")
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        retrieved = [
            {"id": match["id"], "text": match["metadata"]["text"], "score": match["score"]}
            for match in results["matches"]
        ]
        logger.info(f"Dense retrieval returned {len(retrieved)} results.")
        return retrieved

    def sparse_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        """Perform sparse retrieval using BM25."""
        if self.bm25 is None:
            logger.error("BM25 not initialized.")
            raise ValueError("BM25 not initialized. Call initialize_bm25 first.")
        logger.info(f"Performing sparse retrieval for query: {query}")
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved = [
            {"id": f"chunk_{i}", "text": self.chunks[i], "score": scores[i]}
            for i in top_indices if scores[i] > 0
        ]
        logger.info(f"Sparse retrieval returned {len(retrieved)} results.")
        return retrieved

    def extract_project_capacity(self, text: str) -> List[str]:
        """Extract project capacity mentions (like 600 MW, 90 MW) from text."""
        # Look for patterns like "X MW" or "X Megawatt" in various formats
        capacity_pattern = r'(\d+(?:\.\d+)?\s*(?:MW|Megawatt|megawatt|mw))'
        return re.findall(capacity_pattern, text, re.IGNORECASE)
    
    def deduplicate_results(self, results: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
        """Deduplicate results while ensuring capacity diversity."""
        logger.info(f"Deduplicating {len(results)} results with diversity preservation...")
        if not results:
            return results
        
        unique_results = []
        seen_embeddings = []
        seen_capacities = set()
        
        # First pass: Extract and collect all capacities
        all_capacities = {}
        for res in results:
            text = res["text"]
            capacities = self.extract_project_capacity(text)
            if capacities:
                all_capacities[res["id"]] = capacities
        
        # Sort results by score to process highest scored first
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        # Second pass: Deduplicate with capacity diversity
        for res in sorted_results:
            text = res["text"]
            doc_id = res["id"]
            
            # Extract capacities from this result
            capacities = all_capacities.get(doc_id, [])
            
            # New capacity check
            new_capacity_found = False
            for capacity in capacities:
                if capacity.lower() not in seen_capacities:
                    new_capacity_found = True
                    seen_capacities.add(capacity.lower())
            
            # If this has a new capacity, prioritize it
            if new_capacity_found:
                unique_results.append(res)
                seen_embeddings.append(self.embedding_model.encode([text])[0])
                continue
                
            # Otherwise apply normal deduplication
            embedding = self.embedding_model.encode([text])[0]
            is_unique = True
            for seen_emb in seen_embeddings:
                similarity = 1 - cosine(embedding, seen_emb)
                if similarity > similarity_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_results.append(res)
                seen_embeddings.append(embedding)
        
        logger.info(f"Deduplication with diversity preserved, reduced to {len(unique_results)} results.")
        logger.info(f"Found diverse capacities: {seen_capacities}")
        return unique_results

    def hybrid_retrieval(self, query: str, top_k: int = 100, dense_weight: float = 0.6) -> List[Dict]:
        """Combine dense and sparse retrieval with weighted scores."""
        logger.info("Performing hybrid retrieval...")
        dense_results = self.dense_retrieval(query, top_k)
        sparse_results = self.sparse_retrieval(query, top_k)

        # Normalize scores
        dense_scores = {res["id"]: res["score"] for res in dense_results}
        sparse_scores = {res["id"]: res["score"] for res in sparse_results}
        max_dense = max([res["score"] for res in dense_results], default=1.0)
        max_sparse = max([res["score"] for res in sparse_results], default=1.0)

        # Combine results
        combined_results = {}
        for res in dense_results + sparse_results:
            doc_id = res["id"]
            dense_score = dense_scores.get(doc_id, 0) / max_dense
            sparse_score = sparse_scores.get(doc_id, 0) / max_sparse
            combined_score = dense_weight * dense_score + (1 - dense_weight) * sparse_score
            combined_results[doc_id] = {
                "text": res["text"],
                "score": combined_score
            }

        # Sort by combined score
        sorted_results = [
            {"id": doc_id, "text": data["text"], "score": data["score"]}
            for doc_id, data in sorted(combined_results.items(), key=lambda x: x[1]["score"], reverse=True)
        ]

        # Deduplicate with diversity focus
        deduped_results = self.deduplicate_results(sorted_results)
        logger.info(f"Hybrid retrieval returned {len(deduped_results)} results after deduplication.")
        return deduped_results[:top_k]
    
    def rerank_with_diversity(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank retrieved results with diversity enhancement."""
        logger.info(f"Reranking {len(results)} results with diversity enhancement...")
        
        # Extract and collect all capacities
        for res in results:
            res["capacities"] = self.extract_project_capacity(res["text"])
        
        # First, rerank all results using cross-encoder
        sentence_pairs = [[query, res["text"]] for res in results]
        scores = self.cross_encoder.predict(sentence_pairs)
        for res, score in zip(results, scores):
            res["rerank_score"] = float(score)
        
        # Sort all by rerank score
        sorted_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Apply diversity-aware selection
        final_results = []
        seen_capacities = set()
        
        # First, add top results with unique capacities
        for res in sorted_results:
            if len(final_results) >= top_k:
                break
                
            new_capacity = False
            for capacity in res.get("capacities", []):
                if capacity.lower() not in seen_capacities:
                    new_capacity = True
                    seen_capacities.add(capacity.lower())
            
            if new_capacity or not res.get("capacities"):
                final_results.append(res)
        
        # Fill remaining slots with top-scored results not yet selected
        remaining_slots = top_k - len(final_results)
        if remaining_slots > 0:
            added_ids = {res["id"] for res in final_results}
            for res in sorted_results:
                if res["id"] not in added_ids and remaining_slots > 0:
                    final_results.append(res)
                    remaining_slots -= 1
        
        # Final sort by score
        final_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Clean up temporary fields
        for res in final_results:
            if "capacities" in res:
                del res["capacities"]
                
        logger.info(f"Diversity-aware reranking complete, returning {len(final_results)} results.")
        logger.info(f"Found diverse capacities: {seen_capacities}")
        return final_results

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Main retrieval function: hybrid retrieval + deduplication + diversity-aware reranking."""
        logger.info(f"Retrieving for query: {query}")
        hybrid_results = self.hybrid_retrieval(query, top_k=100)
        # Use the diversity-aware reranking
        ranked_results = self.rerank_with_diversity(query, hybrid_results, top_k=top_k)
        logger.info(f"Retrieval complete, returned {len(ranked_results)} results.")
        return ranked_results

if __name__ == "__main__":
    # Example usage
    import os
    from preprocessing import DocumentPreprocessor

    # Initialize preprocessor to get chunks for BM25
    logger.info("Starting retriever test...")
    preprocessor = DocumentPreprocessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    # Load chunks from Pinecone
    logger.info("Fetching chunks from Pinecone...")
    index = preprocessor.pinecone_client.Index("rag-index")
    query_results = index.query(vector=[0]*384, top_k=1000, include_metadata=True)["matches"]
    chunks = [match["metadata"]["text"] for match in query_results]
    logger.info(f"Fetched {len(chunks)} chunks from Pinecone.")

    # Initialize retriever
    retriever = Retriever(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    retriever.initialize_bm25(chunks)

    # Test retrieval
    query = "What is the capacity of the Omkareshwar Dam Floating Solar Power Project?"
    logger.info("Running retrieval test...")
    results = retriever.retrieve(query, top_k=10)  # Changed to return 10 results
    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Text: {res['text'][:100]}...")
        print(f"Score: {res['rerank_score']}\n")