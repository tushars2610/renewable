import os
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np

class DocumentPreprocessor:
    def __init__(self, pinecone_api_key, pinecone_environment, index_name="rag-index"):
        """Initialize the preprocessor with Pinecone and embedding model."""
        load_dotenv()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pinecone_client = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
        self.index_name = index_name
        self._initialize_pinecone_index()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        self.max_metadata_size = 40000  # Pinecone metadata size limit per vector (in bytes)
        self.batch_size = 100  # Number of vectors to upsert in one batch

    def _initialize_pinecone_index(self):
        """Initialize or connect to a Pinecone index."""
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pinecone_client.Index(self.index_name)

    def extract_text(self, file_path):
        """Extract text from PDF, DOCX, or TXT files."""
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join([page.extract_text() or "" for page in reader.pages])
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        else:
            raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
        return text

    def chunk_text(self, text):
        """Chunk text into semantically meaningful parts."""
        chunks = self.text_splitter.split_text(text)
        return chunks

    def generate_embeddings(self, chunks):
        """Generate vector embeddings for text chunks."""
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        return embeddings

    def _check_metadata_size(self, chunk, file_name):
        """Check if metadata size is within Pinecone limits and truncate if necessary."""
        metadata = {"text": chunk, "source": file_name}
        metadata_size = len(str(metadata).encode('utf-8'))
        if metadata_size > self.max_metadata_size:
            # Truncate the text field to fit within limits
            max_text_size = self.max_metadata_size - len(str({"source": file_name}).encode('utf-8')) - 100  # Buffer
            chunk = chunk.encode('utf-8')[:max_text_size].decode('utf-8', errors='ignore')
            metadata["text"] = chunk
        return metadata

    def store_in_pinecone(self, chunks, embeddings, file_name):
        """Store embeddings and metadata in Pinecone in batches."""
        vectors = [
            {
                "id": f"{file_name}_chunk_{i}",
                "values": embedding.tolist(),
                "metadata": self._check_metadata_size(chunk, file_name)
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        # Upsert in batches to avoid exceeding Pinecone's message size limit
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch {i//self.batch_size + 1} for {file_name}: {str(e)}")
                return False
        return True

    def process_document(self, file_path):
        """Process a document: extract, chunk, embed, and store."""
        file_name = os.path.basename(file_path)
        try:
            text = self.extract_text(file_path)
            if not text.strip():
                print(f"Warning: No text extracted from {file_name}")
                return 0
            chunks = self.chunk_text(text)
            if not chunks:
                print(f"Warning: No chunks created from {file_name}")
                return 0
            embeddings = self.generate_embeddings(chunks)
            if self.store_in_pinecone(chunks, embeddings, file_name):
                return len(chunks)
            return 0
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            return 0

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    # Process all documents in the 'docs' folder
    docs_folder = "docs"
    total_chunks = 0

    if not os.path.exists(docs_folder):
        print(f"Directory '{docs_folder}' does not exist.")
    else:
        for file_name in os.listdir(docs_folder):
            file_path = os.path.join(docs_folder, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
                print(f"Processing {file_name}...")
                chunks_processed = preprocessor.process_document(file_path)
                total_chunks += chunks_processed
                print(f"Processed {chunks_processed} chunks from {file_name}")
        
        print(f"Total chunks processed from all documents: {total_chunks}")