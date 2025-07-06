import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
import os
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Complete vector store implementation with FAISS backend that handles:
    - Document storage and retrieval
    - Embedding generation
    - Persistence to disk
    - Error handling
    - Metadata management
    """

    def __init__(self, persist_directory: str = "faiss_db"):
        """
        Initialize the vector store with proper embedding dimension.
        
        Args:
            persist_directory: Directory to store the FAISS index and metadata
        """
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2 embeddings
        self.persist_directory = persist_directory
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load the FAISS index and metadata with enhanced error handling"""
        os.makedirs(self.persist_directory, exist_ok=True)
    
        index_path = os.path.join(self.persist_directory, "index.faiss")
        metadata_path = os.path.join(self.persist_directory, "metadata.json")
    
        try:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load existing index with validation
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Add these validation checks:
                if not isinstance(self.metadata, list):
                    logger.error("Invalid metadata format - recreating store")
                    self._create_new_index()
                    return
                
                if self.index.ntotal != len(self.metadata):
                    logger.error("Index/metadata count mismatch - recreating store")
                    self._create_new_index()
                    return
                
                if self.index.d != self.embedding_dim:
                    logger.error("Dimension mismatch - recreating store")
                    self._create_new_index()
                    return
                
                logger.info(f"Loaded existing store with {len(self.metadata)} documents")
            else:
                self._create_new_index()
        except Exception as e:
            logger.error(f"Error loading store: {str(e)} - creating new store")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index with the correct dimension"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        logger.info("Created new FAISS index")

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Add documents to the vector store with proper chunking and embedding.
        
        Args:
            documents: List of documents as dictionaries with 'text' and 'metadata'
            batch_size: Number of documents to process at once
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        try:
            # Process in batches to avoid memory issues
            for i in tqdm(range(0, len(documents), batch_size)):
                batch = documents[i:i + batch_size]
                
                # Extract texts and metadata
                texts = [doc['text'] for doc in batch]
                metadatas = [doc.get('metadata', {}) for doc in batch]
                
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(texts)
                embeddings = np.array(embeddings, dtype=np.float32)
                
                # Verify dimensionality
                if embeddings.shape[1] != self.embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                        f"got {embeddings.shape[1]}"
                    )
                
                # Add to index
                self.index.add(embeddings)
                
                # Store metadata with reference to the embedding
                for text, metadata in zip(texts, metadatas):
                    self.metadata.append({
                        'text': text,
                        'metadata': metadata
                        })
                
            # Save after each batch
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents based on a query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of documents with their metadata and similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Retrieve results with metadata
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['score'] = float(1 - distance)  # Convert to similarity score
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def get_relevant_context(self, query: str, k: int = 3, max_context_length: int = 3000) -> str:
        """
        Get optimized relevant context for a query with length management.
        
        Args:
            query: The search query
            k: Number of similar documents to consider
            max_context_length: Maximum character length for returned context
            
        Returns:
            Combined and prioritized relevant context from top matches
        """
        try:
            # Get similarity search results
            results = self.similarity_search(query, k=k)
        
            if not results:
                return "No relevant context found."
        
            # Prioritize results by score (highest first)
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
            # Build context intelligently
            context_parts = []
            current_length = 0
        
            for result in sorted_results:
                text = result['text']
                if current_length + len(text) > max_context_length:
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if there's meaningful space left
                        # Add the most relevant part of this result
                        context_parts.append(text[:remaining_space])
                    break
                context_parts.append(text)
                current_length += len(text)
        
            # Join with separators that help the model understand context breaks
            return "CONTEXT BREAK\n\n".join(context_parts)
    
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return "Error retrieving context. Please try again."

    def _save_index(self):
        """Save the current index and metadata to disk"""
        try:
            index_path = os.path.join(self.persist_directory, "index.faiss")
            metadata_path = os.path.join(self.persist_directory, "metadata.json")
            
            faiss.write_index(self.index, index_path)
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Saved index with {len(self.metadata)} documents")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def clear(self):
        """Clear all documents from the vector store"""
        self._create_new_index()
        self._save_index()
        logger.info("Cleared vector store")

    def document_count(self) -> int:
        """Return the number of documents in the store"""
        return len(self.metadata)

    def get_all_documents(self, with_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from the store
        
        Args:
            with_embeddings: Whether to include embeddings in the output
            
        Returns:
            List of all documents with their metadata
        """
        if with_embeddings:
            # This requires reconstructing embeddings from the index
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            return [
                {**doc, 'embedding': embeddings[i].tolist()}
                for i, doc in enumerate(self.metadata)
            ]
        return self.metadata.copy()