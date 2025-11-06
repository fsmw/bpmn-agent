"""
Vector Embedding and Similarity Search for Knowledge Base

Provides semantic similarity search capabilities for BPMN patterns and examples
using sentence-transformers for embeddings and simple in-memory search.
For production use, consider integrating ChromaDB or FAISS.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """Result from a vector similarity search."""
    
    item_id: str = Field(..., description="ID of the matched item")
    item_type: str = Field(..., description="Type: 'pattern' or 'example'")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0-1)"
    )
    item: Optional[Dict[str, Any]] = Field(
        None,
        description="The actual item (pattern or example)"
    )


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses sentence-transformers for semantic embeddings.
    Falls back to simple TF-IDF-like embeddings if transformers unavailable.
    """
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize the embedding service.
        
        Args:
            use_transformers: If True, try to use sentence-transformers.
                            If False or unavailable, use simple keyword-based embeddings.
        """
        self.use_transformers = use_transformers
        self.model = None
        self.embedding_dim = None
        
        if use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                # Using a small, fast model
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dim = 384
            except ImportError:
                # Fallback to simple embeddings
                self.use_transformers = False
                self.embedding_dim = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array representing the embedding
        """
        if not text or not isinstance(text, str):
            text = str(text) if text else ""
        
        if self.use_transformers and self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            # Simple keyword-based embedding fallback
            return self._simple_embed(text)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """
        Simple keyword-based embedding as fallback.
        
        This creates a basic embedding based on keywords and is much
        less effective than semantic embeddings, but requires no dependencies.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding array
        """
        # Normalize text
        text_lower = text.lower()
        
        # Define keyword groups for different process concepts
        keywords = {
            "activity": ["task", "activity", "activity", "do", "perform", "execute"],
            "gateway": ["decision", "gateway", "choice", "split", "join", "fork"],
            "event": ["event", "trigger", "start", "end", "occur", "happen"],
            "actor": ["actor", "role", "user", "participant", "person", "team"],
            "data": ["data", "object", "input", "output", "document", "file"],
            "parallelism": ["parallel", "concurrent", "simultaneous", "together"],
            "sequence": ["then", "after", "follow", "next", "sequence"],
            "condition": ["if", "condition", "conditional", "based", "when"],
        }
        
        # Create a simple embedding based on keyword presence
        embedding = np.zeros(len(keywords) * 3, dtype=np.float32)
        
        for idx, (concept, kws) in enumerate(keywords.items()):
            base_idx = idx * 3
            count = sum(1 for kw in kws if kw in text_lower)
            embedding[base_idx] = count / len(kws)
            embedding[base_idx + 1] = len(text_lower) / 1000.0  # Text length factor
            embedding[base_idx + 2] = concept.count(text_lower)  # Concept in text
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class VectorStore:
    """
    Vector storage and similarity search engine for patterns and examples.
    
    Stores embeddings for patterns and examples, enabling semantic search.
    For production, consider replacing with ChromaDB or FAISS.
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_service: Service for generating embeddings.
                              If None, creates a new instance.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Storage for embeddings and metadata
        self.pattern_embeddings: Dict[str, np.ndarray] = {}
        self.pattern_metadata: Dict[str, Dict[str, Any]] = {}
        
        self.example_embeddings: Dict[str, np.ndarray] = {}
        self.example_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_pattern(
        self,
        pattern_id: str,
        name: str,
        description: str,
        examples: List[str],
        tags: List[str],
        domain: str = "generic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a pattern to the vector store.
        
        Args:
            pattern_id: Unique pattern identifier
            name: Pattern name
            description: Pattern description
            examples: Example texts where this pattern appears
            tags: Searchable tags
            domain: Domain category
            metadata: Additional metadata
        """
        # Combine text for embedding
        combined_text = f"{name} {description} {' '.join(examples)} {' '.join(tags)}"
        embedding = self.embedding_service.embed_text(combined_text)
        
        self.pattern_embeddings[pattern_id] = embedding
        self.pattern_metadata[pattern_id] = {
            "name": name,
            "description": description,
            "domain": domain,
            "tags": tags,
            "examples": examples,
            **(metadata or {})
        }
    
    def add_example(
        self,
        example_id: str,
        text: str,
        domain: str = "generic",
        difficulty: str = "medium",
        patterns: List[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an example to the vector store.
        
        Args:
            example_id: Unique example identifier
            text: Example text
            domain: Domain category
            difficulty: Difficulty level
            patterns: List of pattern IDs related to this example
            metadata: Additional metadata
        """
        embedding = self.embedding_service.embed_text(text)
        
        self.example_embeddings[example_id] = embedding
        self.example_metadata[example_id] = {
            "text": text,
            "domain": domain,
            "difficulty": difficulty,
            "patterns": patterns or [],
            **(metadata or {})
        }
    
    def search_patterns(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        min_similarity: float = 0.3
    ) -> List[VectorSearchResult]:
        """
        Search for similar patterns.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            domain_filter: Optional domain to filter by
            min_similarity: Minimum similarity score to include
            
        Returns:
            List of search results sorted by similarity
        """
        if not self.pattern_embeddings:
            return []
        
        query_embedding = self.embedding_service.embed_text(query)
        results = []
        
        for pattern_id, embedding in self.pattern_embeddings.items():
            metadata = self.pattern_metadata.get(pattern_id, {})
            
            # Apply domain filter if specified
            if domain_filter and metadata.get("domain") != domain_filter:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= min_similarity:
                results.append(
                    VectorSearchResult(
                        item_id=pattern_id,
                        item_type="pattern",
                        similarity_score=float(similarity),
                        item=metadata
                    )
                )
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def search_examples(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        min_similarity: float = 0.3
    ) -> List[VectorSearchResult]:
        """
        Search for similar examples.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            domain_filter: Optional domain to filter by
            min_similarity: Minimum similarity score to include
            
        Returns:
            List of search results sorted by similarity
        """
        if not self.example_embeddings:
            return []
        
        query_embedding = self.embedding_service.embed_text(query)
        results = []
        
        for example_id, embedding in self.example_embeddings.items():
            metadata = self.example_metadata.get(example_id, {})
            
            # Apply domain filter if specified
            if domain_filter and metadata.get("domain") != domain_filter:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= min_similarity:
                results.append(
                    VectorSearchResult(
                        item_id=example_id,
                        item_type="example",
                        similarity_score=float(similarity),
                        item=metadata
                    )
                )
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def search_combined(
        self,
        query: str,
        top_k_patterns: int = 3,
        top_k_examples: int = 2,
        domain_filter: Optional[str] = None
    ) -> Tuple[List[VectorSearchResult], List[VectorSearchResult]]:
        """
        Search for both patterns and examples.
        
        Args:
            query: Search query
            top_k_patterns: Number of patterns to return
            top_k_examples: Number of examples to return
            domain_filter: Optional domain filter
            
        Returns:
            Tuple of (pattern results, example results)
        """
        patterns = self.search_patterns(
            query,
            top_k=top_k_patterns,
            domain_filter=domain_filter
        )
        examples = self.search_examples(
            query,
            top_k=top_k_examples,
            domain_filter=domain_filter
        )
        return patterns, examples
    
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern metadata or None if not found
        """
        return self.pattern_metadata.get(pattern_id)
    
    def get_example(self, example_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an example by ID.
        
        Args:
            example_id: Example ID
            
        Returns:
            Example metadata or None if not found
        """
        return self.example_metadata.get(example_id)
    
    def get_patterns_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all patterns for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of pattern metadata
        """
        return [
            metadata
            for metadata in self.pattern_metadata.values()
            if metadata.get("domain") == domain
        ]
    
    def get_examples_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all examples for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of example metadata
        """
        return [
            metadata
            for metadata in self.example_metadata.values()
            if metadata.get("domain") == domain
        ]
    
    def clear(self) -> None:
        """Clear all stored embeddings and metadata."""
        self.pattern_embeddings.clear()
        self.pattern_metadata.clear()
        self.example_embeddings.clear()
        self.example_metadata.clear()
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        # Handle different sized vectors by padding or slicing
        min_size = min(len(vec1), len(vec2))
        vec1_normalized = vec1[:min_size]
        vec2_normalized = vec2[:min_size]
        
        norm1 = np.linalg.norm(vec1_normalized)
        norm2 = np.linalg.norm(vec2_normalized)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1_normalized, vec2_normalized) / (norm1 * norm2))
