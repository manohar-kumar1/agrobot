"""
Vector store management for document embeddings.
Handles ChromaDB operations for storing and retrieving document vectors.
Manages two separate collections: citrus_diseases and government_schemes.
"""

from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.models.config import settings
from app.services.embedding_service import EmbeddingService


class VectorStore:
    """Manages vector database operations with two collections"""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        """
        Initialize vector store with ChromaDB.

        Args:
            embedding_service: Optional embedding service instance
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.persist_directory = Path(settings.chroma_persist_directory)

        # Collection names
        self.diseases_collection = settings.diseases_collection_name
        self.schemes_collection = settings.schemes_collection_name

        # ChromaDB client and vector stores
        self._client = None
        self._diseases_store = None
        self._schemes_store = None

    def _get_client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client"""
        if self._client is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def initialize(self):
        """Initialize ChromaDB connection and collections"""
        client = self._get_client()
        embeddings = self.embedding_service.get_langchain_embeddings()

        # Initialize diseases collection
        self._diseases_store = Chroma(
            client=client,
            collection_name=self.diseases_collection,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

        # Initialize schemes collection
        self._schemes_store = Chroma(
            client=client,
            collection_name=self.schemes_collection,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

    def get_store(self, collection_type: str) -> Chroma:
        """
        Get the appropriate vector store by collection type.

        Args:
            collection_type: 'disease' or 'scheme'

        Returns:
            Chroma vector store instance
        """
        if self._diseases_store is None or self._schemes_store is None:
            self.initialize()

        if collection_type == "disease":
            return self._diseases_store
        elif collection_type == "scheme":
            return self._schemes_store
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")

    def add_documents(
        self, documents: List[Document], collection_type: str
    ) -> List[str]:
        """
        Add documents to the specified collection.

        Args:
            documents: List of LangChain Document objects
            collection_type: 'disease' or 'scheme'

        Returns:
            List of document IDs
        """
        store = self.get_store(collection_type)

        # Extract IDs from metadata or generate them
        ids = [
            doc.metadata.get("chunk_id", f"{collection_type}_{i}")
            for i, doc in enumerate(documents)
        ]

        store.add_documents(documents=documents, ids=ids)
        return ids

    def similarity_search(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
        filter_metadata: Dict[str, Any] | None = None,
    ) -> List[Document]:
        """
        Search for similar documents in a collection.

        Args:
            query: Search query text
            collection_type: 'disease' or 'scheme'
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching Document objects
        """
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results

        if filter_metadata:
            results = store.similarity_search(query=query, k=k, filter=filter_metadata)
        else:
            results = store.similarity_search(query=query, k=k)

        return results

    def similarity_search_with_scores(
        self, query: str, collection_type: str, top_k: int | None = None
    ) -> List[tuple[Document, float]]:
        """
        Search with relevance scores.

        Args:
            query: Search query text
            collection_type: 'disease' or 'scheme'
            top_k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results

        return store.similarity_search_with_score(query=query, k=k)

    def similarity_search_with_threshold(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
        min_score: float | None = None
    ) -> List[tuple[Document, float]]:
        """
        Search with relevance scores, filtering by minimum threshold.

        Args:
            query: Search query text
            collection_type: 'disease' or 'scheme'
            top_k: Number of results to return
            min_score: Minimum relevance score (0-1)

        Returns:
            List of (Document, score) tuples above threshold
        """
        results = self.similarity_search_with_scores(query, collection_type, top_k)
        
        if min_score is None:
            min_score = settings.min_relevance_score
        
        # Filter by score (convert distance to similarity: 1 - distance)
        filtered = [
            (doc, score) for doc, score in results
            if (1 - score) >= min_score
        ]
        
        return filtered

    def mmr_search(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search for diverse results.

        Args:
            query: Search query text
            collection_type: 'disease' or 'scheme'
            top_k: Number of diverse results to return
            fetch_k: Number of candidates to fetch before MMR
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance)

        Returns:
            List of diverse Document objects
        """
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results

        return store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

    def search_both_collections(
        self, query: str, top_k: int | None = None
    ) -> Dict[str, List[Document]]:
        """
        Search both collections for hybrid queries.

        Args:
            query: Search query text
            top_k: Number of results per collection

        Returns:
            Dict with 'diseases' and 'schemes' results
        """
        return {
            "diseases": self.similarity_search(query, "disease", top_k),
            "schemes": self.similarity_search(query, "scheme", top_k),
        }

    def delete_collection(self, collection_type: str):
        """
        Delete a collection.

        Args:
            collection_type: 'disease' or 'scheme'
        """
        client = self._get_client()
        collection_name = (
            self.diseases_collection
            if collection_type == "disease"
            else self.schemes_collection
        )

        try:
            client.delete_collection(collection_name)
        except ValueError:
            pass  # Collection doesn't exist

        # Reset store reference
        if collection_type == "disease":
            self._diseases_store = None
        else:
            self._schemes_store = None

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get document counts for both collections.

        Returns:
            Dict with collection names and document counts
        """
        client = self._get_client()
        stats = {}

        try:
            diseases_col = client.get_collection(self.diseases_collection)
            stats[self.diseases_collection] = diseases_col.count()
        except ValueError:
            stats[self.diseases_collection] = 0

        try:
            schemes_col = client.get_collection(self.schemes_collection)
            stats[self.schemes_collection] = schemes_col.count()
        except ValueError:
            stats[self.schemes_collection] = 0

        return stats

    def get_retriever(self, collection_type: str, top_k: int | None = None):
        """
        Get a LangChain retriever for use in chains.

        Args:
            collection_type: 'disease' or 'scheme'
            top_k: Number of results to retrieve

        Returns:
            LangChain retriever object
        """
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results
        return store.as_retriever(search_kwargs={"k": k})
