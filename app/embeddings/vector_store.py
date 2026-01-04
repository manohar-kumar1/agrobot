from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.models.config import settings
from app.services.embedding_service import EmbeddingService


class VectorStore:

    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.persist_directory = Path(settings.chroma_persist_directory)
        self.diseases_collection = settings.diseases_collection_name
        self.schemes_collection = settings.schemes_collection_name
        self._client = None
        self._diseases_store = None
        self._schemes_store = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def initialize(self):
        client = self._get_client()
        embeddings = self.embedding_service.get_langchain_embeddings()

        self._diseases_store = Chroma(
            client=client,
            collection_name=self.diseases_collection,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

        self._schemes_store = Chroma(
            client=client,
            collection_name=self.schemes_collection,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

    def get_store(self, collection_type: str) -> Chroma:
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
        store = self.get_store(collection_type)
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
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results

        return store.similarity_search_with_score(query=query, k=k)

    def similarity_search_with_threshold(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> List[tuple[Document, float]]:
        results = self.similarity_search_with_scores(query, collection_type, top_k)
        if min_score is None:
            min_score = settings.min_relevance_score
        filtered = [(doc, score) for doc, score in results if (1 - score) >= min_score]
        return filtered

    def mmr_search(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results

        return store.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def search_both_collections(
        self, query: str, top_k: int | None = None
    ) -> Dict[str, List[Document]]:
        return {
            "diseases": self.similarity_search(query, "disease", top_k),
            "schemes": self.similarity_search(query, "scheme", top_k),
        }

    def delete_collection(self, collection_type: str):
        client = self._get_client()
        collection_name = (
            self.diseases_collection
            if collection_type == "disease"
            else self.schemes_collection
        )

        try:
            client.delete_collection(collection_name)
        except ValueError:
            pass
        if collection_type == "disease":
            self._diseases_store = None
        else:
            self._schemes_store = None

    def get_collection_stats(self) -> Dict[str, int]:
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
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results
        return store.as_retriever(search_kwargs={"k": k})
