from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re
from collections import Counter
import math
import heapq
import pickle

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.models.config import settings
from app.services.embedding_service import EmbeddingService


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.corpus_size: int = 0
        self.documents: List[List[str]] = []
        self._vocabulary: set = set()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: List[str]) -> "BM25":
        self.documents = [self._tokenize(doc) for doc in documents]
        self.corpus_size = len(self.documents)
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.corpus_size, 1)

        self.doc_freqs = {}
        self._vocabulary = set()
        for doc in self.documents:
            unique_terms = set(doc)
            self._vocabulary.update(unique_terms)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        return self

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        query_terms = self._tokenize(query)
        doc = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        term_freqs = Counter(doc)
        score = 0.0

        for term in query_terms:
            if term in term_freqs:
                tf = term_freqs[term]
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score += idf * numerator / denominator

        return score

    def get_scores(self, query: str) -> List[float]:
        return [self.score(query, i) for i in range(self.corpus_size)]

    def get_top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        top_k_indices = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
        return [(idx, scores[idx]) for idx in top_k_indices]

    def save(self, filepath: Path) -> None:
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": self.doc_freqs,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "corpus_size": self.corpus_size,
            "documents": self.documents,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> Optional["BM25"]:
        if not filepath.exists():
            return None
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            bm25 = cls(k1=data["k1"], b=data["b"])
            bm25.doc_freqs = data["doc_freqs"]
            bm25.doc_lengths = data["doc_lengths"]
            bm25.avg_doc_length = data["avg_doc_length"]
            bm25.corpus_size = data["corpus_size"]
            bm25.documents = data["documents"]
            return bm25
        except Exception:
            return None


class TFIDFRetriever:
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents: List[str] = []
        self._fitted = False

    def fit(self, documents: List[str]) -> "TFIDFRetriever":
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self._fitted = True
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self._fitted:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = heapq.nlargest(
            top_k, range(len(similarities)), key=lambda i: similarities[i]
        )

        return [(idx, similarities[idx]) for idx in top_k_indices]

    def save(self, filepath: Path) -> None:
        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "documents": self.documents,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> Optional["TFIDFRetriever"]:
        if not filepath.exists():
            return None
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            retriever = cls()
            retriever.vectorizer = data["vectorizer"]
            retriever.tfidf_matrix = data["tfidf_matrix"]
            retriever.documents = data["documents"]
            retriever._fitted = True
            return retriever
        except Exception:
            return None


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

    def hybrid_search_with_scores(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
        bm25_weight: float | None = None,
        semantic_weight: float | None = None,
    ) -> List[tuple[Document, float]]:
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results
        bm25_w = bm25_weight if bm25_weight is not None else settings.bm25_weight
        semantic_w = (
            semantic_weight if semantic_weight is not None else settings.semantic_weight
        )

        fetch_k = min(k * 3, 20)

        semantic_results = store.similarity_search_with_score(query=query, k=fetch_k)

        if not semantic_results:
            return []

        docs = [doc for doc, _ in semantic_results]
        contents = [doc.page_content for doc in docs]

        bm25 = BM25()
        bm25.fit(contents)
        bm25_scores = bm25.get_scores(query)

        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
        normalized_bm25 = [s / max_bm25 for s in bm25_scores]

        semantic_scores = [1 - score for _, score in semantic_results]
        max_semantic = (
            max(semantic_scores) if semantic_scores and max(semantic_scores) > 0 else 1
        )
        normalized_semantic = [s / max_semantic for s in semantic_scores]

        combined_results = []
        for i, (doc, _) in enumerate(semantic_results):
            hybrid_score = (bm25_w * normalized_bm25[i]) + (
                semantic_w * normalized_semantic[i]
            )
            combined_results.append((doc, hybrid_score))
        top_k_results = heapq.nlargest(k, combined_results, key=lambda x: x[1])
        return top_k_results

    def smart_search(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        if settings.enable_hybrid_search:
            return self.hybrid_search_with_scores(query, collection_type, top_k)
        else:
            return self.similarity_search_with_scores(query, collection_type, top_k)

    def tfidf_search(
        self,
        query: str,
        collection_type: str,
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        store = self.get_store(collection_type)
        k = top_k or settings.top_k_results
        fetch_k = min(k * 5, 50)
        all_results = store.similarity_search_with_score(query=query, k=fetch_k)

        if not all_results:
            return []

        docs = [doc for doc, _ in all_results]
        contents = [doc.page_content for doc in docs]
        tfidf = TFIDFRetriever()
        tfidf.fit(contents)
        tfidf_results = tfidf.search(query, k)

        return [(docs[idx], score) for idx, score in tfidf_results]
