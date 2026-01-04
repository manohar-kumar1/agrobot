from typing import List, Tuple
from cachetools import LRUCache
import hashlib
from langchain_mistralai import MistralAIEmbeddings

from app.models.config import settings


_embedding_cache: LRUCache[str, List[float]] = LRUCache(maxsize=2000)
_cache_hits: int = 0
_cache_misses: int = 0


def _generate_cache_key(text: str) -> str:
    normalized = text.strip().lower()
    if len(normalized) > 100:
        return hashlib.md5(normalized.encode()).hexdigest()
    return normalized


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.mistral_embedding_model
        self._embeddings = None

    @property
    def embeddings(self) -> MistralAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = MistralAIEmbeddings(
                model=self.model_name, api_key=settings.mistral_api_key
            )
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        global _cache_hits, _cache_misses
        cache_key = _generate_cache_key(text)

        if cache_key in _embedding_cache:
            _cache_hits += 1
            return _embedding_cache[cache_key]

        _cache_misses += 1
        embedding = self.embeddings.embed_query(text)

        _embedding_cache[cache_key] = embedding

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        global _cache_hits, _cache_misses
        results: List[List[float] | None] = [None] * len(texts)
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        for i, text in enumerate(texts):
            cache_key = _generate_cache_key(text)
            if cache_key in _embedding_cache:
                results[i] = _embedding_cache[cache_key]
                _cache_hits += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                _cache_misses += 1

        if texts_to_embed:
            new_embeddings = self.embeddings.embed_documents(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                results[idx] = embedding
                cache_key = _generate_cache_key(texts[idx])
                _embedding_cache[cache_key] = embedding

        return results

    def embed_with_metadata(self, text: str) -> Tuple[List[float], bool]:
        cache_key = _generate_cache_key(text)
        was_cached = cache_key in _embedding_cache
        embedding = self.embed_text(text)
        return embedding, was_cached

    def get_embedding_dimension(self) -> int:
        return 1024

    def get_langchain_embeddings(self) -> MistralAIEmbeddings:
        return self.embeddings

    @staticmethod
    def clear_cache() -> None:
        global _cache_hits, _cache_misses
        _embedding_cache.clear()
        _cache_hits = 0
        _cache_misses = 0

    @staticmethod
    def get_cache_stats() -> dict:
        total_requests = _cache_hits + _cache_misses
        hit_rate = _cache_hits / max(total_requests, 1)

        return {
            "cached_queries": len(_embedding_cache),
            "max_cache_size": _embedding_cache.maxsize,
            "cache_hits": _cache_hits,
            "cache_misses": _cache_misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": max(
                0, (_cache_hits + _cache_misses) - len(_embedding_cache) - _cache_misses
            ),
        }

    @staticmethod
    def warm_cache(texts: List[str]) -> int:
        service = EmbeddingService()
        service.embed_batch(texts)
        return len(texts)
