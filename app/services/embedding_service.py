"""
Embedding service for generating text embeddings.
Uses Mistral embeddings API via LangChain.
"""

from typing import List
from langchain_mistralai import MistralAIEmbeddings

from app.models.config import settings


class EmbeddingService:
    """Service for generating text embeddings using Mistral"""

    def __init__(self, model_name: str | None = None):
        """
        Initialize embedding service with Mistral.

        Args:
            model_name: Name of the embedding model (defaults to config value)
        """
        self.model_name = model_name or settings.mistral_embedding_model
        self._embeddings = None

    @property
    def embeddings(self) -> MistralAIEmbeddings:
        """Lazy initialization of embeddings model"""
        if self._embeddings is None:
            self._embeddings = MistralAIEmbeddings(
                model=self.model_name, api_key=settings.mistral_api_key
            )
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of Mistral embeddings.

        Returns:
            Embedding dimension size (1024 for mistral-embed)
        """
        return 1024

    def get_langchain_embeddings(self) -> MistralAIEmbeddings:
        """
        Get the LangChain embeddings object for use with vector stores.

        Returns:
            MistralAIEmbeddings instance
        """
        return self.embeddings
