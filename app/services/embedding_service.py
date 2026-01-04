from typing import List
from langchain_mistralai import MistralAIEmbeddings

from app.models.config import settings


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
        return self.embeddings.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        return 1024

    def get_langchain_embeddings(self) -> MistralAIEmbeddings:
        return self.embeddings
