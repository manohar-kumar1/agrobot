from typing import Annotated
from fastapi import Depends

from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.embeddings.vector_store import VectorStore
from app.agents.rag_agent import RAGAgent
from app.agents.intent_classifier import IntentClassifier
from app.models.config import settings


_llm_service: LLMService | None = None
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStore | None = None
_rag_agent: RAGAgent | None = None
_intent_classifier: IntentClassifier | None = None


async def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(
            model_name=settings.openai_model, temperature=settings.openai_temperature
        )
    return _llm_service


async def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            model_name=settings.openai_embedding_model
        )
    return _embedding_service


async def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        await _vector_store.initialize()
    return _vector_store


async def get_rag_agent(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> RAGAgent:
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent(vector_store=vector_store, llm=llm_service)
    return _rag_agent


async def get_intent_classifier() -> IntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
