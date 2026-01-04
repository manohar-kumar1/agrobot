"""
Configuration management using Pydantic settings.
Loads configuration from environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    app_name: str = "Agrobot API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Mistral Settings
    mistral_api_key: str = ""
    mistral_model: str = "mistral-large-latest"
    mistral_embedding_model: str = "mistral-embed"
    mistral_temperature: float = 0.7

    # LangSmith Settings (for tracing and debugging)
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "agrobot-hackathon"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Vector Store Settings
    chroma_persist_directory: str = "./data/vector_store"

    # Collection names for two knowledge bases
    diseases_collection_name: str = "citrus_diseases"
    schemes_collection_name: str = "government_schemes"

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    # RAG Enhancement Settings (Phase 3)
    min_relevance_score: float = 0.3  # Minimum score to keep documents
    enable_reranking: bool = True  # Enable LLM-based reranking
    rerank_top_k: int = 3  # Final number of docs after reranking

    # PDF paths
    pdf_directory: str = "./data/pdfs"
    diseases_pdf: str = "CitrusPlantPestsAndDiseases.pdf"
    schemes_pdf: str = "GovernmentSchemes.pdf"

    # API Settings
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


# Global settings instance
settings = Settings()


def configure_langsmith():
    """
    Configure LangSmith tracing environment variables.
    Call this at application startup to enable tracing.
    """
    import os
    
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        return True
    return False
