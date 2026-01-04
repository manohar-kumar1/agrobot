from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agrobot API"
    app_version: str = "1.0.0"
    debug: bool = False

    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"  # Faster than mistral-large
    mistral_embedding_model: str = "mistral-embed"
    mistral_temperature: float = 0.3  # Lower temperature for faster, more focused responses

    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "agrobot-hackathon"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    chroma_persist_directory: str = "./data/vector_store"
    diseases_collection_name: str = "citrus_diseases"
    schemes_collection_name: str = "government_schemes"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 3  # Reduced from 5 for speed

    min_relevance_score: float = 0.3
    enable_reranking: bool = False  # Disabled - LLM reranking is slow, use vector similarity instead
    rerank_top_k: int = 3

    pdf_directory: str = "./data/pdfs"
    diseases_pdf: str = "CitrusPlantPestsAndDiseases.pdf"
    schemes_pdf: str = "GovernmentSchemes.pdf"

    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


settings = Settings()
