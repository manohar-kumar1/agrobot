from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agrobot API"
    app_version: str = "1.0.0"
    debug: bool = False

    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    mistral_embedding_model: str = "mistral-embed"
    mistral_temperature: float = 0.2

    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "agrobot-hackathon"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    chroma_persist_directory: str = "./data/vector_store"
    diseases_collection_name: str = "citrus_diseases"
    schemes_collection_name: str = "government_schemes"

    chunk_size: int = 1500
    chunk_overlap: int = 300
    top_k_results: int = 5

    min_relevance_score: float = 0.25
    enable_reranking: bool = True
    rerank_top_k: int = 4

    enable_hybrid_search: bool = True
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7

    pdf_directory: str = "./data/pdfs"
    diseases_pdf: str = "CitrusPlantPestsAndDiseases.pdf"
    schemes_pdf: str = "GovernmentSchemes.pdf"

    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


settings = Settings()
