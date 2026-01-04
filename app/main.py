from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.endpoints.query import router as query_router
from app.models.config import settings
from app.core.logging_config import setup_logging, get_logger


setup_logging(log_level="INFO" if not settings.debug else "DEBUG")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Agrobot API...")
    from app.api.v1.endpoints.query import get_rag_agent

    try:
        agent = get_rag_agent()
        logger.info("RAG agent pre-initialized successfully")
    except Exception as e:
        logger.warning(f"RAG agent pre-initialization failed: {e}")

    yield
    logger.info("Shutting down Agrobot API...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
# Agriculture Chatbot API for Farmers

An intelligent RAG-based system that helps farmers with:

## 🌱 Disease Information
- Citrus crop diseases, symptoms, and identification
- Treatment and prevention methods
- Pest management strategies

## 📋 Government Schemes
- Agricultural subsidies and financial assistance
- Eligibility criteria and application processes
- Available benefits and programs

## 🔀 Hybrid Queries
- Queries combining disease management with government support
- Financial help for pest control and treatment

The system uses **LangGraph** for agentic workflow orchestration and **Mistral AI** for intelligent responses.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix=settings.api_v1_prefix, tags=["query"])
