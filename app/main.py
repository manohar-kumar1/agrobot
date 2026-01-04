"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.routes import router as v1_router
from app.api.v1.endpoints.query import router as query_router
from app.models.config import settings, configure_langsmith
from app.core.logging_config import setup_logging, get_logger


# Setup logging
setup_logging(log_level="INFO" if not settings.debug else "DEBUG")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting up Agrobot API...")

    # Configure LangSmith tracing (commented out)
    # if configure_langsmith():
    #     logger.info(f"LangSmith tracing enabled - Project: {settings.langchain_project}")
    # else:
    #     logger.info("LangSmith tracing not configured (LANGCHAIN_API_KEY not set)")

    # Pre-initialize the RAG agent for faster first query
    from app.api.v1.endpoints.query import get_rag_agent

    try:
        agent = get_rag_agent()
        logger.info("RAG agent pre-initialized successfully")
    except Exception as e:
        logger.warning(f"RAG agent pre-initialization failed: {e}")

    yield
    logger.info("Shutting down Agrobot API...")


# Create FastAPI app
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(v1_router, prefix=settings.api_v1_prefix, tags=["v1"])
app.include_router(query_router, prefix=settings.api_v1_prefix, tags=["query"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Agrobot API",
        "version": settings.app_version,
        "docs": "/docs",
        "description": "Agriculture Chatbot for Farmers - Disease & Scheme Information",
    }


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.app_version}
