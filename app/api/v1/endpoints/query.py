from fastapi import APIRouter, HTTPException
from cachetools import TTLCache
import hashlib
import time

from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.agents.rag_agent import RAGAgent
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

_rag_agent: RAGAgent | None = None

_response_cache: TTLCache[str, QueryResponse] = TTLCache(maxsize=500, ttl=1800)
_cache_stats = {"hits": 0, "misses": 0}


def _generate_query_hash(question: str) -> str:
    normalized = question.strip().lower()
    normalized = " ".join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()


def get_rag_agent() -> RAGAgent:
    global _rag_agent
    if _rag_agent is None:
        logger.info("Initializing RAG agent...")
        _rag_agent = RAGAgent()
        logger.info("RAG agent initialized successfully")
    return _rag_agent


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        200: {"description": "Successful query response"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Process agricultural query",
    description="""
    Main query endpoint for the agriculture chatbot.
    
    Handles three types of queries:
    - **Disease**: Questions about citrus diseases, pests, symptoms, treatment
    - **Scheme**: Questions about government schemes, subsidies, eligibility
    
    The system automatically detects the query intent and routes to appropriate knowledge bases.
    """,
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    global _cache_stats
    start_time = time.perf_counter()

    logger.info(f"Received query: {request.question[:100]}...")

    try:
        cache_key = _generate_query_hash(request.question)
        if cache_key in _response_cache:
            _cache_stats["hits"] += 1
            cached_response = _response_cache[cache_key]
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"Cache HIT - Latency: {elapsed:.2f}ms")
            return cached_response

        _cache_stats["misses"] += 1

        agent = get_rag_agent()
        response = agent.process_query(request.question)
        _response_cache[cache_key] = response

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Query processed - Intent: {response.intent.value}, Latency: {elapsed:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
