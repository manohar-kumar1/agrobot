from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.agents.rag_agent import RAGAgent
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

_rag_agent: RAGAgent | None = None


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
    - **Hybrid**: Questions combining disease management with government support
    
    The system automatically detects the query intent and routes to appropriate knowledge bases.
    """,
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    logger.info(f"Received query: {request.question[:100]}...")

    try:
        agent = get_rag_agent()

        response = agent.process_query(request.question)

        logger.info(
            f"Query processed - Intent: {response.intent.value}"
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
