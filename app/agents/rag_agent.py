"""
RAG (Retrieval-Augmented Generation) agent for processing queries.
Uses LangGraph StateGraph for orchestrating the intent-based routing workflow.
"""

from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END

from app.models.schemas import IntentType, QueryResponse, Source
from app.agents.intent_classifier import IntentClassifier
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.reranker import Reranker
from app.embeddings.vector_store import VectorStore
from app.models.config import settings


class QueryState(TypedDict):
    """State object passed through the LangGraph workflow"""

    query: str
    intent: str
    confidence: float
    reason: str
    disease_docs: List[Dict[str, Any]]
    scheme_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]  # Phase 3: Reranked documents
    response: str
    sources: List[Dict[str, Any]]
    error: str | None
    edge_case: str | None  # Phase 3: Edge case detection


class RAGAgent:
    """
    Main RAG agent using LangGraph StateGraph for query processing.

    Workflow:
    START → classify_intent → route_by_intent → [retrieve_disease | retrieve_scheme | retrieve_hybrid] → generate_response → END
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_service: LLMService | None = None,
        intent_classifier: IntentClassifier | None = None,
    ):
        """
        Initialize RAG agent with required services.

        Args:
            vector_store: Vector database instance
            llm_service: LLM service instance
            intent_classifier: Intent classifier instance
        """
        # Initialize services
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store or VectorStore(
            embedding_service=self.embedding_service
        )
        self.vector_store.initialize()

        self.llm_service = llm_service or LLMService()
        self.intent_classifier = intent_classifier or IntentClassifier(
            llm_service=self.llm_service
        )

        # Phase 3: Initialize reranker
        self.reranker = Reranker()

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph StateGraph workflow.

        Phase 3 Enhanced Workflow:
        START → classify_intent → route_by_intent → [retrieve_*] → rerank_documents → generate_response → END

        Returns:
            Compiled StateGraph workflow
        """
        # Create the graph
        workflow = StateGraph(QueryState)

        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("retrieve_disease", self._retrieve_disease_node)
        workflow.add_node("retrieve_scheme", self._retrieve_scheme_node)
        workflow.add_node("retrieve_hybrid", self._retrieve_hybrid_node)
        workflow.add_node("rerank_documents", self._rerank_documents_node)  # Phase 3
        workflow.add_node("generate_response", self._generate_response_node)

        # Set entry point
        workflow.set_entry_point("classify_intent")

        # Add conditional routing based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "disease": "retrieve_disease",
                "scheme": "retrieve_scheme",
                "hybrid": "retrieve_hybrid",
            },
        )

        # Phase 3: All retrieval nodes go to reranking first
        workflow.add_edge("retrieve_disease", "rerank_documents")
        workflow.add_edge("retrieve_scheme", "rerank_documents")
        workflow.add_edge("retrieve_hybrid", "rerank_documents")

        # Reranking goes to response generation
        workflow.add_edge("rerank_documents", "generate_response")

        # Generation goes to END
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _classify_intent_node(self, state: QueryState) -> QueryState:
        """
        Node: Classify the query intent.

        Args:
            state: Current workflow state

        Returns:
            Updated state with intent classification
        """
        query = state["query"]

        try:
            intent, confidence, reason = self.intent_classifier.classify(query)
            state["intent"] = intent.value
            state["confidence"] = confidence
            state["reason"] = reason
        except Exception as e:
            state["intent"] = "unknown"
            state["confidence"] = 0.0
            state["reason"] = f"Classification error: {str(e)}"
            state["error"] = str(e)

        return state

    def _route_by_intent(
        self, state: QueryState
    ) -> Literal["disease", "scheme", "hybrid"]:
        """
        Routing function: Determine which retrieval node to use.

        Args:
            state: Current workflow state

        Returns:
            Route name (disease, scheme, or hybrid)
        """
        intent = state.get("intent", "unknown")

        if intent == "disease":
            return "disease"
        elif intent == "scheme":
            return "scheme"
        else:
            # For unknown or hybrid, use hybrid retrieval
            return "hybrid"

    def _retrieve_disease_node(self, state: QueryState) -> QueryState:
        """
        Node: Retrieve documents from disease collection only.

        Args:
            state: Current workflow state

        Returns:
            Updated state with disease documents
        """
        query = state["query"]
        top_k = settings.top_k_results

        try:
            results = self.vector_store.similarity_search_with_scores(
                query=query, collection_type="disease", top_k=top_k
            )

            disease_docs = []
            for doc, score in results:
                disease_docs.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "Unknown"),
                        "page": doc.metadata.get("page"),
                        "score": 1 - score,  # Convert distance to similarity
                    }
                )

            state["disease_docs"] = disease_docs
            state["scheme_docs"] = []

        except Exception as e:
            state["disease_docs"] = []
            state["error"] = f"Disease retrieval error: {str(e)}"

        return state

    def _retrieve_scheme_node(self, state: QueryState) -> QueryState:
        """
        Node: Retrieve documents from scheme collection only.

        Args:
            state: Current workflow state

        Returns:
            Updated state with scheme documents
        """
        query = state["query"]
        top_k = settings.top_k_results

        try:
            results = self.vector_store.similarity_search_with_scores(
                query=query, collection_type="scheme", top_k=top_k
            )

            scheme_docs = []
            for doc, score in results:
                scheme_docs.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "Unknown"),
                        "page": doc.metadata.get("page"),
                        "score": 1 - score,  # Convert distance to similarity
                    }
                )

            state["disease_docs"] = []
            state["scheme_docs"] = scheme_docs

        except Exception as e:
            state["scheme_docs"] = []
            state["error"] = f"Scheme retrieval error: {str(e)}"

        return state

    def _retrieve_hybrid_node(self, state: QueryState) -> QueryState:
        """
        Node: Retrieve documents from BOTH collections (5 from each = 10 total).

        Args:
            state: Current workflow state

        Returns:
            Updated state with documents from both collections
        """
        query = state["query"]
        top_k = settings.top_k_results

        try:
            # Get disease documents with scores
            disease_results = self.vector_store.similarity_search_with_scores(
                query=query, collection_type="disease", top_k=top_k
            )

            disease_docs = []
            for doc, score in disease_results:
                disease_docs.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "Unknown"),
                        "page": doc.metadata.get("page"),
                        "score": 1 - score,
                    }
                )

            # Get scheme documents with scores
            scheme_results = self.vector_store.similarity_search_with_scores(
                query=query, collection_type="scheme", top_k=top_k
            )

            scheme_docs = []
            for doc, score in scheme_results:
                scheme_docs.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "Unknown"),
                        "page": doc.metadata.get("page"),
                        "score": 1 - score,
                    }
                )

            state["disease_docs"] = disease_docs
            state["scheme_docs"] = scheme_docs

        except Exception as e:
            state["disease_docs"] = []
            state["scheme_docs"] = []
            state["error"] = f"Hybrid retrieval error: {str(e)}"

        return state

    def _rerank_documents_node(self, state: QueryState) -> QueryState:
        """
        Node: Re-rank retrieved documents for relevance optimization.

        Phase 3 addition for improved retrieval efficacy.

        Args:
            state: Current workflow state

        Returns:
            Updated state with reranked documents
        """
        query = state["query"]
        intent = state["intent"]
        disease_docs = state.get("disease_docs", [])
        scheme_docs = state.get("scheme_docs", [])

        # Combine all documents
        all_docs = disease_docs + scheme_docs

        if not all_docs:
            state["reranked_docs"] = []
            state["edge_case"] = "no_documents"
            return state

        try:
            if settings.enable_reranking:
                # Apply reranking based on intent
                if intent == "hybrid":
                    # Use RRF for hybrid queries to balance both sources
                    fused_docs = self.reranker.reciprocal_rank_fusion(
                        disease_docs, scheme_docs
                    )
                    # Then rerank the fused results
                    reranked = self.reranker.rerank_documents(
                        query, fused_docs, settings.rerank_top_k * 2  # More for hybrid
                    )
                else:
                    # Standard reranking for single-source queries
                    reranked = self.reranker.rerank_documents(
                        query, all_docs, settings.rerank_top_k
                    )

                # Filter by threshold
                reranked = self.reranker.filter_by_threshold(reranked)

                # Check if all docs were filtered out
                if not reranked:
                    state["edge_case"] = "low_relevance"
                    # Keep top docs anyway but note the edge case
                    reranked = all_docs[: settings.rerank_top_k]

                state["reranked_docs"] = reranked
            else:
                # Reranking disabled - just use original docs
                state["reranked_docs"] = all_docs[: settings.rerank_top_k]

        except Exception as e:
            # Fallback to original docs on error
            state["reranked_docs"] = all_docs[: settings.rerank_top_k]
            state["error"] = f"Reranking error: {str(e)}"

        return state

    def _generate_response_node(self, state: QueryState) -> QueryState:
        """
        Node: Generate response using reranked documents.

        Phase 3: Now uses reranked_docs instead of raw retrieved docs.

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated response and sources
        """
        query = state["query"]
        intent = state["intent"]
        edge_case = state.get("edge_case")
        confidence = state.get("confidence", 0.0)

        # Phase 3: Use reranked docs
        docs_to_use = state.get("reranked_docs", [])

        # Handle edge case: ambiguous query (low confidence)
        if confidence < 0.4 and not docs_to_use:
            state["response"] = self.llm_service.generate_ambiguous_query_response(
                query
            )
            state["sources"] = []
            state["edge_case"] = "ambiguous_query"
            return state

        # Handle edge case: no relevant documents found
        if not docs_to_use or edge_case == "no_documents":
            state["response"] = (
                "I couldn't find relevant information to answer your question. "
                "Please try rephrasing your query or ask about citrus diseases or government agricultural schemes."
            )
            state["sources"] = []
            return state

        try:
            # Generate response using LLM with reranked documents
            result = self.llm_service.generate_with_citations(
                query=query, documents=docs_to_use, intent=intent
            )

            state["response"] = result["answer"]
            state["sources"] = result["citations"]

        except Exception as e:
            state["response"] = (
                f"An error occurred while generating the response: {str(e)}"
            )
            state["sources"] = []
            state["error"] = str(e)

        return state

    def process_query(self, query: str) -> QueryResponse:
        """
        Process a query through the full RAG pipeline.

        Args:
            query: User's question

        Returns:
            QueryResponse with answer, intent, sources, and confidence
        """
        # Initialize state
        initial_state: QueryState = {
            "query": query,
            "intent": "unknown",
            "confidence": 0.0,
            "reason": "",
            "disease_docs": [],
            "scheme_docs": [],
            "reranked_docs": [],  # Phase 3
            "response": "",
            "sources": [],
            "error": None,
            "edge_case": None,  # Phase 3
        }

        # Run the workflow
        final_state = self.workflow.invoke(initial_state)

        # Convert to QueryResponse
        sources = [
            Source(
                document=s.get("document", "Unknown"),
                page=s.get("page"),
                relevance_score=s.get("relevance_score"),
                excerpt=s.get("excerpt"),
            )
            for s in final_state.get("sources", [])
        ]

        # Map intent string to enum
        intent_map = {
            "disease": IntentType.DISEASE,
            "scheme": IntentType.SCHEME,
            "hybrid": IntentType.HYBRID,
            "unknown": IntentType.UNKNOWN,
        }
        intent = intent_map.get(
            final_state.get("intent", "unknown"), IntentType.UNKNOWN
        )

        return QueryResponse(
            answer=final_state.get("response", "No response generated"),
            intent=intent,
            sources=sources,
            confidence=final_state.get("confidence"),
        )

    def get_workflow_graph(self) -> str:
        """
        Get a string representation of the workflow for debugging.

        Returns:
            ASCII representation of the workflow graph
        """
        return """
        LangGraph RAG Workflow (Phase 3 Enhanced):
        
        ┌─────────────────────────┐
        │         START           │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │    classify_intent      │
        │  (LLM-based analysis)   │
        └───────────┬─────────────┘
                    │
            ┌───────┼───────┐
            │       │       │
            ▼       ▼       ▼
        ┌───────┐ ┌───────┐ ┌───────┐
        │disease│ │scheme │ │hybrid │
        │retriev│ │retriev│ │retriev│
        └───┬───┘ └───┬───┘ └───┬───┘
            │         │         │
            └─────────┴─────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │   rerank_documents      │
        │   (LLM scoring + RRF)   │  ← Phase 3
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   generate_response     │
        │   (RAG with LLM)        │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │          END            │
        └─────────────────────────┘
        """
