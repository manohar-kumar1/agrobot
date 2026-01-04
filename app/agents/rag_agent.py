from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END

from app.models.schemas import IntentType, QueryResponse
from app.agents.intent_classifier import IntentClassifier
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.reranker import Reranker
from app.embeddings.vector_store import VectorStore
from app.models.config import settings


class QueryState(TypedDict):
    query: str
    intent: str
    confidence: float
    reason: str
    disease_docs: List[Dict[str, Any]]
    scheme_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    response: str
    sources: List[Dict[str, Any]]
    error: str | None
    edge_case: str | None


class RAGAgent:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_service: LLMService | None = None,
        intent_classifier: IntentClassifier | None = None,
    ):
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store or VectorStore(
            embedding_service=self.embedding_service
        )
        self.vector_store.initialize()

        self.llm_service = llm_service or LLMService()
        self.intent_classifier = intent_classifier or IntentClassifier(
            llm_service=self.llm_service
        )
        self.reranker = Reranker()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(QueryState)
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("retrieve_disease", self._retrieve_disease_node)
        workflow.add_node("retrieve_scheme", self._retrieve_scheme_node)
        workflow.add_node("retrieve_hybrid", self._retrieve_hybrid_node)
        workflow.add_node("rerank_documents", self._rerank_documents_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "disease": "retrieve_disease",
                "scheme": "retrieve_scheme",
                "hybrid": "retrieve_hybrid",
            },
        )
        workflow.add_edge("retrieve_disease", "rerank_documents")
        workflow.add_edge("retrieve_scheme", "rerank_documents")
        workflow.add_edge("retrieve_hybrid", "rerank_documents")
        workflow.add_edge("rerank_documents", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _classify_intent_node(self, state: QueryState) -> QueryState:
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
        intent = state.get("intent", "unknown")

        if intent == "disease":
            return "disease"
        elif intent == "scheme":
            return "scheme"
        else:
            return "hybrid"

    def _retrieve_disease_node(self, state: QueryState) -> QueryState:
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
                        "score": 1 - score,
                    }
                )

            state["disease_docs"] = disease_docs
            state["scheme_docs"] = []

        except Exception as e:
            state["disease_docs"] = []
            state["error"] = f"Disease retrieval error: {str(e)}"

        return state

    def _retrieve_scheme_node(self, state: QueryState) -> QueryState:
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
                        "score": 1 - score,
                    }
                )

            state["disease_docs"] = []
            state["scheme_docs"] = scheme_docs

        except Exception as e:
            state["scheme_docs"] = []
            state["error"] = f"Scheme retrieval error: {str(e)}"

        return state

    def _retrieve_hybrid_node(self, state: QueryState) -> QueryState:
        query = state["query"]
        top_k = settings.top_k_results

        try:
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
        query = state["query"]
        intent = state["intent"]
        disease_docs = state.get("disease_docs", [])
        scheme_docs = state.get("scheme_docs", [])
        all_docs = disease_docs + scheme_docs

        if not all_docs:
            state["reranked_docs"] = []
            state["edge_case"] = "no_documents"
            return state

        try:
            if settings.enable_reranking:
                if intent == "hybrid":
                    fused_docs = self.reranker.reciprocal_rank_fusion(
                        disease_docs, scheme_docs
                    )
                    reranked = self.reranker.rerank_documents(
                        query, fused_docs, settings.rerank_top_k * 2
                    )
                else:
                    reranked = self.reranker.rerank_documents(
                        query, all_docs, settings.rerank_top_k
                    )
                reranked = self.reranker.filter_by_threshold(reranked)
                if not reranked:
                    state["edge_case"] = "low_relevance"
                    reranked = all_docs[: settings.rerank_top_k]

                state["reranked_docs"] = reranked
            else:
                state["reranked_docs"] = all_docs[: settings.rerank_top_k]

        except Exception as e:
            state["reranked_docs"] = all_docs[: settings.rerank_top_k]
            state["error"] = f"Reranking error: {str(e)}"

        return state

    def _generate_response_node(self, state: QueryState) -> QueryState:
        query = state["query"]
        intent = state["intent"]
        edge_case = state.get("edge_case")
        confidence = state.get("confidence", 0.0)
        docs_to_use = state.get("reranked_docs", [])
        if confidence < 0.4 and not docs_to_use:
            state["response"] = self.llm_service.generate_ambiguous_query_response(
                query
            )
            state["sources"] = []
            state["edge_case"] = "ambiguous_query"
            return state
        if not docs_to_use or edge_case == "no_documents":
            state["response"] = (
                "I couldn't find relevant information to answer your question. "
                "Please try rephrasing your query or ask about citrus diseases or government agricultural schemes."
            )
            state["sources"] = []
            return state

        try:
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

    def _get_route_to(self, intent: str) -> str:
        """Get the route_to description based on intent."""
        route_map = {
            "disease": "Citrus Pests & Diseases Knowledge Base",
            "scheme": "Government Schemes Knowledge Base",
            "hybrid": "BOTH Knowledge Bases",
            "unknown": "Unknown",
        }
        return route_map.get(intent, "Unknown")

    def process_query(self, query: str) -> QueryResponse:
        initial_state: QueryState = {
            "query": query,
            "intent": "unknown",
            "confidence": 0.0,
            "reason": "",
            "disease_docs": [],
            "scheme_docs": [],
            "reranked_docs": [],
            "response": "",
            "sources": [],
            "error": None,
            "edge_case": None,
        }
        final_state = self.workflow.invoke(initial_state)
        intent_str = final_state.get("intent", "unknown")
        intent_map = {
            "disease": IntentType.DISEASE,
            "scheme": IntentType.SCHEME,
            "hybrid": IntentType.HYBRID,
            "unknown": IntentType.UNKNOWN,
        }
        intent = intent_map.get(intent_str, IntentType.UNKNOWN)

        return QueryResponse(
            answer=final_state.get("response", "No response generated"),
            intent=intent,
            route_to=self._get_route_to(intent_str),
        )
