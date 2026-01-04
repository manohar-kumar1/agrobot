

from typing import TypedDict, List, Dict, Any, Literal, Set
from langgraph.graph import StateGraph, END

from app.models.schemas import IntentType, QueryResponse
from app.agents.intent_classifier import IntentClassifier
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.reranker import Reranker
from app.embeddings.vector_store import VectorStore
from app.models.config import settings
from app.utils.knowledge_graph import get_knowledge_graph, AgriculturalKnowledgeGraph
from app.utils.bloom_filter import DocumentDeduplicator


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
    kg_context: Dict[str, Any] | None


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

        self.knowledge_graph: AgriculturalKnowledgeGraph = get_knowledge_graph()

        self.deduplicator = DocumentDeduplicator(expected_docs=5000)

        self._disease_keywords: Set[str] = {
            "canker",
            "hlb",
            "greening",
            "tristeza",
            "scab",
            "melanose",
            "gummosis",
            "rot",
            "anthracnose",
            "psyllid",
            "mite",
            "aphid",
            "whitefly",
            "disease",
            "pest",
            "symptom",
            "treatment",
            "fungicide",
        }
        self._scheme_keywords: Set[str] = {
            "scheme",
            "yojana",
            "subsidy",
            "grant",
            "loan",
            "pmksy",
            "nhm",
            "pmfby",
            "kisan",
            "credit",
            "insurance",
            "government",
            "benefit",
        }

        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(QueryState)
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("handle_greeting", self._handle_greeting_node)
        workflow.add_node("handle_out_of_scope", self._handle_out_of_scope_node)
        workflow.add_node("retrieve_disease", self._retrieve_disease_node)
        workflow.add_node("retrieve_scheme", self._retrieve_scheme_node)
        workflow.add_node("retrieve_hybrid", self._retrieve_hybrid_node)
        workflow.add_node("enrich_with_kg", self._enrich_with_knowledge_graph_node)
        workflow.add_node("rerank_documents", self._rerank_documents_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "greeting": "handle_greeting",
                "out_of_scope": "handle_out_of_scope",
                "disease": "retrieve_disease",
                "scheme": "retrieve_scheme",
                "hybrid": "retrieve_hybrid",
            },
        )
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("handle_out_of_scope", END)
        workflow.add_edge("retrieve_disease", "enrich_with_kg")
        workflow.add_edge("retrieve_scheme", "enrich_with_kg")
        workflow.add_edge("retrieve_hybrid", "enrich_with_kg")
        workflow.add_edge("enrich_with_kg", "rerank_documents")
        workflow.add_edge("rerank_documents", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _enrich_with_knowledge_graph_node(self, state: QueryState) -> QueryState:
        query = state["query"].lower()
        intent = state["intent"]

        try:
            kg_context: Dict[str, Any] = {
                "related_entities": [],
                "treatments": [],
                "schemes": [],
            }

            query_words = set(query.split())

            entity_mapping = {
                "canker": "citrus_canker",
                "hlb": "hlb",
                "greening": "hlb",
                "tristeza": "tristeza",
                "scab": "scab",
                "melanose": "melanose",
                "gummosis": "gummosis",
                "rot": "root_rot",
                "anthracnose": "anthracnose",
                "yellowing": "yellowing",
                "psyllid": "asian_citrus_psyllid",
                "leafminer": "citrus_leafminer",
                "aphid": "aphids",
            }

            matched_entities: Set[str] = set()
            for word in query_words:
                if word in entity_mapping:
                    matched_entities.add(entity_mapping[word])

            for entity_id in matched_entities:
                if self.knowledge_graph.get_node(entity_id):
                    if intent in ("disease", "hybrid"):
                        context = self.knowledge_graph.get_disease_context(entity_id)
                        if context:
                            kg_context["related_entities"].append(context)
                            kg_context["treatments"].extend(
                                context.get("treatments", [])
                            )

                    if intent in ("scheme", "hybrid"):
                        schemes = self.knowledge_graph.find_schemes_for_disease(
                            entity_id
                        )
                        kg_context["schemes"].extend(schemes)

            seen_treatments: Set[str] = set()
            unique_treatments = []
            for t in kg_context["treatments"]:
                if t["id"] not in seen_treatments:
                    seen_treatments.add(t["id"])
                    unique_treatments.append(t)
            kg_context["treatments"] = unique_treatments

            state["kg_context"] = kg_context

        except Exception as e:
            state["kg_context"] = None

        return state

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
    ) -> Literal["greeting", "out_of_scope", "disease", "scheme", "hybrid"]:
        intent = state.get("intent", "unknown")

        if intent == "greeting":
            return "greeting"
        elif intent == "out_of_scope":
            return "out_of_scope"
        elif intent == "disease":
            return "disease"
        elif intent == "scheme":
            return "scheme"
        else:
            return "hybrid"

    def _handle_greeting_node(self, state: QueryState) -> QueryState:
        query = state["query"].lower().strip()

        if any(word in query for word in ["thank", "thanks", "dhanyavaad", "shukriya"]):
            response = (
                "You're welcome! I'm always here to help you with your citrus farming needs. "
                "Feel free to ask me about: 1) Citrus diseases and pest management, "
                "2) Government agricultural schemes and subsidies, "
                "3) Treatment recommendations for your crops. Have a great day!"
            )
        elif any(word in query for word in ["bye", "goodbye", "alvida", "see you"]):
            response = (
                "Goodbye! Wishing you a bountiful harvest. Remember, I'm here 24/7 to help with: "
                "1) Disease identification and treatment, 2) Government scheme information, "
                "3) Pest control advice. Take care of your crops!"
            )
        elif any(
            word in query
            for word in ["who are you", "what are you", "your name", "about you"]
        ):
            response = (
                "I am AgroBot, your AI-powered agricultural assistant specializing in citrus farming! "
                "I can help you with: 1) Identifying and treating citrus diseases like Citrus Canker, "
                "HLB (Citrus Greening), and pest infestations, 2) Information about government schemes "
                "like PMKSY, NHM, PM-Kisan, and agricultural subsidies, 3) Integrated advice combining "
                "disease management with available government support. Just ask me your question!"
            )
        elif any(
            word in query for word in ["help", "what can you do", "how to use", "guide"]
        ):
            response = (
                "I'm here to assist you with citrus farming! Here's what you can ask me: "
                "1) Disease queries - 'What are the symptoms of citrus canker?' or 'How to treat leaf curl?', "
                "2) Scheme queries - 'What subsidies are available for drip irrigation?' or 'How to apply for PM-Kisan?', "
                "3) Combined queries - 'Is there government help for HLB management?'. "
                "Try describing your crop problem or asking about any agricultural scheme!"
            )
        elif any(
            word in query
            for word in ["how are you", "how're you", "kaise ho", "kaisa hai"]
        ):
            response = (
                "I'm doing great, thank you for asking! I'm ready to help you with your farming queries. "
                "You can ask me about: 1) Any citrus disease or pest problem you're facing, "
                "2) Government schemes and subsidies for farmers, 3) Treatment recommendations. "
                "What would you like to know today?"
            )
        else:
            response = (
                "Hello! Welcome to AgroBot - your citrus farming assistant! "
                "I can help you with: 1) Citrus disease identification and treatment (e.g., 'My orange leaves have yellow spots'), "
                "2) Government agricultural schemes and subsidies (e.g., 'What is PMKSY subsidy?'), "
                "3) Pest management advice (e.g., 'How to control aphids on citrus?'). "
                "How can I assist you today?"
            )

        state["response"] = response
        state["sources"] = []
        return state

    def _handle_out_of_scope_node(self, state: QueryState) -> QueryState:
        query = state["query"]

        response = (
            f"I appreciate your question, but I specialize specifically in citrus farming assistance. "
            "I can help you with: 1) Citrus crop diseases - symptoms, identification, and treatment "
            "(e.g., Citrus Canker, HLB, leaf miners, aphids), 2) Government agricultural schemes - "
            "subsidies, loans, and benefits for farmers (e.g., PMKSY, NHM, PM-Kisan, KCC), "
            "3) Integrated support combining disease management with government assistance. "
            "Please ask me something related to citrus farming or agricultural schemes, and I'll be happy to help!"
        )

        state["response"] = response
        state["sources"] = []
        state["edge_case"] = "out_of_scope"
        return state

    def _retrieve_disease_node(self, state: QueryState) -> QueryState:
        query = state["query"]
        top_k = settings.top_k_results

        try:
            results = self.vector_store.smart_search(
                query=query, collection_type="disease", top_k=top_k
            )

            disease_docs = []
            for doc, score in results:
                doc_dict = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "score": score,
                }
                if self.deduplicator.add_document(
                    doc.page_content,
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                ):
                    disease_docs.append(doc_dict)

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
            results = self.vector_store.smart_search(
                query=query, collection_type="scheme", top_k=top_k
            )

            scheme_docs = []
            for doc, score in results:
                doc_dict = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "score": score,
                }
                if self.deduplicator.add_document(
                    doc.page_content,
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                ):
                    scheme_docs.append(doc_dict)

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
            self.deduplicator.clear()

            disease_results = self.vector_store.smart_search(
                query=query, collection_type="disease", top_k=top_k
            )

            disease_docs = []
            for doc, score in disease_results:
                doc_dict = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "score": score,
                }
                if self.deduplicator.add_document(
                    doc.page_content,
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                ):
                    disease_docs.append(doc_dict)

            scheme_results = self.vector_store.smart_search(
                query=query, collection_type="scheme", top_k=top_k
            )

            scheme_docs = []
            for doc, score in scheme_results:
                doc_dict = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "score": score,
                }
                if self.deduplicator.add_document(
                    doc.page_content,
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                ):
                    scheme_docs.append(doc_dict)

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
        kg_context = state.get("kg_context")
        all_docs = disease_docs + scheme_docs

        if not all_docs:
            state["reranked_docs"] = []
            state["edge_case"] = "no_documents"
            return state

        try:
            if settings.enable_reranking:
                if intent == "hybrid":
                    fused_docs = self.reranker.reciprocal_rank_fusion(
                        disease_docs, scheme_docs, top_n=settings.rerank_top_k * 2
                    )
                    reranked = self.reranker.rerank_documents(
                        query, fused_docs, settings.rerank_top_k * 2
                    )
                else:
                    reranked = self.reranker.rerank_documents(
                        query, all_docs, settings.rerank_top_k
                    )

                reranked = self.reranker.deduplicate_documents(reranked)
                reranked = self.reranker.filter_by_threshold(reranked)

                if not reranked:
                    state["edge_case"] = "low_relevance"
                    reranked = all_docs[: settings.rerank_top_k]

                state["reranked_docs"] = reranked
            else:
                unique_docs = self.reranker.deduplicate_documents(all_docs)
                state["reranked_docs"] = unique_docs[: settings.rerank_top_k]

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

    def process_query(self, query: str) -> QueryResponse:
        self.deduplicator.clear()

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
            "kg_context": None,
        }
        final_state = self.workflow.invoke(initial_state)
        intent_str = final_state.get("intent", "unknown")
        intent_map = {
            "disease": IntentType.DISEASE,
            "scheme": IntentType.SCHEME,
            "hybrid": IntentType.HYBRID,
            "greeting": IntentType.GREETING,
            "out_of_scope": IntentType.OUT_OF_SCOPE,
            "unknown": IntentType.UNKNOWN,
        }
        intent = intent_map.get(intent_str, IntentType.UNKNOWN)

        return QueryResponse(
            answer=final_state.get("response", "No response generated"),
            intent=intent,
        )

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        return self.knowledge_graph.get_stats()

    def get_deduplicator_stats(self) -> Dict[str, Any]:
        return self.deduplicator.get_stats()
