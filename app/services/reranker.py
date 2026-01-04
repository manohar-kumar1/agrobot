"""
Re-ranking service for document relevance optimization.
Uses LLM-based scoring and Reciprocal Rank Fusion for hybrid queries.
"""

from typing import List, Dict, Any, Tuple
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.models.config import settings


class Reranker:
    """Service for re-ranking retrieved documents by relevance"""

    def __init__(self, llm: ChatMistralAI | None = None):
        """
        Initialize reranker with LLM.

        Args:
            llm: Optional ChatMistralAI instance (uses default if not provided)
        """
        self._llm = llm

    @property
    def llm(self) -> ChatMistralAI:
        """Lazy initialization of Mistral LLM for reranking"""
        if self._llm is None:
            self._llm = ChatMistralAI(
                model=settings.mistral_model,
                api_key=settings.mistral_api_key,
                temperature=0.0  # Deterministic scoring
            )
        return self._llm

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using LLM-based relevance scoring.

        Args:
            query: User's question
            documents: List of retrieved documents with content and metadata
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents with updated scores
        """
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k

        # Score each document
        scored_docs = []
        for doc in documents:
            score = self._score_document(query, doc)
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)

        # Sort by rerank score descending
        scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Return top_k documents
        return scored_docs[:top_k]

    def _score_document(self, query: str, document: Dict[str, Any]) -> float:
        """
        Score a single document's relevance to the query.

        Args:
            query: User's question
            document: Document with content

        Returns:
            Relevance score between 0.0 and 1.0
        """
        content = document.get("content", "")[:1000]  # Limit content length

        scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance scoring expert. Score how relevant this document passage is to answering the user's question.

Scoring criteria:
- 0.0-0.3: Not relevant or only tangentially related
- 0.4-0.6: Somewhat relevant, contains related information
- 0.7-0.8: Highly relevant, directly addresses the question
- 0.9-1.0: Perfectly relevant, contains the exact answer

Respond with ONLY a single decimal number between 0.0 and 1.0."""),
            ("user", """Question: {query}

Document passage:
{content}

Relevance score:""")
        ])

        try:
            chain = scoring_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query, "content": content})

            # Parse score
            score = float(result.strip())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1

        except (ValueError, Exception):
            # Return original similarity score if LLM scoring fails
            return document.get("score", 0.5)

    def filter_by_threshold(
        self,
        documents: List[Dict[str, Any]],
        threshold: float | None = None
    ) -> List[Dict[str, Any]]:
        """
        Filter documents below minimum relevance threshold.

        Args:
            documents: List of documents with scores
            threshold: Minimum score threshold

        Returns:
            Filtered list of documents
        """
        threshold = threshold or settings.min_relevance_score
        return [
            doc for doc in documents
            if doc.get("score", 0) >= threshold or doc.get("rerank_score", 0) >= threshold
        ]

    def reciprocal_rank_fusion(
        self,
        disease_docs: List[Dict[str, Any]],
        scheme_docs: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results from multiple collections using Reciprocal Rank Fusion.

        Args:
            disease_docs: Documents from disease collection
            scheme_docs: Documents from scheme collection
            k: RRF parameter (default 60)

        Returns:
            Combined and re-ranked document list
        """
        # Build RRF scores
        rrf_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # Process disease docs
        for rank, doc in enumerate(disease_docs):
            doc_id = f"disease_{doc.get('page', 0)}_{hash(doc.get('content', '')[:100])}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_copy = doc.copy()
            doc_copy["collection"] = "disease"
            rrf_scores[doc_id] = (rrf_score, doc_copy)

        # Process scheme docs
        for rank, doc in enumerate(scheme_docs):
            doc_id = f"scheme_{doc.get('page', 0)}_{hash(doc.get('content', '')[:100])}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_copy = doc.copy()
            doc_copy["collection"] = "scheme"

            if doc_id in rrf_scores:
                # Combine scores if document appears in both
                existing_score, existing_doc = rrf_scores[doc_id]
                rrf_scores[doc_id] = (existing_score + rrf_score, existing_doc)
            else:
                rrf_scores[doc_id] = (rrf_score, doc_copy)

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )

        # Add RRF score to documents
        result = []
        for rrf_score, doc in sorted_docs:
            doc["rrf_score"] = rrf_score
            result.append(doc)

        return result
