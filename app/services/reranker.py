from typing import List, Dict, Any, Tuple
import re

from app.models.config import settings


class Reranker:
    """Fast score-based reranker without LLM calls."""

    def __init__(self):
        # Keywords for boosting relevance
        self.disease_keywords = [
            "disease", "pest", "symptom", "treatment", "canker", "greening",
            "hlb", "psyllid", "whitefly", "leafminer", "fungus", "bacteria",
            "infection", "spray", "insecticide", "fungicide", "copper",
            "yellowing", "lesion", "rot", "mite", "aphid", "scale"
        ]
        self.scheme_keywords = [
            "scheme", "subsidy", "government", "pmksy", "nhm", "rkvy",
            "mission", "yojana", "assistance", "financial", "loan", "credit",
            "nabard", "kcc", "insurance", "pmfby", "grant", "support"
        ]

    def rerank_documents(
        self, query: str, documents: List[Dict[str, Any]], top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """Fast reranking using keyword matching and vector scores."""
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        scored_docs = []
        for doc in documents:
            score = self._compute_relevance_score(query_lower, query_words, doc)
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return scored_docs[:top_k]

    def _compute_relevance_score(
        self, query_lower: str, query_words: set, document: Dict[str, Any]
    ) -> float:
        """Compute relevance score using multiple signals."""
        content = document.get("content", "").lower()
        content_words = set(re.findall(r'\w+', content[:500]))  # First 500 chars
        
        # Base score from vector similarity
        base_score = document.get("score", 0.5)
        
        # Word overlap boost
        overlap = len(query_words & content_words)
        overlap_boost = min(overlap * 0.05, 0.2)  # Max 0.2 boost
        
        # Exact phrase match boost
        phrase_boost = 0.15 if query_lower[:20] in content else 0.0
        
        # Keyword relevance boost
        keyword_boost = 0.0
        for word in query_words:
            if word in self.disease_keywords or word in self.scheme_keywords:
                if word in content:
                    keyword_boost += 0.05
        keyword_boost = min(keyword_boost, 0.15)  # Cap at 0.15
        
        final_score = base_score + overlap_boost + phrase_boost + keyword_boost
        return min(final_score, 1.0)

    def filter_by_threshold(
        self, documents: List[Dict[str, Any]], threshold: float | None = None
    ) -> List[Dict[str, Any]]:
        threshold = threshold or settings.min_relevance_score
        return [
            doc
            for doc in documents
            if doc.get("score", 0) >= threshold
            or doc.get("rerank_score", 0) >= threshold
        ]

    def reciprocal_rank_fusion(
        self,
        disease_docs: List[Dict[str, Any]],
        scheme_docs: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for rank, doc in enumerate(disease_docs):
            doc_id = (
                f"disease_{doc.get('page', 0)}_{hash(doc.get('content', '')[:100])}"
            )
            rrf_score = 1.0 / (k + rank + 1)
            doc_copy = doc.copy()
            doc_copy["collection"] = "disease"
            rrf_scores[doc_id] = (rrf_score, doc_copy)

        for rank, doc in enumerate(scheme_docs):
            doc_id = f"scheme_{doc.get('page', 0)}_{hash(doc.get('content', '')[:100])}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_copy = doc.copy()
            doc_copy["collection"] = "scheme"

            if doc_id in rrf_scores:
                existing_score, existing_doc = rrf_scores[doc_id]
                rrf_scores[doc_id] = (existing_score + rrf_score, existing_doc)
            else:
                rrf_scores[doc_id] = (rrf_score, doc_copy)

        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)

        result = []
        for rrf_score, doc in sorted_docs:
            doc["rrf_score"] = rrf_score
            result.append(doc)

        return result
