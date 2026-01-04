from typing import List, Dict, Any, Tuple, Set
import re
import heapq

from app.models.config import settings
from app.utils.trie import KeywordMatcher, get_keyword_matcher


class Reranker:
    def __init__(self):
        # Original keyword lists for Trie initialization
        self.disease_keywords = [
            "disease",
            "pest",
            "symptom",
            "treatment",
            "infection",
            "control",
            "prevention",
            "management",
            "spray",
            "application",
            "damage",
            "canker",
            "greening",
            "hlb",
            "tristeza",
            "scab",
            "melanose",
            "gummosis",
            "root rot",
            "foot rot",
            "phytophthora",
            "anthracnose",
            "sooty mold",
            "citrus variegated chlorosis",
            "cvc",
            "exocortis",
            "psyllid",
            "whitefly",
            "leafminer",
            "mite",
            "aphid",
            "scale",
            "thrips",
            "fruit fly",
            "borer",
            "nematode",
            "mealybug",
            "spider mite",
            "asian citrus psyllid",
            "acp",
            "citrus leaf miner",
            "rust mite",
            "yellowing",
            "lesion",
            "rot",
            "wilting",
            "dieback",
            "chlorosis",
            "necrosis",
            "leaf curl",
            "fruit drop",
            "gall",
            "spots",
            "blotch",
            "fungicide",
            "insecticide",
            "pesticide",
            "copper",
            "neem",
            "bordeaux",
            "carbendazim",
            "imidacloprid",
            "thiamethoxam",
            "biological control",
            "ipm",
            "integrated pest management",
        ]

        self.scheme_keywords = [
            "scheme",
            "yojana",
            "mission",
            "programme",
            "program",
            "pmksy",
            "nhm",
            "rkvy",
            "nfsm",
            "pkvy",
            "pmfby",
            "kcc",
            "pradhan mantri",
            "pm kisan",
            "pm-kisan",
            "kisan credit",
            "national horticulture",
            "rashtriya krishi",
            "paramparagat",
            "soil health card",
            "shc",
            "nabard",
            "apeda",
            "subsidy",
            "grant",
            "loan",
            "credit",
            "insurance",
            "assistance",
            "financial",
            "benefit",
            "support",
            "compensation",
            "relief",
            "interest",
            "subvention",
            "reimbursement",
            "government",
            "central",
            "state",
            "district",
            "eligible",
            "eligibility",
            "application",
            "registration",
            "enrollment",
            "portal",
            "online",
            "office",
            "officer",
            "department",
            "farmer",
            "kisan",
            "agriculture",
            "horticulture",
            "irrigation",
            "drip",
            "micro irrigation",
            "sprinkler",
            "organic",
            "certification",
        ]

        # Initialize Trie-based keyword matcher
        self._keyword_matcher = get_keyword_matcher()
        if not self._keyword_matcher.is_initialized:
            self._keyword_matcher.initialize(
                self.disease_keywords, self.scheme_keywords
            )
        self._disease_keywords_set: Set[str] = set(
            kw.lower() for kw in self.disease_keywords
        )
        self._scheme_keywords_set: Set[str] = set(
            kw.lower() for kw in self.scheme_keywords
        )

    def rerank_documents(
        self, query: str, documents: List[Dict[str, Any]], top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k
        query_lower = query.lower()
        query_words: Set[str] = set(re.findall(r"\w+", query_lower))

        scored_docs = []
        for doc in documents:
            score = self._compute_relevance_score(query_lower, query_words, doc)
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            scored_docs.append(doc_copy)
        top_docs = heapq.nlargest(
            top_k, scored_docs, key=lambda x: x.get("rerank_score", 0)
        )
        return top_docs

    def _compute_relevance_score(
        self, query_lower: str, query_words: Set[str], document: Dict[str, Any]
    ) -> float:
        content = document.get("content", "").lower()
        content_words: Set[str] = set(re.findall(r"\w+", content))

        base_score = document.get("score", 0.5)
        overlap = len(query_words & content_words)
        overlap_boost = min(overlap * 0.04, 0.25)

        phrase_boost = 0.0
        query_phrases = [
            query_lower[i : i + 30] for i in range(0, min(len(query_lower), 60), 15)
        ]
        for phrase in query_phrases:
            if len(phrase) > 10 and phrase in content:
                phrase_boost = 0.15
                break
        keyword_boost = 0.0
        matched_keywords = 0
        query_disease_matches = self._keyword_matcher.match_disease_keywords(
            query_lower
        )
        query_scheme_matches = self._keyword_matcher.match_scheme_keywords(query_lower)
        content_disease_matches = self._keyword_matcher.match_disease_keywords(content)
        content_scheme_matches = self._keyword_matcher.match_scheme_keywords(content)
        disease_overlap = query_disease_matches & content_disease_matches
        scheme_overlap = query_scheme_matches & content_scheme_matches

        matched_keywords = len(disease_overlap) + len(scheme_overlap)
        keyword_boost = min(matched_keywords * 0.03, 0.2)

        if matched_keywords >= 3:
            keyword_boost += 0.05

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
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        seen_hashes: Set[str] = set()

        for rank, doc in enumerate(disease_docs):
            content_hash = hash(doc.get("content", "")[:100])
            doc_id = f"disease_{doc.get('page', 0)}_{content_hash}"
            if content_hash in seen_hashes:
                # Merge scores for duplicates
                if doc_id in rrf_scores:
                    existing_score, existing_doc = rrf_scores[doc_id]
                    rrf_scores[doc_id] = (
                        existing_score + 1.0 / (k + rank + 1),
                        existing_doc,
                    )
                continue
            seen_hashes.add(content_hash)
            rrf_score = 1.0 / (k + rank + 1)
            doc_copy = doc.copy()
            doc_copy["collection"] = "disease"
            rrf_scores[doc_id] = (rrf_score, doc_copy)

        for rank, doc in enumerate(scheme_docs):
            content_hash = hash(doc.get("content", "")[:100])
            doc_id = f"scheme_{doc.get('page', 0)}_{content_hash}"
            rrf_score = 1.0 / (k + rank + 1)
            doc_copy = doc.copy()
            doc_copy["collection"] = "scheme"

            if doc_id in rrf_scores:
                existing_score, existing_doc = rrf_scores[doc_id]
                rrf_scores[doc_id] = (existing_score + rrf_score, existing_doc)
            else:
                rrf_scores[doc_id] = (rrf_score, doc_copy)
        items = list(rrf_scores.values())
        if top_n:
            sorted_items = heapq.nlargest(top_n, items, key=lambda x: x[0])
        else:
            sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

        result = []
        for rrf_score, doc in sorted_items:
            doc["rrf_score"] = rrf_score
            result.append(doc)

        return result

    def deduplicate_documents(
        self,
        documents: List[Dict[str, Any]],
        content_key: str = "content",
        prefix_len: int = 100,
    ) -> List[Dict[str, Any]]:
        seen_hashes: Set[int] = set()
        unique_docs: List[Dict[str, Any]] = []

        for doc in documents:
            content = doc.get(content_key, "")
            content_hash = hash(content[:prefix_len].lower().strip())

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)

        return unique_docs
