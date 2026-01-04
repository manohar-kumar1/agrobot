"""
Intent classification agent for routing queries to appropriate handlers.
Classifies queries as: Disease, Scheme, or Hybrid using LLM-based analysis.
"""

from typing import Tuple
from app.models.schemas import IntentType
from app.services.llm_service import LLMService


class IntentClassifier:
    """Classifies user queries into different intent types using Mistral LLM"""

    def __init__(self, llm_service: LLMService | None = None):
        """
        Initialize intent classifier with LLM service.

        Args:
            llm_service: Optional LLM service instance
        """
        self.llm_service = llm_service or LLMService()

    def classify(self, query: str) -> Tuple[IntentType, float, str]:
        """
        Classify the user's query intent with confidence score.

        Args:
            query: User's question

        Returns:
            Tuple of (IntentType enum, confidence score, reason)
        """
        # Use LLM service for classification
        result = self.llm_service.classify_intent(query)

        intent_str = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        reason = result.get("reason", "")

        # Map string to IntentType enum
        intent_map = {
            "disease": IntentType.DISEASE,
            "scheme": IntentType.SCHEME,
            "hybrid": IntentType.HYBRID,
            "unknown": IntentType.UNKNOWN,
        }

        intent = intent_map.get(intent_str.lower(), IntentType.UNKNOWN)

        return intent, confidence, reason

    def classify_simple(self, query: str) -> IntentType:
        """
        Simple classification returning only IntentType.

        Args:
            query: User's question

        Returns:
            IntentType enum value
        """
        intent, _, _ = self.classify(query)
        return intent
