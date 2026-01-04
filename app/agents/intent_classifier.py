from typing import Tuple
from app.models.schemas import IntentType
from app.services.llm_service import LLMService


class IntentClassifier:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    def classify(self, query: str) -> Tuple[IntentType, float, str]:
        result = self.llm_service.classify_intent(query)

        intent_str = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        reason = result.get("reason", "")

        intent_map = {
            "disease": IntentType.DISEASE,
            "scheme": IntentType.SCHEME,
            "hybrid": IntentType.HYBRID,
            "unknown": IntentType.UNKNOWN,
        }

        intent = intent_map.get(intent_str.lower(), IntentType.UNKNOWN)

        return intent, confidence, reason

    def classify_simple(self, query: str) -> IntentType:
        intent, _, _ = self.classify(query)
        return intent
