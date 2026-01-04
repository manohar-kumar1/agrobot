from pydantic import BaseModel, Field
from typing import List, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    DISEASE = "disease"
    SCHEME = "scheme"
    HYBRID = "hybrid"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question", min_length=1)
    user_id: str | None = Field(None, description="Optional user identifier")
    session_id: str | None = Field(None, description="Optional session identifier")

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What are the symptoms of citrus canker?"}]
        }
    }


class Source(BaseModel):
    document: str = Field(..., description="Source document name")
    page: int | None = Field(None, description="Page number if applicable")
    relevance_score: float | None = Field(None, description="Relevance score")
    excerpt: str | None = Field(None, description="Relevant text excerpt")


class QueryResponse(BaseModel):
    success: bool = Field(True, description="Whether query was processed successfully")
    intent: IntentType = Field(..., description="Detected query intent")
    answer: str = Field(..., description="Generated answer")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "intent": "greeting",
                    "answer": "Namaste! Welcome to AgroBot - your citrus farming assistant! I can help you with: 1) Citrus disease identification and treatment, 2) Government agricultural schemes and subsidies, 3) Pest management advice. How can I assist you today?",
                },
                {
                    "success": True,
                    "intent": "disease",
                    "answer": "Citrus canker is a bacterial disease caused by Xanthomonas citri. Symptoms include raised, corky lesions on leaves and fruit with yellow halos. Treatment: 1) Apply copper-based fungicides (Copper oxychloride 0.3%), 2) Remove and burn infected plant parts, 3) Use canker-free nursery stock, 4) Control citrus leafminer to prevent entry points.",
                },
                {
                    "success": True,
                    "intent": "scheme",
                    "answer": "Drip irrigation subsidies are available under Pradhan Mantri Krishi Sinchai Yojana (PMKSY). The scheme provides: 1) Subsidy up to 55% for small and marginal farmers, 2) Subsidy up to 45% for other farmers, 3) Additional 10% assistance for SC/ST farmers, 4) Coverage includes cost of drip system, installation, and training. Application process: Apply through your District Agriculture Office or online portal.",
                },
                {
                    "success": True,
                    "intent": "hybrid",
                    "answer": "For managing Citrus Greening (HLB), here's integrated support available: DISEASE MANAGEMENT: Citrus Greening is a fatal bacterial disease spread by Asian citrus psyllid. Key management: 1) Remove and destroy infected trees immediately, 2) Control psyllids using systemic insecticides (Imidacloprid), 3) Use certified disease-free planting material. GOVERNMENT SUPPORT: 1) National Horticulture Mission (NHM) - provides assistance for replanting with disease-free material, 2) Pradhan Mantri Fasal Bima Yojana (PMFBY) - compensation for yield loss due to pest/disease. Contact your District Horticulture Officer for support.",
                },
                {
                    "success": True,
                    "intent": "out_of_scope",
                    "answer": "I appreciate your question, but I specialize specifically in citrus farming assistance. I can help you with: 1) Citrus crop diseases - symptoms, identification, and treatment, 2) Government agricultural schemes - subsidies, loans, and benefits for farmers, 3) Integrated support combining disease management with government assistance. Please ask me something related to citrus farming!",
                },
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    timestamp: str | None = None
