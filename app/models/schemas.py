"""
Pydantic models and schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    """Query intent types"""
    DISEASE = "disease"
    SCHEME = "scheme"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str = Field(..., description="User's question", min_length=1)
    user_id: str | None = Field(None, description="Optional user identifier")
    session_id: str | None = Field(None, description="Optional session identifier")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the symptoms of citrus canker?"
                }
            ]
        }
    }


class Source(BaseModel):
    """Source citation model"""
    document: str = Field(..., description="Source document name")
    page: int | None = Field(None, description="Page number if applicable")
    relevance_score: float | None = Field(None, description="Relevance score")
    excerpt: str | None = Field(None, description="Relevant text excerpt")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool = Field(True, description="Whether query was processed successfully")
    intent: IntentType = Field(..., description="Detected query intent")
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    confidence: float | None = Field(None, description="Confidence score", ge=0, le=1)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "intent": "disease",
                    "answer": "Citrus canker is a bacterial disease caused by Xanthomonas citri. Symptoms include raised, corky lesions on leaves and fruit with yellow halos. Treatment: Apply copper-based fungicides (Copper oxychloride 0.3%) and remove infected plant parts.",
                    "sources": [
                        {
                            "document": "CitrusPlantPestsAndDiseases.pdf",
                            "page": 5,
                            "relevance_score": 0.95,
                            "excerpt": "Citrus canker causes raised lesions..."
                        }
                    ],
                    "confidence": 0.92
                },
                {
                    "success": True,
                    "intent": "scheme",
                    "answer": "Drip irrigation subsidies are available under PMKSY. Small farmers get 55% subsidy, others get 45%. Apply through your District Agriculture Office.",
                    "sources": [
                        {
                            "document": "GovernmentSchemes.pdf",
                            "page": 12,
                            "relevance_score": 0.88
                        }
                    ],
                    "confidence": 0.85
                },
                {
                    "success": True,
                    "intent": "hybrid",
                    "answer": "For Citrus Greening management: Remove infected trees, control psyllids with Imidacloprid. Government support: NHM provides subsidy for replanting, PMFBY covers crop loss.",
                    "sources": [
                        {
                            "document": "CitrusPlantPestsAndDiseases.pdf",
                            "page": 8,
                            "relevance_score": 0.90
                        },
                        {
                            "document": "GovernmentSchemes.pdf", 
                            "page": 15,
                            "relevance_score": 0.82
                        }
                    ],
                    "confidence": 0.88
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str | None = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str | None = None
    timestamp: str | None = None
