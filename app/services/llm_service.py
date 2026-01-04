"""
LLM service for managing language model interactions.
Provides a centralized interface for Mistral AI API calls.
"""

from typing import List, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.models.config import settings


class LLMService:
    """Service for managing Mistral LLM interactions"""

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        """
        Initialize LLM service with Mistral.

        Args:
            model_name: Mistral model name (defaults to config)
            temperature: Sampling temperature (defaults to config)
        """
        self.model_name = model_name or settings.mistral_model
        self.temperature = (
            temperature if temperature is not None else settings.mistral_temperature
        )
        self._llm = None

    @property
    def llm(self) -> ChatMistralAI:
        """Lazy initialization of Mistral LLM"""
        if self._llm is None:
            self._llm = ChatMistralAI(
                model=self.model_name,
                api_key=settings.mistral_api_key,
                temperature=self.temperature,
            )
        return self._llm

    def generate_response(
        self, prompt: str, context: List[str] | None = None, max_tokens: int = 1000
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: User's question or prompt
            context: Optional context documents
            max_tokens: Maximum tokens in response

        Returns:
            Generated response text
        """
        if context:
            context_text = "\n\n".join(context)
            full_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt

        response = self.llm.invoke(full_prompt)
        return response.content

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent using LLM with confidence score.

        Args:
            query: User's question

        Returns:
            Dict with intent and confidence
        """
        classification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert intent classifier for an agriculture chatbot helping farmers with citrus crops.

Classify the farmer's query into exactly ONE of these three categories:

1. DISEASE - Questions about:
   - Crop diseases, pests, symptoms, infections
   - Treatment methods, pesticides, fungicides
   - Prevention and control measures
   - Plant health issues, nutritional deficiencies
   - Identification of problems on leaves, fruits, stems

2. SCHEME - Questions about:
   - Government subsidies and financial assistance
   - Agricultural programs and schemes (PMKSY, NHM, RKVY, etc.)
   - Eligibility criteria and application processes
   - Loans, insurance, and benefits for farmers
   - Policy-related queries

3. HYBRID - Questions that combine BOTH disease/pest management AND government support:
   - Financial help for disease treatment
   - Subsidies for pest control equipment
   - Government schemes for managing specific diseases
   - Any query connecting agricultural problems with government assistance

Analyze the query carefully and respond in this EXACT format:
INTENT: <disease|scheme|hybrid>
CONFIDENCE: <0.0-1.0>
REASON: <brief explanation>

Be strict: Only classify as HYBRID if the query explicitly connects disease/pest issues with government support.""",
                ),
                ("user", "{query}"),
            ]
        )

        chain = classification_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})

        # Parse the response
        lines = result.strip().split("\n")
        intent = "unknown"
        confidence = 0.5
        reason = ""

        for line in lines:
            line = line.strip()
            if line.upper().startswith("INTENT:"):
                intent_value = line.split(":", 1)[1].strip().lower()
                if intent_value in ["disease", "scheme", "hybrid"]:
                    intent = intent_value
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return {"intent": intent, "confidence": confidence, "reason": reason}

    def generate_rag_response(
        self, query: str, documents: List[Dict[str, Any]], intent: str
    ) -> str:
        """
        Generate response using retrieved documents with RAG.

        Args:
            query: User's question
            documents: Retrieved documents with metadata
            intent: Classified intent type

        Returns:
            Generated response text
        """
        # Format context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "Unknown")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{content}")

        context = "\n\n".join(context_parts)

        # Different prompts based on intent
        if intent == "hybrid":
            system_prompt = """You are an expert agricultural advisor helping Indian farmers with citrus crops.

The farmer is asking about BOTH disease/pest management AND government support schemes.

Using the provided context, give a comprehensive response that:
1. First explains the disease/pest issue and management strategies
2. Then describes relevant government schemes and financial support available
3. Provides actionable steps the farmer can take
4. Uses simple, farmer-friendly language

Structure your response with clear sections:
- DISEASE MANAGEMENT: (treatment, prevention, identification)
- GOVERNMENT SUPPORT: (schemes, subsidies, how to apply)

If specific information is not in the context, say so rather than making things up.
Always cite which source document the information comes from."""
        elif intent == "disease":
            system_prompt = """You are an expert agricultural advisor helping Indian farmers with citrus crops.

Using the provided context about citrus diseases and pests, give a helpful response that:
1. Identifies the disease/pest if symptoms are described
2. Explains treatment and prevention methods
3. Provides practical, actionable advice
4. Uses simple, farmer-friendly language

If the farmer describes symptoms, try to identify potential causes.
If specific information is not in the context, say so rather than making things up."""
        else:  # scheme
            system_prompt = """You are an expert agricultural advisor helping Indian farmers understand government schemes.

Using the provided context about government agricultural schemes, give a helpful response that:
1. Explains relevant schemes and their benefits
2. Describes eligibility criteria
3. Outlines the application process
4. Mentions subsidy amounts and coverage where available
5. Uses simple, farmer-friendly language

Provide specific details about schemes mentioned in the context.
If specific information is not in the context, say so rather than making things up."""

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    """Context from knowledge base:
{context}

Farmer's Question: {query}

Provide a helpful, detailed response based on the context above. Include specific recommendations where possible.""",
                ),
            ]
        )

        chain = rag_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})

        return response

    def generate_with_citations(
        self, query: str, documents: List[Dict[str, Any]], intent: str
    ) -> Dict[str, Any]:
        """
        Generate response with source citations.

        Args:
            query: User's question
            documents: Retrieved documents with metadata
            intent: Classified intent type

        Returns:
            Response dict with answer and formatted citations
        """
        # Handle edge case: no documents
        if not documents:
            return {
                "answer": self._generate_no_results_response(query, intent),
                "citations": [],
            }

        answer = self.generate_rag_response(query, documents, intent)

        # Format citations from documents with enhanced scoring
        citations = []
        seen_sources = set()

        for doc in documents:
            source = doc.get("source", "Unknown")
            page = doc.get("page")
            # Use rerank_score if available, otherwise use similarity score
            score = doc.get("rerank_score") or doc.get("score", 0.0)
            content = doc.get("content", "")
            collection = doc.get("collection", "unknown")

            # Create unique key to avoid duplicates
            source_key = f"{source}_{page}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)

                # Determine confidence level
                if score >= 0.8:
                    confidence = "high"
                elif score >= 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"

                citations.append(
                    {
                        "document": source,
                        "page": page,
                        "relevance_score": round(score, 3) if score else None,
                        "confidence": confidence,
                        "collection": collection,
                        "excerpt": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                    }
                )

        return {"answer": answer, "citations": citations}

    def _generate_no_results_response(self, query: str, intent: str) -> str:
        """
        Generate response when no relevant documents found.

        Args:
            query: User's question
            intent: Classified intent type

        Returns:
            Helpful response explaining no results
        """
        if intent == "disease":
            return (
                "I couldn't find specific information about that disease or pest in my knowledge base. "
                "Please try describing the symptoms in more detail (e.g., 'yellow spots on leaves', "
                "'wilting fruit'), or ask about common citrus issues like Citrus Canker, "
                "Citrus Greening (HLB), or whitefly infestations."
            )
        elif intent == "scheme":
            return (
                "I couldn't find information about that specific government scheme. "
                "Try asking about popular programs like PMKSY (drip irrigation), "
                "National Horticulture Mission (NHM), or Kisan Credit Card (KCC). "
                "You can also ask about subsidies for specific farming activities."
            )
        else:
            return (
                "I couldn't find relevant information to answer your question. "
                "I specialize in citrus crop diseases/pests and government agricultural schemes. "
                "Please try rephrasing your query or ask about topics like disease treatment, "
                "pest control, or available farmer subsidies."
            )

    def generate_ambiguous_query_response(self, query: str) -> str:
        """
        Generate response for ambiguous or unclear queries.

        Args:
            query: User's ambiguous question

        Returns:
            Response asking for clarification
        """
        clarification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful agricultural advisor. The farmer's query is unclear or too brief.
Generate a friendly response that:
1. Acknowledges you want to help
2. Explains what information you need
3. Gives 2-3 specific example questions they could ask

Keep the response brief and farmer-friendly.""",
                ),
                ("user", "Farmer's query: {query}"),
            ]
        )

        chain = clarification_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
