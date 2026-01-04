from typing import List, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.models.config import settings


class LLMService:
    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        self.model_name = model_name or settings.mistral_model
        self.temperature = (
            temperature if temperature is not None else settings.mistral_temperature
        )
        self._llm = None

    @property
    def llm(self) -> ChatMistralAI:
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
        if context:
            context_text = "\n\n".join(context)
            full_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt

        response = self.llm.invoke(full_prompt)
        return response.content

    def classify_intent(self, query: str) -> Dict[str, Any]:
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
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "Unknown")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{content}")

        context = "\n\n".join(context_parts)
        if intent == "hybrid":
            system_prompt = """You are an expert agricultural advisor helping Indian farmers with citrus crops.

The farmer is asking about BOTH disease/pest management AND government support schemes.

CRITICAL RESPONSE FORMAT - Follow this EXACT structure:

For managing [disease/pest name], here's integrated support available:

DISEASE MANAGEMENT:
[Disease name] is [brief description]. Key management: 1) [First action], 2) [Second action], 3) [Third action], 4) [Fourth action if applicable].

GOVERNMENT SUPPORT:
1) [Scheme Name 1] - [specific benefits and details], 2) [Scheme Name 2] - [specific benefits and details], 3) [Scheme Name 3] - [specific benefits and details], 4) [Additional scheme if applicable]. Contact your District Horticulture/Agriculture Officer for [relevant support].

CRITICAL FORMATTING RULES - MUST FOLLOW:
- Use numbered lists with ) format: 1), 2), 3)
- Keep the response as a SINGLE PARAGRAPH for each section (no line breaks within sections)
- Include specific details: chemical names, concentrations, subsidy percentages, scheme acronyms
- Provide actionable steps farmers can immediately follow
- Mention application processes and where to apply
- Be comprehensive but concise - aim for detailed yet readable responses

ABSOLUTELY NO MARKDOWN FORMATTING:
- Do NOT use asterisks (*) for any formatting
- Do NOT use underscores (_) for emphasis
- Do NOT use backticks (`) for code
- Do NOT use any special formatting characters
- Use ONLY plain text with numbered lists using ) format

If specific information is not in the context, say so rather than making things up."""
        elif intent == "disease":
            system_prompt = """You are an expert agricultural advisor helping Indian farmers with citrus crops.

Using the provided context about citrus diseases and pests, generate a response following this EXACT FORMAT:

CRITICAL RESPONSE FORMAT:
- Start directly with the answer (no greetings or preamble)
- Use numbered lists with ) format: 1), 2), 3), 4), 5)
- Keep as a SINGLE FLOWING PARAGRAPH - do NOT use line breaks or bullet points
- Include specific details: chemical names with concentrations (e.g., "Copper oxychloride 0.3%"), dosages (e.g., "1 ml/L"), timing (e.g., "February, June-July, October")
- Provide actionable, step-by-step advice farmers can immediately follow

Example format for disease prevention:
"To prevent [Disease] in your orchard: 1) [First action with specific details], 2) [Second action with specific details], 3) [Third action with specific details], 4) [Fourth action with specific details], 5) [Fifth action with specific details], 6) [Additional action if needed]."

Example format for symptom identification:
"The [symptoms described] could indicate [Disease Name]. This is characterized by [specific symptoms]. Other symptoms include [additional symptoms]. [Disease Name] is [brief description of cause/transmission]. Immediate actions: 1) [First action], 2) [Second action], 3) [Third action], 4) [Fourth action]."

Example format for treatment:
"For [pest/disease] control on citrus: 1) Biological Control: [specific method with details], 2) Organic Options: [specific products with concentrations], 3) Chemical Control (if severe): [specific chemicals with dosages], 4) [Additional integrated management step], 5) [Final recommendation]."

ABSOLUTELY NO MARKDOWN FORMATTING:
- Do NOT use asterisks (*) for any formatting or emphasis
- Do NOT use underscores (_) for emphasis
- Do NOT use backticks (`) for code
- Do NOT use any special formatting characters
- Use ONLY plain text with numbered lists using ) format
- Write scientific names in plain text without italics (e.g., write "Diaphorina citri" not "*Diaphorina citri*")

If specific information is not in the context, say so rather than making things up."""
        else:
            system_prompt = """You are an expert agricultural advisor helping Indian farmers understand government schemes.

Using the provided context about government agricultural schemes, generate a response following this EXACT FORMAT:

CRITICAL RESPONSE FORMAT:
- Start directly with the answer (no greetings or preamble)
- Use numbered lists with ) format: 1), 2), 3), 4), 5)
- Keep as a SINGLE FLOWING PARAGRAPH - do NOT use line breaks or bullet points
- Include specific details: scheme names with acronyms, subsidy percentages, amounts in ₹, eligibility criteria

Example format for available schemes:
"Several government schemes are available for [topic]: 1) [Scheme Name (Acronym)] - [specific benefits with amounts/percentages], 2) [Scheme Name (Acronym)] - [specific benefits], 3) [Scheme Name (Acronym)] - [specific benefits with ₹ amounts], 4) [Scheme Name (Acronym)] - [specific benefits], 5) [Additional scheme or credit options]."

Example format for specific scheme inquiry:
"Yes, [topic] subsidies are available under [Scheme Name (Acronym)]. The scheme provides: 1) Subsidy up to [X]% for small and marginal farmers, 2) Subsidy up to [Y]% for other farmers, 3) Additional [Z]% assistance for SC/ST farmers, 4) Coverage includes [specific items covered]. Application process: [Step-by-step application instructions]."

Example format for financial assistance:
"Financial assistance for [topic] is available through: 1) [Scheme Name (Acronym)] - provides ₹[amount] per hectare over [duration], includes [specific support], 2) [Scheme Name (Acronym)] - [specific assistance], 3) [Scheme Name (Acronym)] - up to [X]% subsidy for [specific items]. Benefits include: [list of benefits]. Apply through: [application method and contact points]."

ABSOLUTELY NO MARKDOWN FORMATTING:
- Do NOT use asterisks (*) for any formatting or emphasis
- Do NOT use underscores (_) for emphasis
- Do NOT use backticks (`) for code
- Do NOT use any special formatting characters
- Use ONLY plain text with numbered lists using ) format

If specific information is not in the context, say so rather than making things up."""

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "user",
                    """Context from knowledge base:
{context}

Farmer's Question: {query}

Generate a response following the EXACT format specified. Use numbered lists with ) format. Keep response as flowing paragraphs without line breaks. Include specific details from the context.""",
                ),
            ]
        )

        chain = rag_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})

        return response

    def generate_with_citations(
        self, query: str, documents: List[Dict[str, Any]], intent: str
    ) -> Dict[str, Any]:
        if not documents:
            return {
                "answer": self._generate_no_results_response(query, intent),
                "citations": [],
            }

        answer = self.generate_rag_response(query, documents, intent)

        return {"answer": answer, "citations": []}

    def _generate_no_results_response(self, query: str, intent: str) -> str:
        if intent == "disease":
            return (
                "I couldn't find specific information about that disease or pest in my knowledge base. "
                "Please try: 1) Describing the symptoms in more detail (e.g., 'yellow spots on leaves', 'wilting fruit'), "
                "2) Asking about common citrus issues like Citrus Canker, Citrus Greening (HLB), or whitefly infestations, "
                "3) Specifying the affected plant part (leaves, fruits, stems, roots)."
            )
        elif intent == "scheme":
            return (
                "I couldn't find information about that specific government scheme. "
                "Try asking about: 1) Pradhan Mantri Krishi Sinchai Yojana (PMKSY) for drip irrigation subsidies, "
                "2) National Horticulture Mission (NHM) for citrus plantation support, "
                "3) Kisan Credit Card (KCC) for agricultural loans, "
                "4) Paramparagat Krishi Vikas Yojana (PKVY) for organic farming assistance."
            )
        else:
            return (
                "I couldn't find relevant information to answer your question. "
                "I specialize in: 1) Citrus crop diseases and pests - symptoms, treatment, and prevention, "
                "2) Government agricultural schemes - subsidies, eligibility, and application processes, "
                "3) Combined support for disease management with government assistance. "
                "Please try rephrasing your query with more specific details."
            )

    def generate_ambiguous_query_response(self, query: str) -> str:
        clarification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful agricultural advisor. The farmer's query is unclear or too brief.

Generate a response in this EXACT FORMAT:
"I'd like to help you with your query about [topic]. To provide accurate information, please specify: 1) [First clarification needed], 2) [Second clarification needed], 3) [Third clarification if applicable]. Example questions you could ask: '[Example question 1]', '[Example question 2]', '[Example question 3]'."

IMPORTANT:
- Use numbered lists with ) format
- Keep as a single flowing paragraph
- Be farmer-friendly and helpful
- Provide specific example questions related to citrus farming or government schemes""",
                ),
                ("user", "Farmer's query: {query}"),
            ]
        )

        chain = clarification_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
