# **Agriculture Chatbot for farmers**

## **🌾 Problem Statement**

Design and develop an intelligent backend application using FastAPI that helps farmers get accurate information about citrus diseases and government agricultural schemes through a conversational interface.

## **📋 Challenge Overview**

**Duration:** 9am, 3rd January to 11:59pm 4th January  
 **Position:** Full Stack Engineer (primarily a Backend developer) at FiduraAI and consideration for Backend Roles at other Startups  
 **Tech Stack:** Python, FastAPI, LangChain/LangGraph/LangSmith, Vector Database

## **🎯 What You'll Build**

Farmers need quick, accurate answers to their queries about:

* Citrus crop diseases, symptoms, and management  
* Government agricultural schemes and benefits  
* Combination of both (e.g., "What schemes can help me manage Citrus Canker?")

Your task is to build an **Agentic RAG (Retrieval Augmented Generation) system** that:

1. Understands farmer intent from natural language queries  
2. Dynamically routes queries to appropriate knowledge bases  
3. Retrieves relevant information accurately  
4. Returns helpful, context-aware responses

## **📚 Input Documents**

Two PDF documents are provided:

1. [**GovernmentSchemes.pdf**](https://drive.google.com/file/d/1BJ-dR70I67ueu1TUsTQ_sXMAvCiZlGBZ/view?usp=drive_link) \- Government agricultural schemes and benefits  
2. [**CitrusPlantPestsAndDiseases.pdf**](https://drive.google.com/file/d/1RxSRrPnHjHoMUl3_kLaJ7NEi8s4b1hZO/view?usp=drive_link) \- Comprehensive citrus disease guide covering symptoms, prevention, and management

## **🧠 Intent Detection & Query Routing**

Your agentic AI system must intelligently understand the farmer's intent and route queries to the appropriate knowledge base:

### **Intent Categories**

#### **1\. Disease Intent → Routes to Citrus Pests & Diseases Knowledge Base**

Farmer is asking about:

* Disease symptoms and identification  
* Pest problems and infestations  
* Treatment and prevention methods  
* Nutritional deficiencies  
* Plant health issues

#### **2\. Scheme Intent → Routes to Government Schemes Knowledge Base**

Farmer is asking about:

* Government subsidies and financial assistance  
* Agricultural support programs  
* Eligibility criteria for schemes  
* Application processes  
* Available benefits for farmers

#### **3\. Hybrid Intent → Routes to BOTH Knowledge Bases**

Farmer is asking about:

* Financial support for disease management  
* Schemes that help with specific pest control  
* Government assistance combined with agricultural problems  
* Any query connecting schemes with diseases/pests

### **Query Routing Logic**

Your LangChain/LangGraph agent should:

1. **Analyze** the user's natural language query  
2. **Classify** the intent (Disease / Scheme / Hybrid)  
3. **Route** to appropriate vector store(s)  
4. **Retrieve** relevant context from selected knowledge base(s)  
5. **Synthesize** a coherent, farmer-friendly response  
6. **Cite** sources from the retrieved documents

## **📝 Sample Query Examples & Expected Responses**

### **Example 1: Disease Intent**

**User Query:**

```
"My citrus leaves are showing yellow blotchy patches. What could this be?"
```

**Expected System Behavior:**

* **Intent Detected:** Disease  
* **Route To:** Citrus Pests & Diseases Knowledge Base  
* **Response:**

json

````json
{
  "success": true,
  "intent": "disease",
  "answer": "The yellow blotchy patches on your citrus leaves could indicate Huanglongbing (HLB) or Citrus Greening disease. This is characterized by asymmetrical mottling on leaves - random yellow and green patterns that don't mirror across the leaf midrib. Other symptoms include stunted yellow shoots and small, lopsided fruits with bitter taste. HLB is a serious bacterial disease transmitted by the Asian citrus psyllid. Immediate actions: 1) Inspect for psyllid insects on new growth, 2) Remove infected trees to prevent spread, 3) Control psyllids with systemic insecticides like imidacloprid, 4) Use certified disease-free planting material.",
}
```

### Example 2: Disease Intent
**User Query:**
```
"How do I prevent Citrus Canker in my orchard?"
````

**Expected System Behavior:**

* **Intent Detected:** Disease  
* **Route To:** Citrus Pests & Diseases Knowledge Base  
* **Response:**

json

````json
{
  "success": true,
  "intent": "disease",
  "answer": "To prevent Citrus Canker in your orchard: 1) Use canker-free nursery stock from certified sources, 2) Plant windbreaks around orchard borders to reduce wind-driven rain spread, 3) Control citrus leafminer as their damage creates entry points for bacteria, 4) Apply preventive copper-based fungicide sprays (Copper oxychloride 0.3%) three times yearly - February (spring flush), June-July (monsoon flush), and October (late flush), 5) Practice field sanitation by removing and burning infected plant parts, 6) Avoid overhead irrigation that can spread bacteria.",
```

### Example 3: Disease Intent
**User Query:**
```
"What treatment should I use for whitefly infestation on my citrus trees?"
````

**Expected System Behavior:**

* **Intent Detected:** Disease  
* **Route To:** Citrus Pests & Diseases Knowledge Base  
* **Response:**

json

````json
{
  "success": true,
  "intent": "disease",
  "answer": "For whitefly control on citrus: 1) Biological Control: Encourage Encarsia parasitoid wasps that naturally attack whitefly nymphs - avoid broad-spectrum insecticides that kill these beneficial insects, 2) Organic Options: Spray 2% neem oil or horticultural mineral oil to smother nymphs and clean sooty mold, 3) Chemical Control (if severe): Apply systemic insecticides like Imidacloprid as soil drench, or use IGR (Insect Growth Regulator) Buprofezin at 1 ml/L which is effective on nymphs and safe for beneficials, 4) Control ants that protect whiteflies, 5) Ensure spray coverage on leaf undersides where nymphs reside.",
}
```

### Example 4: Scheme Intent
**User Query:**
```
"What government schemes are available for citrus farmers in Andhra Pradesh?"
````

**Expected System Behavior:**

* **Intent Detected:** Scheme  
* **Route To:** Government Schemes Knowledge Base  
* **Response:**

json

````json
{
  "success": true,
  "intent": "scheme",
  "answer": "Several government schemes are available for citrus farmers in Andhra Pradesh: 1) Pradhan Mantri Krishi Sinchai Yojana (PMKSY) - provides subsidies for drip and sprinkler irrigation systems, 2) National Horticulture Mission (NHM) - offers financial assistance for citrus plantation establishment, rejuvenation, and protected cultivation, 3) Paramparagat Krishi Vikas Yojana (PKVY) - supports organic citrus farming with ₹50,000 per hectare assistance, 4) Rashtriya Krishi Vikas Yojana (RKVY) - state-specific schemes for infrastructure development, 5) Agricultural Credit schemes through NABARD and cooperative banks with interest subvention.",
 
}
```

### Example 5: Scheme Intent
**User Query:**
```
"Are there any subsidies for setting up drip irrigation in my citrus farm?"
````

**Expected System Behavior:**

* **Intent Detected:** Scheme  
* **Route To:** Government Schemes Knowledge Base  
* **Response:**

json

````json
{
  "success": true,
  "intent": "scheme",
  "answer": "Yes, drip irrigation subsidies for citrus farms are available under Pradhan Mantri Krishi Sinchai Yojana (PMKSY). The scheme provides: 1) Subsidy up to 55% for small and marginal farmers, 2) Subsidy up to 45% for other farmers, 3) Additional 10% assistance for SC/ST farmers, 4) Coverage includes cost of drip system, installation, and training. Application process: Apply through your District Agriculture Office or online portal, submit land documents and farm details, get technical feasibility assessment, receive approval and subsidy directly in bank account after installation verification.",}
```

### Example 6: Scheme Intent
**User Query:**
```
"How can I get financial help to start organic citrus farming?"
````

**Expected System Behavior:**

* **Intent Detected:** Scheme  
* **Route To:** Government Schemes Knowledge Base  
* **Response:**

json

````json
{
  "success": true,
  "intent": "scheme",
  "answer": "Financial assistance for organic citrus farming is available through: 1) Paramparagat Krishi Vikas Yojana (PKVY) - provides ₹50,000 per hectare over 3 years for organic farming adoption, includes support for organic inputs, certification, and training, 2) National Program for Organic Production (NPOP) - certification assistance and market linkages, 3) Capital Investment Subsidy Scheme - up to 33% subsidy for organic input production units. Benefits include: cluster formation support, organic certification cost coverage, vermicompost and bio-fertilizer subsidies, premium pricing through organic markets. Apply through: State Agriculture Department, register with Participatory Guarantee System (PGS), or approach recognized certification bodies.",
  }
```

### Example 7: Hybrid Intent
**User Query:**
```
"What government schemes can help me manage Citrus Greening disease in my farm?"
````

**Expected System Behavior:**

* **Intent Detected:** Hybrid (Disease \+ Scheme)  
* **Route To:** BOTH Knowledge Bases  
* **Response:**

json

````json
{
  "success": true,
  "intent": "hybrid",
  "answer": "For managing Citrus Greening (HLB), here's integrated support available:\n\nDISEASE MANAGEMENT:\nCitrus Greening is a fatal bacterial disease spread by Asian citrus psyllid. Key management: 1) Remove and destroy infected trees immediately to prevent spread, 2) Aggressive psyllid control using systemic insecticides (Imidacloprid) and biocontrol (Tamarixia wasps), 3) Use only certified disease-free planting material, 4) Provide nutritional support through foliar sprays to prolong productivity of infected trees.\n\nGOVERNMENT SUPPORT:\n1) National Horticulture Mission (NHM) - provides assistance for replanting with disease-free certified material, subsidy for insecticides and biocontrol agents, 2) Rashtriya Krishi Vikas Yojana (RKVY) - state schemes for HLB management including mass psyllid control programs, 3) Crop Insurance under PMFBY - compensation for yield loss due to pest/disease, 4) Soil Health Card scheme - free testing and nutrient management recommendations. Contact your District Horticulture Officer for integrated disease management support and subsidy applications.",
  
}
```

### Example 8: Hybrid Intent
**User Query:**
```
"I need help with pest control equipment and funding. What options do I have?"
````

**Expected System Behavior:**

* **Intent Detected:** Hybrid (Disease \+ Scheme)  
* **Route To:** BOTH Knowledge Bases  
* **Response:**

json

````json
{
  "success": true,
  "intent": "hybrid",
  "answer": "For pest control equipment and funding:\n\nPEST MANAGEMENT EQUIPMENT NEEDS:\nEssential equipment for citrus pest control: 1) Knapsack or motorized sprayers for insecticide/fungicide application, 2) Pheromone traps for monitoring leafminer and fruit flies, 3) Yellow sticky traps for whitefly and psyllid monitoring, 4) Protective gear (masks, gloves) for safe pesticide handling. Integrated Pest Management (IPM) requires regular monitoring and timely intervention.\n\nFUNDING OPTIONS:\n1) Sub-Mission on Agricultural Mechanization (SMAM) - 40-50% subsidy on sprayers and application equipment, 2) National Horticulture Mission - assistance for IPM implementation including equipment and biocontrol agents, 3) Agricultural Credit through Kisan Credit Card (KCC) - loans at 4% interest for purchasing equipment and inputs, 4) NABARD schemes for farm mechanization with low-interest loans. Application: Submit proposal through District Agriculture Mechanization Officer with equipment quotations and land details.",
}
```

### Example 9: Hybrid Intent
**User Query:**
```
"Can I get government support for setting up drip irrigation to prevent root diseases?"
````

**Expected System Behavior:**

* **Intent Detected:** Hybrid (Disease \+ Scheme)  
* **Route To:** BOTH Knowledge Bases  
* **Response:**

json

```json
{
  "success": true,
  "intent": "hybrid",
  "answer": "Yes, drip irrigation helps prevent root diseases and government support is available:\n\nDISEASE PREVENTION BENEFITS:\nDrip irrigation significantly reduces Phytophthora foot rot (gummosis) and other root diseases by: 1) Preventing water accumulation at trunk base - major cause of root rot, 2) Maintaining consistent soil moisture without waterlogging, 3) Reducing fungal spore splash from soil to lower trunk, 4) Allowing better soil aeration for healthy root development. Drip irrigation should be positioned away from trunk (30-40 cm radius) to keep the collar area dry.\n\nGOVERNMENT SUBSIDY:\nPradhan Mantri Krishi Sinchai Yojana (PMKSY) - Per Drop More Crop component provides: 1) 55% subsidy for small/marginal farmers, 2) 45% subsidy for other farmers, 3) Additional 10% for SC/ST categories, 4) Coverage includes drip system, filters, fertigation unit, and installation. Combined with proper drainage management and mulching, drip irrigation is a key component of disease prevention. Apply through State Agriculture/Horticulture Department with farm survey and technical plan.",
}
```

## **🔧 Technical Requirements**

### **Technologies**

* **Framework:** FastAPI  
* **Agentic AI:** LangChain/LangGraph/LangSmith (mandatory)  
* **Vector Database:** Chroma
* **LLM:**  Mistral  
* **Python:** 3.13

### **Application Features**

#### **Phase 1: Document Processing & Vector Store Creation**

* Load and parse the provided PDF documents  
* Implement intelligent text chunking strategy (recommend 500-1000 tokens with overlap)  
* Generate embeddings for document chunks using appropriate embedding model  
* Store embeddings in vector database with metadata  
* Create separate collections/namespaces for:  
  * **Collection 1:** Citrus Pests & Diseases knowledge base  
  * **Collection 2:** Government Schemes knowledge base  
* Implement efficient indexing for fast retrieval

#### **Phase 2: Intent Detection & Agentic Routing (LangChain/LangGraph)**

Build an intelligent agent using LangChain/LangGraph that:

* **Analyzes** the natural language query to understand farmer's intent  
* **Classifies** queries into three categories:  
  * Disease Intent → Route to Citrus Pests & Diseases KB  
  * Scheme Intent → Route to Government Schemes KB  
  * Hybrid Intent → Route to BOTH knowledge bases  
* **Makes routing decisions** dynamically based on query semantics  
* **Handles ambiguous queries** by asking clarifying questions or making intelligent assumptions  
* **Maintains conversation context** for multi-turn interactions

**Implementation Requirements:**

* Use LangChain Agents or LangGraph for orchestration  
* Implement decision-making logic using LLM-based classification  
* Create separate retriever chains for each knowledge base  
* Build hybrid workflow for queries requiring both knowledge bases  
* Use LangSmith for tracing and debugging agent workflows (bonus)

#### **Phase 3: RAG Implementation**

* Implement semantic search in vector stores using similarity metrics  
* Retrieve top-k relevant chunks (recommend k=3-5)  
* Re-rank results for relevance (optional but recommended)  
* Implement context assembly from retrieved chunks  
* Generate farmer-friendly responses using LLM with retrieved context  
* Add source citations with page numbers and confidence scores  
* Handle edge cases:  
  * Query has no relevant information in knowledge base  
  * Conflicting information from multiple sources  
  * Ambiguous or unclear queries

#### **Phase 4: FastAPI Backend Development**

**Primary Endpoint:**

python

```py
POST /query
Content-Type: application/json

Request Body:
{
  "question": "Tell me about Citrus Canker and available government schemes",
}

Response:
{
  "success": true,
  "intent": "hybrid",
  "answer": "Comprehensive response combining disease info and schemes...",
}
```

##  **Evaluation Criteria**

| Intent Detection and Routing | 30% | Accurate classification of user queries into categories such as Disease, Scheme, or Hybrid, and subsequent routing to the relevant knowledge base(s) utilizing LangChain/LangGraph frameworks. |
| :---- | :---- | :---- |
| **Retrieval Efficacy** | 25% | Provision of highly pertinent and precise information extracted from the appropriate knowledge bases, ensuring adequate contextualization. |
| **LangChain/LangGraph Implementation Fidelity** | 20% | Appropriate utilization of LangChain agents, chains, and sophisticated LangGraph workflows for systematic orchestration and informed decision-making. |
| **Response Quality and Clarity** | 15% | Delivery of unambiguous, actionable, and user-friendly responses tailored for farmers, complete with proper citations and professional recommendations. |
| **API Architecture and Documentation** | 10% | Implementation of a streamlined FastAPI solution accompanied by comprehensive Swagger/OpenAPI documentation. |

### **Edge Cases to Handle**

* Queries with spelling mistakes or colloquial language  
* Very broad or very specific queries  
* Queries mixing multiple diseases or schemes  
* Questions outside the knowledge base scope  
* Ambiguous queries requiring clarification

## **🛠️ Setup Instructions**

### **Prerequisites**

* Python 3.9+  
* Virtual environment (venv/conda)  
* API keys for chosen LLM provider (Mistral API)  
* Vector database setup (local or cloud)  
* Git for version control

### **Environment Setup**

Create a `.env` file with:

bash

```shell
# LLM Provider
MISTRAL_API_KEY=your_gemini_apikey
# or
ANTHROPIC_API_KEY=your_anthropic_key

# LangChain (optional)
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agrigpt-hackathon

# Vector Database
VECTOR_DB_URL=your_vector_db_url
VECTOR_DB_API_KEY=your_vector_db_key  # if using cloud

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
```

**2\. Comprehensive README.md** including:

* Project overview and objectives  
  * Architecture diagram (visual representation)  
  * Setup and installation instructions  
  * Environment variables documentation  
  * API endpoint documentation with examples  
  * LangChain/LangGraph workflow explanation  
  * Intent detection and routing logic  
  * Vector database choice justification  
  * Chunking strategy and reasoning  
  * Challenges faced and solutions implemented  
  * Performance optimization techniques  
  * Future improvements and scalability considerations  
3. **API Documentation**  
   * Automatic Swagger/OpenAPI docs via FastAPI  
   * Postman collection (optional but recommended)  
   * Example requests and responses for all 3 intent types  
4. **Deployment**  
   * Deploy to one of:  
     * Render (recommended for FastAPI)  
     * Railway  
     * Fly.io  
     * Hugging Face Spaces  
     * Google Cloud Run  
   * Provide working API endpoint URL  
   * Include deployment instructions in README  
5. **Demo Video (Optional, 3-5 minutes)**  
   * Live API demonstration with sample queries  
   * Walkthrough of code architecture  
   * Explanation of LangChain/LangGraph workflows  
   * Intent detection and routing demonstration  
   * Discussion of technical decisions and trade-offs  
   * Upload to YouTube/Loom and include link in README

### **Submission Checklist**

* GitHub repository with complete, documented code  
* README with comprehensive setup and architecture documentation  
* Working deployed API endpoint (publicly accessible)  
* `.env.example` with all required environment variables  
* `requirements.txt` or `pyproject.toml` with dependencies  
* API documentation (Swagger/OpenAPI)  
* Test queries covering all 3 intent types (Disease/Scheme/Hybrid)  
* LangChain/LangGraph workflow diagram  
* Source code comments and docstrings  
* Optional: Demo video link  
* Optional: LangSmith trace examples

### **Submission Form**

**Submit your application at:**  
 [https://docs.google.com/forms/d/e/1FAIpQLSfmO91ttMW1\_JPvIcjvGyGeTK6E5siGfDYUJyrhz0W7vlr\_cw/viewform](https://docs.google.com/forms/d/e/1FAIpQLSfmO91ttMW1_JPvIcjvGyGeTK6E5siGfDYUJyrhz0W7vlr_cw/viewform)

### **Questions?**

Write an email to **support@alumnx.com**