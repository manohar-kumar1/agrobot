# Agrobot - Agriculture Chatbot API

An intelligent FastAPI-based chatbot backend for farmers, providing information on citrus diseases and government agricultural schemes through a conversational RAG (Retrieval-Augmented Generation) interface.

## 🏗️ Project Structure

```
agrobot/
├── app/                          # Main application package
│   ├── api/                      # API layer
│   │   └── v1/                   # API version 1
│   │       ├── endpoints/        # API endpoints
│   │       │   └── query.py      # Main query endpoint
│   │       └── routes.py         # Router configuration
│   ├── agents/                   # AI agents
│   │   ├── intent_classifier.py  # Intent classification agent
│   │   └── rag_agent.py          # RAG processing agent
│   ├── core/                     # Core utilities
│   │   ├── dependencies.py       # FastAPI dependencies
│   │   └── logging_config.py     # Logging configuration
│   ├── embeddings/               # Vector embeddings
│   │   ├── document_processor.py # PDF processing
│   │   └── vector_store.py       # ChromaDB operations
│   ├── models/                   # Data models
│   │   ├── config.py             # Configuration settings
│   │   └── schemas.py            # Pydantic schemas
│   ├── services/                 # Business logic services
│   │   ├── embedding_service.py  # Embedding generation
│   │   └── llm_service.py        # LLM interactions
│   ├── utils/                    # Utility functions
│   │   └── text_processing.py    # Text utilities
│   └── main.py                   # FastAPI app initialization
├── scripts/                      # Utility scripts
│   ├── ingest_documents.py       # Document ingestion script
│   └── test_api.py               # API testing script
├── data/                         # Data storage
│   └── chroma/                   # Vector database (gitignored)
├── logs/                         # Application logs (gitignored)
├── main.py                       # Application entry point
├── pyproject.toml                # Project dependencies
└── .env                          # Environment variables

```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- OpenAI API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd d:\programs\Hackathon\agrobot
   ```

2. **Activate virtual environment:**
   ```bash
   .venv\Scripts\activate
   ```

3. **Install dependencies (already done with uv):**
   ```bash
   uv sync
   ```

4. **Configure environment variables:**
   
   Update your `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   OPENAI_TEMPERATURE=0.7
   
   CHROMA_PERSIST_DIRECTORY=./data/chroma
   CHROMA_COLLECTION_NAME=agrobot_docs
   
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K_RESULTS=5
   
   DEBUG=false
   ```

### Running the Application

1. **Ingest documents (first time only):**
   ```bash
   python scripts/ingest_documents.py
   ```

2. **Start the API server:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Testing

Run the test script:
```bash
python scripts/test_api.py
```

## 📡 API Endpoints

### Health Check
```
GET /health
GET /api/v1/health
```

### Query Endpoint
```
POST /api/v1/query
```

**Request Body:**
```json
{
  "question": "What are the symptoms of citrus canker?",
  "user_id": "farmer_123",
  "session_id": "session_456"
}
```

**Response:**
```json
{
  "answer": "Citrus canker is a bacterial disease...",
  "intent": "disease",
  "sources": [
    {
      "document": "CitrusPlantPestsAndDiseases.pdf",
      "page": 5,
      "relevance_score": 0.95,
      "excerpt": "..."
    }
  ],
  "confidence": 0.92
}
```

## 🧩 Architecture

### Intent Classification
Queries are classified into three types:
- **Disease**: Questions about citrus diseases, pests, symptoms, treatments
- **Scheme**: Questions about government schemes, subsidies, eligibility
- **Hybrid**: Questions combining both disease and scheme information

### RAG Pipeline
1. **Document Processing**: PDFs are chunked and embedded
2. **Vector Storage**: Embeddings stored in ChromaDB
3. **Retrieval**: Relevant chunks retrieved based on query similarity
4. **Generation**: LLM generates farmer-friendly responses with citations

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: ChromaDB
- **Orchestration**: LangChain/LangGraph
- **Python**: 3.13

## 📝 Development Notes

### Key Design Decisions

1. **Minimal `__init__.py` files**: Only created where Python requires them for package imports
2. **Dependency Injection**: Using FastAPI's dependency injection for service management
3. **Singleton Pattern**: Services are instantiated once and reused
4. **Async/Await**: All I/O operations are async for better performance
5. **Pydantic Settings**: Environment-based configuration management

### Next Steps (TODOs)

- [ ] Implement ChromaDB initialization in `vector_store.py`
- [ ] Implement PDF loading in `document_processor.py`
- [ ] Implement LLM service methods in `llm_service.py`
- [ ] Implement RAG pipeline in `rag_agent.py`
- [ ] Implement intent classification in `intent_classifier.py`
- [ ] Add error handling and validation
- [ ] Add unit tests
- [ ] Add API rate limiting
- [ ] Add authentication/authorization
- [ ] Deploy to production

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors here]
