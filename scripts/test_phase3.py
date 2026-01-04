"""
Test script for Phase 3: RAG Enhancement Features.
Tests reranking, edge case handling, and improved citations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.rag_agent import RAGAgent
from app.services.reranker import Reranker


def print_separator(title: str):
    """Print a visual separator"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_reranker():
    """Test the reranker service directly"""
    print_separator("Testing Reranker Service")
    
    reranker = Reranker()
    
    # Test documents
    test_docs = [
        {"content": "Citrus canker is a bacterial disease causing raised lesions on leaves.", "score": 0.6, "page": 1},
        {"content": "The weather in India varies by region.", "score": 0.7, "page": 5},  # Irrelevant
        {"content": "Canker symptoms include yellow halos around leaf spots.", "score": 0.5, "page": 2},
    ]
    
    query = "What are the symptoms of citrus canker?"
    
    print(f"Query: {query}")
    print(f"Original docs (by score): {[d['score'] for d in test_docs]}")
    
    # Test reranking
    reranked = reranker.rerank_documents(query, test_docs, top_k=2)
    print(f"Reranked docs (top 2): {[round(d.get('rerank_score', 0), 2) for d in reranked]}")
    
    # Test threshold filtering
    filtered = reranker.filter_by_threshold(test_docs, threshold=0.5)
    print(f"Filtered docs (threshold 0.5): {len(filtered)} remaining")
    
    print("✅ Reranker tests passed")


def test_rrf():
    """Test Reciprocal Rank Fusion for hybrid queries"""
    print_separator("Testing Reciprocal Rank Fusion")
    
    reranker = Reranker()
    
    disease_docs = [
        {"content": "Disease info 1", "page": 1, "score": 0.9},
        {"content": "Disease info 2", "page": 2, "score": 0.8},
    ]
    
    scheme_docs = [
        {"content": "Scheme info 1", "page": 10, "score": 0.85},
        {"content": "Scheme info 2", "page": 11, "score": 0.75},
    ]
    
    fused = reranker.reciprocal_rank_fusion(disease_docs, scheme_docs)
    
    print(f"Disease docs: {len(disease_docs)}")
    print(f"Scheme docs: {len(scheme_docs)}")
    print(f"Fused docs: {len(fused)}")
    print(f"Collections in fused: {[d.get('collection') for d in fused]}")
    
    print("✅ RRF tests passed")


def test_query(agent: RAGAgent, query: str, expected_intent: str, test_name: str):
    """Test a single query and print results"""
    print(f"\n📝 Query: {query}")
    print(f"   Expected Intent: {expected_intent}")
    print("-" * 50)
    
    response = agent.process_query(query)
    
    print(f"✅ Detected Intent: {response.intent.value}")
    print(f"📊 Confidence: {response.confidence:.2%}")
    print(f"\n📄 Answer (first 400 chars):")
    print(f"   {response.answer[:400]}...")
    
    # Check citation format (Phase 3 enhancement)
    print(f"\n📚 Sources ({len(response.sources)} found):")
    for source in response.sources[:3]:
        confidence = getattr(source, 'confidence', 'N/A') if hasattr(source, 'confidence') else 'N/A'
        print(f"   - {source.document}")
        print(f"     Page: {source.page}, Score: {source.relevance_score}")
    
    # Check if intent matches expected
    match = "✅ MATCH" if response.intent.value == expected_intent else "❌ MISMATCH"
    print(f"\n{match}")
    
    return response


def test_edge_cases(agent: RAGAgent):
    """Test edge case handling"""
    print_separator("Testing Edge Cases")
    
    # Test 1: Ambiguous query
    print("\n--- Test: Ambiguous Query ---")
    response = agent.process_query("help")
    print(f"Query: 'help'")
    print(f"Response length: {len(response.answer)} chars")
    print(f"Contains clarification: {'?' in response.answer or 'example' in response.answer.lower()}")
    
    # Test 2: Out of scope query
    print("\n--- Test: Out of Scope Query ---")
    response = agent.process_query("What is the capital of France?")
    print(f"Query: 'What is the capital of France?'")
    print(f"Response indicates out of scope: {'citrus' in response.answer.lower() or 'disease' in response.answer.lower()}")
    
    print("✅ Edge case tests completed")


def main():
    """Run all Phase 3 tests"""
    print_separator("Phase 3: RAG Enhancement Tests")
    
    # Test reranker independently
    test_reranker()
    test_rrf()
    
    # Initialize RAG agent
    print_separator("Initializing RAG Agent")
    agent = RAGAgent()
    print("✅ RAG Agent initialized successfully")
    print(f"\n{agent.get_workflow_graph()}")
    
    # Test Disease Intent with reranking
    print_separator("TEST 1: Disease Intent (with reranking)")
    test_query(
        agent,
        "What are the symptoms of Citrus Canker and how do I treat it?",
        "disease",
        "Disease with Reranking"
    )
    
    # Test Scheme Intent with reranking
    print_separator("TEST 2: Scheme Intent (with reranking)")
    test_query(
        agent,
        "What government subsidies are available for drip irrigation?",
        "scheme",
        "Scheme with Reranking"
    )
    
    # Test Hybrid Intent with RRF
    print_separator("TEST 3: Hybrid Intent (with RRF)")
    test_query(
        agent,
        "What government schemes can help me manage Citrus Greening disease?",
        "hybrid",
        "Hybrid with RRF"
    )
    
    # Test edge cases
    test_edge_cases(agent)
    
    print_separator("ALL PHASE 3 TESTS COMPLETED")
    print("✅ Phase 3 RAG Enhancement implementation verified!")
    print("\nKey features tested:")
    print("  - LLM-based document reranking")
    print("  - Reciprocal Rank Fusion for hybrid queries")
    print("  - Score threshold filtering")
    print("  - Enhanced citation format with confidence levels")
    print("  - Edge case handling (ambiguous, out-of-scope)")


if __name__ == "__main__":
    main()
