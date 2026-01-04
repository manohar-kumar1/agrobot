"""
Test script for Phase 2: Intent Detection & RAG Pipeline.
Tests all three intent types: disease, scheme, and hybrid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.rag_agent import RAGAgent


def print_separator(title: str):
    """Print a visual separator"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_query(agent: RAGAgent, query: str, expected_intent: str):
    """Test a single query and print results"""
    print(f"\n📝 Query: {query}")
    print(f"   Expected Intent: {expected_intent}")
    print("-" * 50)

    response = agent.process_query(query)

    print(f"✅ Detected Intent: {response.intent.value}")
    print(f"📊 Confidence: {response.confidence:.2%}")
    print(f"\n📄 Answer (first 500 chars):")
    print(f"   {response.answer[:500]}...")

    print(f"\n📚 Sources ({len(response.sources)} found):")
    for source in response.sources[:3]:
        print(
            f"   - {source.document} (Page {source.page}, Score: {source.relevance_score})"
        )

    # Check if intent matches expected
    match = "✅ MATCH" if response.intent.value == expected_intent else "❌ MISMATCH"
    print(f"\n{match}")

    return response


def main():
    """Run all tests"""
    print_separator("Initializing RAG Agent")
    agent = RAGAgent()
    print("✅ RAG Agent initialized successfully")
    print(f"\n{agent.get_workflow_graph()}")

    # Test Disease Intent
    print_separator("TEST 1: Disease Intent")
    test_query(
        agent,
        "What are the symptoms of Citrus Canker and how do I treat it?",
        "disease",
    )

    # Test another Disease query
    print_separator("TEST 2: Disease Intent (Pest)")
    test_query(
        agent,
        "My citrus leaves are showing yellow patches. What could be wrong?",
        "disease",
    )

    # Test Scheme Intent
    print_separator("TEST 3: Scheme Intent")
    test_query(
        agent, "What government subsidies are available for drip irrigation?", "scheme"
    )

    # Test another Scheme query
    print_separator("TEST 4: Scheme Intent (Financial)")
    test_query(agent, "How can I apply for PMKSY scheme for my citrus farm?", "scheme")

    # Test Hybrid Intent
    print_separator("TEST 5: Hybrid Intent")
    test_query(
        agent,
        "What government schemes can help me manage Citrus Greening disease?",
        "hybrid",
    )

    # Test another Hybrid query
    print_separator("TEST 6: Hybrid Intent (Equipment + Funding)")
    test_query(
        agent,
        "I need help with pest control equipment and funding. What options do I have?",
        "hybrid",
    )

    print_separator("ALL TESTS COMPLETED")
    print("✅ Phase 2 implementation test completed!")


if __name__ == "__main__":
    main()
