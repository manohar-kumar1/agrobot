"""
Quick test to verify vector store functionality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService


def main():
    """Test vector store retrieval"""
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)
    vector_store.initialize()

    # Get collection stats
    print("\n" + "=" * 60)
    print("Vector Store Collection Stats")
    print("=" * 60)
    stats = vector_store.get_collection_stats()
    for collection, count in stats.items():
        print(f"{collection}: {count} documents")

    # Test disease query
    print("\n" + "=" * 60)
    print("Test Query 1: Disease Intent")
    print("=" * 60)
    query1 = "What are the symptoms of Citrus Canker?"
    results1 = vector_store.similarity_search(query1, "disease", top_k=3)

    print(f"\nQuery: {query1}")
    print(f"Results: {len(results1)} documents")
    for i, doc in enumerate(results1, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source_file', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")

    # Test scheme query
    print("\n" + "=" * 60)
    print("Test Query 2: Scheme Intent")
    print("=" * 60)
    query2 = "What subsidies are available for irrigation?"
    results2 = vector_store.similarity_search(query2, "scheme", top_k=3)

    print(f"\nQuery: {query2}")
    print(f"Results: {len(results2)} documents")
    for i, doc in enumerate(results2, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source_file', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")

    print("\n" + "=" * 60)
    print("Vector store test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
