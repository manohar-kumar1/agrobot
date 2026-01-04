"""
Simple script to test the API endpoints.
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())


def test_query(question: str):
    """Test query endpoint"""
    payload = {
        "question": question,
        "user_id": "test_user"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        json=payload
    )
    print(f"\nQuery: {question}")
    print("Response:", json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("Testing Agrobot API...\n")
    
    # Test health
    test_health()
    
    # Test queries
    test_query("What are the symptoms of citrus canker?")
    test_query("What government schemes are available for farmers?")
    test_query("How to treat citrus greening disease and get subsidy?")
