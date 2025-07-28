#!/usr/bin/env python3
"""
Simple test script to verify the Data Analyst Agent API functionality
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app
from fastapi.testclient import TestClient

def test_api():
    """Test the API endpoints"""
    client = TestClient(app)
    
    print("🧪 Testing Data Analyst Agent API...")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Health check: {data['message']}")
    
    # Test 2: Detailed health check
    print("\n2. Testing detailed health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    print(f"✅ Detailed health: {health_data['status']}")
    
    # Test 3: API endpoint with unsupported task
    print("\n3. Testing API with unsupported task...")
    response = client.post("/api/", content="test unsupported task")
    assert response.status_code == 501
    api_data = response.json()
    print(f"✅ Unsupported task response: {api_data['message']}")
    
    # Test 4: Empty request
    print("\n4. Testing empty request...")
    response = client.post("/api/", content="")
    assert response.status_code == 400
    error_data = response.json()
    print(f"✅ Empty request handled: {error_data['detail']}")
    
    print("\n🎉 All basic tests passed!")
    print("\n📝 Manual testing commands:")
    print("For Wikipedia analysis:")
    print('curl -X POST "http://localhost:8000/api/" -H "Content-Type: text/plain" -d "Analyze wikipedia.org/wiki/list_of_highest-grossing_films data"')
    print("\nFor High Court analysis:")
    print('curl -X POST "http://localhost:8000/api/" -H "Content-Type: text/plain" -d "Analyze indian high court judgement dataset"')
    
    return True

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)
