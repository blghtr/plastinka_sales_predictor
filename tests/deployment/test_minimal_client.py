"""
Minimal test file that uses the client fixture from conftest.py to check if that causes hanging.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

def test_minimal_with_client_fixture(client):
    """A minimal test that uses the client fixture from conftest.py."""
    print("Running minimal test with client fixture")
    print(f"Got client fixture: {client}")
    
    # Make a simple request to the client
    response = client.get("/health")
    print(f"Response status code: {response.status_code}")
    
    assert True 