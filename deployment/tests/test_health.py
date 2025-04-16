from fastapi.testclient import TestClient


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Check response format
    assert "status" in data
    assert "components" in data
    assert "timestamp" in data
    
    # Check components
    assert "api" in data["components"]
    assert "database" in data["components"]
    
    # Check status values
    assert data["components"]["api"] == "healthy"
    assert data["components"]["database"] == "healthy"
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    
    # Check response format
    assert "message" in data
    assert "docs" in data
    assert "version" in data
    
    # Check values
    assert data["message"] == "Plastinka Sales Predictor API"
    assert data["docs"] == "/docs"
    assert data["version"] == "0.1.0" 