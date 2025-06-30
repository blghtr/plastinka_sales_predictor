
def test_root_endpoint(client):
    """Test the root endpoint returns a welcome message."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Plastinka Sales Predictor API" in data["message"]
    assert "version" in data
