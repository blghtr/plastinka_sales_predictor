def test_root_endpoint(api_client):
    """Test the root endpoint returns a welcome message."""
    response = api_client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Plastinka Sales Predictor API" in data["message"]
    assert "version" in data
