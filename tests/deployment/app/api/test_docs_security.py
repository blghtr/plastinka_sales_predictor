import pytest
from fastapi.testclient import TestClient

# The raw admin token is defined in tests/deployment/app/api/conftest.py
TEST_ADMIN_USERNAME = "admin"
TEST_ADMIN_PASSWORD = "test_admin_token"

class TestDocsSecurity:
    """Test suite for API documentation security using HTTP Basic Auth."""

    def test_docs_accessible_with_correct_credentials(self, client: TestClient):
        """Test that /docs is accessible with valid admin credentials."""
        response = client.get("/docs", auth=(TEST_ADMIN_USERNAME, TEST_ADMIN_PASSWORD))
        assert response.status_code == 200
        assert "Swagger UI" in response.text

    def test_redoc_accessible_with_correct_credentials(self, client: TestClient):
        """Test that /redoc is accessible with valid admin credentials."""
        response = client.get("/redoc", auth=(TEST_ADMIN_USERNAME, TEST_ADMIN_PASSWORD))
        assert response.status_code == 200
        assert "ReDoc" in response.text

    def test_docs_inaccessible_without_auth(self, client: TestClient):
        """Test that /docs is inaccessible without any authentication."""
        response = client.get("/docs")
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Basic"

    def test_docs_inaccessible_with_invalid_password(self, client: TestClient):
        """Test that /docs is inaccessible with an invalid password."""
        response = client.get("/docs", auth=(TEST_ADMIN_USERNAME, "wrong_password"))
        assert response.status_code == 401

    def test_docs_inaccessible_with_invalid_username(self, client: TestClient):
        """Test that /docs is inaccessible with an invalid username."""
        response = client.get("/docs", auth=("wrong_user", TEST_ADMIN_PASSWORD))
        assert response.status_code == 401

    def test_docs_inaccessible_with_bearer_token(self, client: TestClient):
        """Test that /docs is inaccessible with a Bearer token, as it should only accept Basic Auth."""
        response = client.get("/docs", headers={"Authorization": f"Bearer {TEST_ADMIN_PASSWORD}"})
        assert response.status_code == 401

    def test_docs_inaccessible_with_x_api_key(self, client: TestClient):
        """Test that /docs is inaccessible with an X-API-Key, as it should only accept Basic Auth."""
        response = client.get("/docs", headers={"X-API-Key": "some_key"})
        assert response.status_code == 401