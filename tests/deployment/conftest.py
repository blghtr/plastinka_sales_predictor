import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import sqlite3
from pathlib import Path

from deployment.app.main import app


@pytest.fixture
def test_db_path():
    """Create a temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    
    # Initialize test database
    conn = sqlite3.connect(db_path)
    with open("deployment/app/db/schema.py") as f:
        schema_content = f.read()
        start_index = schema_content.find("SCHEMA_SQL = \"\"\"")
        end_index = schema_content.find("\"\"\"", start_index + 15)
        schema_sql = schema_content[start_index + 15:end_index]
        conn.executescript(schema_sql)
    conn.close()
    
    yield db_path
    
    # Clean up
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(monkeypatch, test_db_path):
    """Create a FastAPI TestClient with a test database."""
    # Mock the database path
    monkeypatch.setattr("app.db.schema.init_db", lambda: None)
    monkeypatch.setattr("app.main.init_db", lambda: None)
    
    # Patch database connection in health check
    def mock_connect(*args, **kwargs):
        return sqlite3.connect(test_db_path)
    
    monkeypatch.setattr("sqlite3.connect", mock_connect)
    
    # Return test client
    with TestClient(app) as client:
        yield client 