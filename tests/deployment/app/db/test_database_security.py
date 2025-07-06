"""
Security tests for database module.
Tests for SQL injection prevention, path traversal protection, and data leakage prevention.
"""

import os
import tempfile
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from deployment.app.db.database import (
    DatabaseError,
    get_top_configs,
    delete_model_record_and_file,
    delete_models_by_ids,
    _is_path_safe,
    create_model_record,
    create_training_result,
    create_or_get_config,
)
from deployment.app.config import get_settings
from deployment.app.db.schema import SCHEMA_SQL

@pytest.fixture
def test_db_connection():
    """Create a temporary in-memory database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    # Initialize schema
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    
    yield conn
    
    conn.close()

class TestDatabaseSecurity:
    """Test suite for database security features."""

    def test_database_error_does_not_leak_params(self):
        """Test that DatabaseError does not expose query parameters in its string representation."""
        sensitive_params = ("password123", "secret_key")
        error = DatabaseError(
            message="Test error",
            query="SELECT * FROM users WHERE password = ?",
            params=sensitive_params
        )
        
        error_str = str(error)
        assert "password123" not in error_str
        assert "secret_key" not in error_str
        assert str(error) == "Test error"

    def test_database_error_handles_no_params(self):
        """Test that DatabaseError properly handles case with no parameters."""
        error = DatabaseError(message="Test error")
        assert str(error) == "Test error"

    def test_path_safe_validation_prevents_traversal(self):
        """Test that _is_path_safe prevents path traversal attempts."""
        base_dir = "/app/data"
        
        # Test valid paths
        assert _is_path_safe(base_dir, "/app/data/file.txt")
        assert _is_path_safe(base_dir, "/app/data/subdir/file.txt")
        
        # Test path traversal attempts
        assert not _is_path_safe(base_dir, "/app/data/../secret.txt")
        assert not _is_path_safe(base_dir, "/app/data/subdir/../../secret.txt")
        assert not _is_path_safe(base_dir, "/etc/passwd")

    def test_path_safe_handles_invalid_paths(self):
        """Test that _is_path_safe properly handles invalid or malformed paths."""
        base_dir = "/app/data"
        
        assert not _is_path_safe(base_dir, None)
        assert not _is_path_safe(base_dir, "")
        assert not _is_path_safe(base_dir, "../../etc/passwd")
        assert not _is_path_safe(None, "/app/data/file.txt")

    def test_path_traversal_with_relative_paths(self):
        """Test path traversal prevention with relative paths."""
        base_dir = "./data"
        
        # Valid relative paths
        assert _is_path_safe(base_dir, "./data/file.txt")
        assert _is_path_safe(base_dir, "data/subdir/file.txt")
        
        # Invalid relative paths
        assert not _is_path_safe(base_dir, "../secret.txt")
        assert not _is_path_safe(base_dir, "./data/../secret.txt")

    def test_get_top_configs_prevents_sql_injection(self, test_db_connection):
        """Test that get_top_configs prevents SQL injection attempts."""
        malicious_metrics = [
            "val_MIC; DROP TABLE configs; --",
            "val_MIC' UNION SELECT * FROM configs; --",
            "val_MIC' OR '1'='1",
        ]
        
        for metric in malicious_metrics:
            with pytest.raises(ValueError) as exc_info:
                get_top_configs(metric_name=metric, connection=test_db_connection)
            assert "Invalid metric_name" in str(exc_info.value)

    def test_get_top_configs_allows_only_whitelisted_metrics(self, test_db_connection):
        """Test that get_top_configs only accepts whitelisted metrics."""
        # Valid metric should work
        get_top_configs(metric_name="val_MIC", connection=test_db_connection)
        
        # Invalid metrics should raise ValueError
        invalid_metrics = [
            "unknown_metric",
            "custom_metric",
            "metrics.val_MIC",  # Trying to access object properties
            "__proto__.val_MIC",  # Prototype pollution attempt
        ]
        
        for metric in invalid_metrics:
            with pytest.raises(ValueError) as exc_info:
                get_top_configs(metric_name=metric, connection=test_db_connection)
            assert "Invalid metric_name" in str(exc_info.value)

    def test_delete_model_prevents_path_traversal(self, test_db_connection):
        """Test that delete_model_record_and_file prevents path traversal in file deletion."""
        settings = get_settings()
        models_dir = settings.models_dir
        
        # Create a test model record with suspicious path
        model_id = "test_model"
        suspicious_path = str(Path(models_dir).parent / "sensitive_file.txt")
        
        # Create the model record
        create_model_record(
            model_id=model_id,
            job_id="test_job",
            model_path=suspicious_path,
            created_at="2024-01-01",
            connection=test_db_connection
        )
        
        # Create a dummy file
        with open(suspicious_path, "w") as f:
            f.write("sensitive data")
        
        try:
            # Attempt to delete the model
            delete_model_record_and_file(model_id, connection=test_db_connection)
            
            # The file outside models_dir should not be deleted
            assert os.path.exists(suspicious_path)
            
        finally:
            # Cleanup
            if os.path.exists(suspicious_path):
                os.remove(suspicious_path)

    def test_delete_models_by_ids_prevents_path_traversal(self, test_db_connection):
        """Test that delete_models_by_ids prevents path traversal in batch file deletion."""
        settings = get_settings()
        models_dir = settings.models_dir
        
        # Create test model records with suspicious paths
        model_records = [
            ("model1", str(Path(models_dir).parent / "sensitive1.txt")),
            ("model2", str(Path(models_dir).parent / "sensitive2.txt")),
        ]
        
        # Create the model records and dummy files
        for model_id, path in model_records:
            create_model_record(
                model_id=model_id,
                job_id="test_job",
                model_path=path,
                created_at="2024-01-01",
                connection=test_db_connection
            )
            
            with open(path, "w") as f:
                f.write(f"sensitive data for {model_id}")
        
        try:
            # Attempt to delete the models
            delete_models_by_ids([m[0] for m in model_records], connection=test_db_connection)
            
            # Files outside models_dir should not be deleted
            for _, path in model_records:
                assert os.path.exists(path)
                
        finally:
            # Cleanup
            for _, path in model_records:
                if os.path.exists(path):
                    os.remove(path)

    def test_security_logging_does_not_expose_sensitive_data(self, caplog):
        """Test that security-related logging doesn't expose sensitive data."""
        # Test DatabaseError logging
        with pytest.raises(ValueError):
            get_top_configs(metric_name="malicious_metric")

        # Check that sensitive data is not in the logs
        log_messages = [record.message for record in caplog.records]
        for message in log_messages:
            assert "malicious_metric" not in message
            assert "Invalid metric name provided" in message

    def test_metric_validation_comprehensive(self, test_db_connection):
        """Comprehensive test of metric name validation across functions."""
        invalid_metrics = [
            # SQL Injection attempts
            "metric; DROP TABLE configs;",
            "metric' OR '1'='1",
            "metric' UNION SELECT * FROM users--",
            # NoSQL Injection attempts
            '{"$gt": ""}',
            '{"$where": "function() { return true }"}',
            # Path traversal attempts
            "../../../etc/passwd",
            "..\\..\\Windows\\System32",
            # Other malicious inputs
            "<script>alert(1)</script>",
            "metric\x00hidden",
            "metric\n\rinjection",
        ]
        
        for metric in invalid_metrics:
            with pytest.raises(ValueError) as exc_info:
                get_top_configs(metric_name=metric, connection=test_db_connection)
            assert "Invalid metric_name" in str(exc_info.value) 