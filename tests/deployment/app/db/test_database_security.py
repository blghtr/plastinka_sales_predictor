"""
Security tests for database module.
Tests for SQL injection prevention, path traversal protection, and data leakage prevention.
Updated for PostgreSQL/async DAL.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from deployment.app.config import get_settings
from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.utils import _is_path_safe

class TestDatabaseSecurity:
    """Test suite for database security features."""

    def test_database_error_does_not_leak_params(self):
        """Test that DatabaseError does not expose query parameters."""
        sensitive_params = ("password123", "secret_key")
        error = DatabaseError(
            message="Test error",
            query="SELECT * FROM users WHERE password = $1",
            params=sensitive_params
        )

        error_str = str(error)
        assert "password123" not in error_str
        assert "secret_key" not in error_str
        assert "Test error" in error_str

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

    @pytest.mark.asyncio
    async def test_get_top_configs_prevents_sql_injection(self, dal):
        """Test that get_top_configs prevents SQL injection attempts."""
        malicious_metrics = [
            "val_MIC; DROP TABLE configs; --",
            "val_MIC' UNION SELECT * FROM configs; --",
            "val_MIC' OR '1'='1",
        ]

        for metric in malicious_metrics:
            with pytest.raises(ValueError) as exc_info:
                await dal.get_top_configs(metric_name=metric, limit=10)
            assert "Invalid metric_name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_top_configs_allows_only_whitelisted_metrics(self, dal):
        """Test that get_top_configs only accepts whitelisted metrics."""
        # Create a config first
        await dal.create_or_get_config({"test": "config"}, is_active=False)
        
        # Valid metric should work
        await dal.get_top_configs(metric_name="val_MIC", limit=10)

        # Invalid metrics should raise ValueError
        invalid_metrics = [
            "unknown_metric",
            "custom_metric",
            "metrics.val_MIC",  # Trying to access object properties
            "__proto__.val_MIC",  # Prototype pollution attempt
        ]

        for metric in invalid_metrics:
            with pytest.raises(ValueError) as exc_info:
                await dal.get_top_configs(metric_name=metric, limit=10)
            assert "Invalid metric_name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_model_prevents_path_traversal(self, dal, tmp_path):
        """Test that delete_model_record_and_file prevents path traversal."""
        settings = get_settings()
        models_dir = Path(settings.models_dir) if hasattr(settings, 'models_dir') else tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create a test job first
        job_id = await dal.create_job(job_type="training", parameters={})
        
        # Create a test model record with suspicious path
        model_id = "test_model"
        suspicious_path = str(models_dir.parent / "sensitive_file.txt")

        # Create the model record
        from datetime import date
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path=suspicious_path,
            created_at=date(2024, 1, 1),
            is_active=False,
        )

        # Create a dummy file
        with open(suspicious_path, "w") as f:
            f.write("sensitive data")

        try:
            # Attempt to delete the model
            await dal.delete_model_record_and_file(model_id)

            # The file outside models_dir should not be deleted
            assert os.path.exists(suspicious_path)

        finally:
            # Cleanup
            if os.path.exists(suspicious_path):
                os.remove(suspicious_path)

    @pytest.mark.asyncio
    async def test_delete_models_by_ids_prevents_path_traversal(self, dal, tmp_path):
        """Test that delete_models_by_ids prevents path traversal."""
        settings = get_settings()
        models_dir = Path(settings.models_dir) if hasattr(settings, 'models_dir') else tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create a test job first
        job_id = await dal.create_job(job_type="training", parameters={})

        # Create test model records with suspicious paths
        model_records = [
            ("model1", str(models_dir.parent / "sensitive1.txt")),
            ("model2", str(models_dir.parent / "sensitive2.txt")),
        ]

        # Create the model records and dummy files
        from datetime import date
        for model_id, path in model_records:
            await dal.create_model_record(
                model_id=model_id,
                job_id=job_id,
                model_path=path,
                created_at=date(2024, 1, 1),
                is_active=False,
            )

            with open(path, "w") as f:
                f.write(f"sensitive data for {model_id}")

        try:
            # Attempt to delete the models
            await dal.delete_models_by_ids([m[0] for m in model_records])

            # Files outside models_dir should not be deleted
            for _, path in model_records:
                assert os.path.exists(path)

        finally:
            # Cleanup
            for _, path in model_records:
                if os.path.exists(path):
                    os.remove(path)

    @pytest.mark.asyncio
    async def test_security_logging_does_not_expose_sensitive_data(self, dal, caplog):
        """Test that security-related logging doesn't expose sensitive data."""
        # Test DatabaseError logging
        with pytest.raises(ValueError):
            await dal.get_top_configs(metric_name="malicious_metric", limit=10)

        # Check that sensitive data is not in the logs
        log_messages = [record.message for record in caplog.records]
        for message in log_messages:
            assert "malicious_metric" not in message

    @pytest.mark.asyncio
    async def test_metric_validation_comprehensive(self, dal):
        """Comprehensive test of metric name validation."""
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
                await dal.get_top_configs(metric_name=metric, limit=10)
            assert "Invalid metric_name" in str(exc_info.value)

    # File permissions test removed - PostgreSQL uses connection pooling,
    # file permissions are managed by PostgreSQL server, not application code
