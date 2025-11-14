"""
Regression tests for unique constraint violation retry behavior.

These tests ensure that unique constraint violations (PostgreSQL error code 23505)
are correctly identified as permanent errors and not retried.
"""

import pytest
import asyncpg

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query
from deployment.app.db.utils import _db_should_give_up


class TestUniqueViolationRetryRegression:
    """Regression tests to prevent retrying on unique constraint violations."""

    @pytest.mark.asyncio
    async def test_unique_violation_error_code_23505_not_retried(self, dal):
        """Test that PostgreSQL error code 23505 (unique_violation) is not retried."""
        async with dal._pool.acquire() as conn:
            # Create table with unique constraint and clean it up
            await conn.execute("DROP TABLE IF EXISTS test_unique_regression")
            await conn.execute(
                "CREATE TABLE test_unique_regression "
                "(id BIGSERIAL PRIMARY KEY, name TEXT UNIQUE)"
            )
            
            # Insert first record
            await execute_query(
                "INSERT INTO test_unique_regression (name) VALUES ($1)",
                conn,
                ("unique-test",)
            )
            
            # Try to insert duplicate - should fail immediately without retries
            # This verifies that error code 23505 is recognized as permanent
            with pytest.raises(DatabaseError) as exc_info:
                await execute_query(
                    "INSERT INTO test_unique_regression (name) VALUES ($1)",
                    conn,
                    ("unique-test",)
                )
            
            # Verify error is recognized as non-retryable
            assert _db_should_give_up(exc_info.value) is True
            
            # Clean up
            await conn.execute("DROP TABLE test_unique_regression")

    @pytest.mark.asyncio
    async def test_unique_violation_error_code_23xxx_not_retried(self, dal):
        """Test that all PostgreSQL error codes 23xxx (integrity violations) are not retried."""
        async with dal._pool.acquire() as conn:
            # Create table with unique constraint and clean it up
            await conn.execute("DROP TABLE IF EXISTS test_unique_23xxx")
            await conn.execute(
                "CREATE TABLE test_unique_23xxx "
                "(id BIGSERIAL PRIMARY KEY, name TEXT UNIQUE)"
            )
            
            # Insert first record
            await execute_query(
                "INSERT INTO test_unique_23xxx (name) VALUES ($1)",
                conn,
                ("test-23xxx",)
            )
            
            # Try to insert duplicate - should fail immediately
            with pytest.raises(DatabaseError) as exc_info:
                await execute_query(
                    "INSERT INTO test_unique_23xxx (name) VALUES ($1)",
                    conn,
                    ("test-23xxx",)
                )
            
            # Verify _db_should_give_up recognizes 23xxx codes as permanent
            assert _db_should_give_up(exc_info.value) is True
            
            # Verify original error has sqlstate starting with "23"
            orig_error = getattr(exc_info.value, "original_error", None)
            if isinstance(orig_error, asyncpg.exceptions.PostgresError):
                error_code = getattr(orig_error, "sqlstate", None)
                if error_code:
                    assert error_code.startswith("23"), f"Expected 23xxx error code, got {error_code}"
            
            # Clean up
            await conn.execute("DROP TABLE test_unique_23xxx")

