"""PostgreSQL test fixtures for database tests."""

import asyncio
import os
import logging
from typing import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio
from asyncpg import Pool

from deployment.app.db.connection import init_db_pool, close_db_pool
from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
from deployment.app.db.schema_postgresql import init_postgres_schema, SCHEMA_SQL

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def postgres_pool() -> AsyncGenerator[Pool, None]:
    """
    Create a PostgreSQL connection pool for testing.
    
    Uses environment variables for connection:
    - TEST_POSTGRES_HOST (default: localhost)
    - TEST_POSTGRES_PORT (default: 5432)
    - TEST_POSTGRES_DATABASE (default: plastinka_ml_test)
    - TEST_POSTGRES_USER (default: postgres)
    - TEST_POSTGRES_PASSWORD (default: postgres)
    
    Alternatively, can use testcontainers if available.
    """
    host = os.getenv("TEST_POSTGRES_HOST", "localhost")
    port = int(os.getenv("TEST_POSTGRES_PORT", "5432"))
    database = os.getenv("TEST_POSTGRES_DATABASE", "plastinka_ml_test")
    user = os.getenv("TEST_POSTGRES_USER", "postgres")
    password = os.getenv("TEST_POSTGRES_PASSWORD", "postgres")
    
    try:
        # Create a test database if it doesn't exist (only once)
        admin_pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database="postgres",  # Connect to default database
            user=user,
            password=password,
            min_size=1,
            max_size=1,
        )
        
        async with admin_pool.acquire() as conn:
            # Check if test database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )
            
            if not db_exists:
                # Create test database
                await conn.execute(f'CREATE DATABASE "{database}"')
                logger.info(f"Created test database: {database}")
        
        await admin_pool.close()
        
        # Create pool for test database
        pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=2,
            max_size=10,
        )
        
        # Apply schema once at session start
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
            logger.info("PostgreSQL schema applied at session start")
        
        logger.info(f"PostgreSQL test pool created: {database}")
        yield pool
        
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL test pool: {e}", exc_info=True)
        pytest.skip(f"PostgreSQL not available: {e}")
    finally:
        if 'pool' in locals():
            await pool.close()
            logger.info("PostgreSQL test pool closed")


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def test_db_schema(postgres_pool: Pool) -> None:
    """
    Clean up tables before each test for test isolation.
    
    Schema is already applied in postgres_pool fixture (session-scoped).
    This fixture only truncates tables to ensure clean state.
    """
    # Use async context manager to ensure proper connection cleanup
    async with postgres_pool.acquire() as conn:
        # Truncate all tables for test isolation
        truncate_order = [
            "retry_events",
            "report_features",
            "job_submission_locks",
            "tuning_results",
            "report_results",
            "prediction_results",
            "training_results",
            "data_upload_results",
            "job_status_history",
            "jobs",
            "models",
            "configs",
            "processing_runs",
            "fact_predictions",
            "fact_stock_movement",
            "fact_sales",
            "dim_multiindex_mapping",
        ]
        
        # Truncate tables (they should exist after schema is applied in postgres_pool)
        for table in truncate_order:
            try:
                await conn.execute(f'TRUNCATE TABLE "{table}" CASCADE')
            except Exception as e:
                logger.warning(f"Failed to truncate table {table}: {e}")
        
        logger.debug("Test database tables truncated")


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def dal(postgres_pool: Pool, test_db_schema: None) -> AsyncGenerator[DataAccessLayer, None]:
    """
    Create a DataAccessLayer instance with test database pool.
    
    Args:
        postgres_pool: PostgreSQL connection pool fixture
        test_db_schema: Schema application fixture (ensures clean state before test)
        
    Yields:
        DataAccessLayer instance configured for testing
    """
    user_context = UserContext(roles=[UserRoles.SYSTEM])
    dal = DataAccessLayer(user_context=user_context, pool=postgres_pool)
    
    yield dal


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def async_dal(postgres_pool: Pool) -> AsyncGenerator[DataAccessLayer, None]:
    """
    Alias for dal fixture for clarity in async tests.
    """
    user_context = UserContext(roles=[UserRoles.SYSTEM])
    dal_instance = DataAccessLayer(user_context=user_context, pool=postgres_pool)
    yield dal_instance

