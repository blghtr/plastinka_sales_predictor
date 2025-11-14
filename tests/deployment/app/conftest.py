"""
Фикстуры и моки для тестов deployment/app

- PostgreSQL fixtures are available in tests/deployment/app/db/conftest.py:
  - postgres_pool: PostgreSQL connection pool
  - test_db_schema: Applies PostgreSQL schema before each test
  - dal: Async DataAccessLayer instance with PostgreSQL pool
- mocked_db — предоставляет моки для всех основных операций с БД, гарантирует передачу connection между слоями.
- Все фикстуры имеют scope='function' для предотвращения state leakage.
- pyfakefs и патчинг aiofiles используются только в тестах, где это необходимо.
- Все временные файлы и директории удаляются после теста.
- Моки DataSphere и DAL возвращают структуры, строго соответствующие контракту теста.
"""

import gc
import os
import shutil
import tempfile
import uuid
from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from passlib.context import CryptContext

from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
from deployment.app.db.feature_storage import SQLFeatureStore
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)


def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@pytest.fixture
def mock_active_config():
    """
    Создает фиктивный набор активных параметров для тестирования.
    """
    valid_training_params = {
        "config_id": "default-active-config-id",
        "parameters": {
            "model_config": {
                "num_encoder_layers": 3,
                "num_decoder_layers": 2,
                "decoder_output_dim": 128,
                "temporal_width_past": 12,
                "temporal_width_future": 6,
                "temporal_hidden_size_past": 64,
                "temporal_hidden_size_future": 64,
                "temporal_decoder_hidden": 128,
                "batch_size": 32,
                "dropout": 0.2,
                "use_reversible_instance_norm": True,
                "use_layer_norm": True,
            },
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
            "model_id": "default_model",
            "additional_params": {
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2022-12-31",
            },
        },
    }
    return valid_training_params


@pytest.fixture
def sample_model_data():
    """
    Sample model data for testing
    """
    return {
        "model_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "model_path": "/path/to/model.onnx",
        "created_at": datetime.now(),
        "metadata": {"framework": "pytorch", "version": "1.9.0"},
    }


@pytest.fixture
def sample_config(create_training_params_fn):
    """
    Sample config for testing, now returns a valid nested TrainingConfig structure.
    """
    # Use the helper from the root conftest to create a valid, nested config
    return create_training_params_fn().model_dump(mode="json")


# SQLite fixtures removed as part of PostgreSQL migration.
# Use PostgreSQL fixtures from tests/deployment/app/db/conftest.py instead:
# - postgres_pool: PostgreSQL connection pool
# - test_db_schema: Applies PostgreSQL schema before each test
# - dal: Async DataAccessLayer instance with PostgreSQL pool


# file_based_db fixture removed as part of PostgreSQL migration.
# Use PostgreSQL fixtures from tests/deployment/app/db/conftest.py instead:
# - postgres_pool: PostgreSQL connection pool
# - test_db_schema: Applies PostgreSQL schema before each test
# - dal: Async DataAccessLayer instance with PostgreSQL pool


@pytest.fixture
def sample_job_data():
    """
    Sample job data for testing
    """
    return {
        "job_id": str(uuid.uuid4()),
        "job_type": "training",
        "parameters": {"batch_size": 32, "learning_rate": 0.001},
    }




# clean_retry_events_table fixture removed as part of PostgreSQL migration.
# The test_db_schema fixture in tests/deployment/app/db/conftest.py handles
# test isolation by dropping and recreating tables before each test.



@pytest.fixture(scope="session", autouse=True)
def set_session_db_path(session_monkeypatch, tmp_path_factory):
    """
    Session-scoped fixture to set a temporary file-based database path for AppSettings.
    Ensures all database interactions use an isolated DB for the entire test session.
    """

    from deployment.app.config import get_settings

    # Use a real temporary directory for the session
    temp_dir = tmp_path_factory.mktemp("session_data")
    temp_dir / "session_test.db"
    models_dir = temp_dir / "models"
    logs_dir = temp_dir / "logs"
    temp_upload_dir = temp_dir / "temp_uploads"

    # Create directories
    models_dir.mkdir()
    logs_dir.mkdir()
    temp_upload_dir.mkdir()

    # The DAL will handle schema initialization when it's instantiated with db_path

    # --- Create a REAL, fully-populated AppSettings instance ---
    # This avoids all the AttributeError problems from an incomplete MagicMock.
    # We override only the paths and critical values for testing.

    # Use monkeypatch to set environment variables that AppSettings will read
    session_monkeypatch.setenv("DATA_ROOT_DIR", str(temp_dir))
    session_monkeypatch.setenv("DB_FILENAME", "session_test.db")
    session_monkeypatch.setenv("API_ADMIN_API_KEY_HASH", CryptContext(schemes=["bcrypt"]).hash("test_admin_token"))
    session_monkeypatch.setenv("API_X_API_KEY_HASH", CryptContext(schemes=["bcrypt"]).hash("test_x_api_key_conftest"))
    session_monkeypatch.setenv("METRIC_THESH_FOR_HEALTH_CHECK", "0.5") # Add missing health check metric

    # Clear the get_settings cache to force re-reading with our new env vars
    get_settings.cache_clear()

    # The original AppSettings will now load with our temporary paths and test keys.
    # No need to mock the class itself.

    # Patch configure_logging to prevent FileNotFoundError during tests
    session_monkeypatch.setattr("deployment.app.logger_config.configure_logging", MagicMock())

    yield

    # Cleanup: clear the cache again after the session
    get_settings.cache_clear()



# temp_db and temp_db_with_data fixtures removed as part of PostgreSQL migration.
# PostgreSQL fixtures are defined in tests/deployment/app/db/conftest.py:
# - postgres_pool: PostgreSQL connection pool
# - test_db_schema: Applies PostgreSQL schema before each test
# - dal: Async DataAccessLayer instance with PostgreSQL pool
#
# For tests in subdirectories, we need to make dal available here.
# Pytest will automatically find conftest.py files in parent directories,
# but db/conftest.py is in a sibling directory, so we copy the fixture definitions here.
import pytest_asyncio
import asyncio
import os
import logging
from typing import AsyncGenerator
import asyncpg
from asyncpg import Pool

from deployment.app.db.schema_postgresql import SCHEMA_SQL

logger = logging.getLogger(__name__)

# Copy postgres_pool fixture from db/conftest.py
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def postgres_pool() -> AsyncGenerator[Pool, None]:
    """
    Create a PostgreSQL connection pool for testing.
    Copied from tests/deployment/app/db/conftest.py to make it available to subdirectories.
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

# Copy test_db_schema fixture from db/conftest.py
@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def test_db_schema(postgres_pool: Pool) -> None:
    """
    Clean up tables before each test for test isolation.
    Copied from tests/deployment/app/db/conftest.py to make it available to subdirectories.
    """
    async with postgres_pool.acquire() as conn:
        truncate_order = [
            "retry_events", "report_features", "job_submission_locks", "tuning_results",
            "report_results", "prediction_results", "training_results", "data_upload_results",
            "job_status_history", "jobs", "models", "configs", "processing_runs",
            "fact_predictions", "fact_stock_movement", "fact_sales", "dim_multiindex_mapping",
        ]
        for table in truncate_order:
            try:
                await conn.execute(f'TRUNCATE TABLE "{table}" CASCADE')
            except Exception as e:
                logger.warning(f"Failed to truncate table {table}: {e}")
        logger.debug("Test database tables truncated")

@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def dal(postgres_pool: Pool, test_db_schema) -> AsyncGenerator[DataAccessLayer, None]:
    """
    DataAccessLayer fixture for tests in deployment/app/* subdirectories.
    """
    user_context = UserContext(roles=[UserRoles.SYSTEM])
    dal_instance = DataAccessLayer(user_context=user_context, pool=postgres_pool)
    yield dal_instance

@pytest.fixture
async def temp_db_with_data(dal, sample_config):
    """
    Like temp_db, but pre-populates jobs, models, and configs. Provides all IDs needed for most test contracts.
    Updated for PostgreSQL/async DAL.
    """
    model_id = str(uuid.uuid4())
    job_for_model_id = str(uuid.uuid4())

    # Create two jobs
    job_id_1 = await dal.create_job(
        job_type="training",
        parameters={},
    )
    job_for_model_id = await dal.create_job(
        job_type="training",
        parameters={},
    )
    # Create a model for job_for_model_id
    await dal.create_model_record(
        model_id=model_id,
        job_id=job_for_model_id,
        model_path="/fake/path/model.onnx",
        created_at=datetime.now(),
        is_active=True,
    )
    # Create a config
    config_id = await dal.create_or_get_config(
        sample_config,
        is_active=True,
    )
    result = {
        "dal": dal,
        "job_id": job_id_1,
        "model_id": model_id,
        "job_for_model_id": job_for_model_id,
        "config_id": config_id,
    }
    yield result

# isolated_db_session and db_with_sales_data fixtures removed as part of PostgreSQL migration.
# Use PostgreSQL fixtures from tests/deployment/app/db/conftest.py instead:
# - postgres_pool: PostgreSQL connection pool
# - test_db_schema: Applies PostgreSQL schema before each test
# - dal: Async DataAccessLayer instance with PostgreSQL pool

# NOTE: All DB fixtures yield dicts. Use .get('conn') for DB operations, .get('db_path') for path-based tests.
