"""
NEW TESTING ARCHITECTURE: ML/DataSphere Compatible Fixtures

This file replaces session-scoped pyfakefs anti-patterns with proper testing infrastructure:
- Real filesystem with temporary directories for DataSphere tests
- Function-scoped pyfakefs ONLY for pure file operations
- No session-scoped filesystem mocking that blocks ML frameworks

Migration from: tests/deployment/app/services/conftest.py (old problematic version)
To: This new architecture that supports PyTorch, DataSphere SDK, and ML frameworks
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)

# Global mocks registry for session optimization without scope mismatch
_GLOBAL_MOCKS_REGISTRY = {}

# =================================================================================
# NEW ARCHITECTURE: Real Filesystem + Temporary Directories
# =================================================================================


# SQLite fixtures removed as part of PostgreSQL migration.
# Use PostgreSQL fixtures from tests/deployment/app/db/conftest.py instead:
# - postgres_pool: PostgreSQL connection pool
# - test_db_schema: Applies PostgreSQL schema before each test
# - dal: Async DataAccessLayer instance with PostgreSQL pool


@pytest.fixture
def temp_workspace():
    """
    Real filesystem with temporary directories for DataSphere tests.

    COMPATIBILITY:
    - ✅ Works with PyTorch, DataSphere SDK, ML frameworks
    - ✅ No import conflicts during pytest collection
    - ✅ Cross-platform (Windows, Linux, macOS)

    PERFORMANCE:
    - Fast setup/teardown
    - Isolated between tests
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = {
            "temp_dir": temp_dir,
            "input_dir": os.path.join(temp_dir, "datasphere_input"),
            "output_dir": os.path.join(temp_dir, "datasphere_output"),
            "models_dir": os.path.join(temp_dir, "models"),
            "logs_dir": os.path.join(temp_dir, "logs"),
            "job_dir": os.path.join(temp_dir, "datasphere_jobs"),
            "config_path": os.path.join(temp_dir, "datasphere_jobs", "config.yaml"),
        }

        # Create all directories
        for key, dir_path in workspace.items():
            if key != "temp_dir" and key != "config_path":
                os.makedirs(dir_path, exist_ok=True)
            elif key == "config_path":
                os.makedirs(os.path.dirname(dir_path), exist_ok=True)
                with open(dir_path, "w") as f:
                    f.write(
                        "# Fake DataSphere job config\nname: test_job\ndesc: Test job\ncmd: python -m test"
                    )
        # Additionally, create train/ and tune/ subdirectories with stub config.yaml
        for sub in ["train", "tune"]:
            subdir = os.path.join(workspace["job_dir"], sub)
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, "config.yaml"), "w") as f:
                f.write("name: test_job\ndesc: Test job\ncmd: python -m test")

        yield workspace


@pytest.fixture(scope="session")
def session_monkeypatch():
    """Session-scoped monkeypatch for performance optimization."""
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture
def mock_datasphere_env(temp_workspace, monkeypatch, dal):
    """
    Replacement for mock_service_env without session-scoped pyfakefs.

    ARCHITECTURE CHANGES:
    - Uses real filesystem with temp directories
    - No pyfakefs conflicts with ML frameworks
    - Function-scoped for proper test isolation
    - Compatible with DataSphere SDK imports
    """
    mocks = {}

    # --- Environment Variables ---
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("CONFIG_PATH", "configs/test_config.yml")
    monkeypatch.setenv("SECRET_PATH", "configs/test_secret.yml")
    monkeypatch.setenv("AUTO_SELECT_BEST_MODEL", "false")
    monkeypatch.setenv("DATASPHERE_MAX_POLLS", "3")
    monkeypatch.setenv("DATASPHERE_POLL_INTERVAL", "0.1")

    # --- Mock Settings with Real Paths ---
    mock_settings = MagicMock()
    mock_settings.datasphere_input_dir = temp_workspace["input_dir"]
    mock_settings.datasphere_output_dir = temp_workspace["output_dir"]
    mock_settings.datasphere_job_config_path = temp_workspace["config_path"]
    mock_settings.datasphere_job_dir = temp_workspace["job_dir"]
    mock_settings.models_dir = temp_workspace["models_dir"]
    mock_settings.project_root_dir = temp_workspace["temp_dir"]
    mock_settings.datasphere_job_train_dir = os.path.join(temp_workspace["job_dir"], "train")
    mock_settings.datasphere_job_tune_dir = os.path.join(temp_workspace["job_dir"], "tune")

    # DataSphere api_client settings
    mock_settings.datasphere = MagicMock()
    mock_settings.datasphere.project_id = "test-project-id-new-arch"
    mock_settings.datasphere.folder_id = "test-folder-id-new-arch"
    mock_settings.datasphere.max_polls = 10 # Increased from 3 to allow full status sequence to complete
    mock_settings.datasphere.poll_interval = 0.1
    mock_settings.datasphere.client_submit_timeout_seconds = 60
    mock_settings.datasphere.client_status_timeout_seconds = 30
    mock_settings.datasphere.client_download_timeout_seconds = 600
    mock_settings.datasphere.client_init_timeout_seconds = 60
    mock_settings.datasphere.client_cancel_timeout_seconds = 60
    mock_settings.datasphere.api_client = {
        "project_id": "test-project",
        "folder_id": "test-folder",
        "oauth_token": "test-token",
    }

    # Explicitly mock db settings
    mock_settings.db = MagicMock()
    mock_settings.db.database_busy_timeout = 5000  # Set to a default integer value

    # Set max_models_to_keep to an integer value to avoid TypeError
    mock_settings.max_models_to_keep = 5

    mocks["settings"] = mock_settings

    # --- Patch get_settings function ---
    monkeypatch.setattr("deployment.app.config.get_settings", lambda: mock_settings)
    # CRITICAL: Also patch get_settings within the database module itself

    # Now, apply our mock to the imports
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.get_settings", lambda: mock_settings
    )
    monkeypatch.setattr("deployment.app.config.get_settings", lambda: mock_settings)

    # --- DataSphere api_client Mock ---
    mock_client = MagicMock()
    mock_client.submit_job.return_value = "ds-job-default-new-arch"
    mock_client.get_job_status.return_value = "COMPLETED"

    def default_download_side_effect(ds_job_id, results_dir, **kwargs):
        os.makedirs(results_dir, exist_ok=True)
        # Create result files in REAL filesystem
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            f.write('{"val_loss": 0.1, "mape": 10.5}')
        with open(os.path.join(results_dir, "model.onnx"), "w") as f:
            f.write("fake onnx model data")
        with open(os.path.join(results_dir, "predictions.csv"), "w") as f:
            f.write("barcode,pred\n123456789012,10")

    mock_client.download_job_results.side_effect = default_download_side_effect
    mocks["api_client"] = mock_client

    # NEW: Patch DataSphereClient where it's used in datasphere_service.py
    monkeypatch.setattr("deployment.app.services.datasphere_service.DataSphereClient", MagicMock(return_value=mock_client))

    # --- Database Mocks ---
    # Use PostgreSQL DAL from fixture (dal parameter)
    # The dal fixture provides an async DataAccessLayer with PostgreSQL pool

    # Always provide an AsyncMock for save_predictions_to_db for test compatibility (must be async)
    mocks["save_predictions_to_db"] = AsyncMock(return_value={"result_id": "mock_result_id", "predictions_count": 0})
    monkeypatch.setattr("deployment.app.services.datasphere_service.save_predictions_to_db", mocks["save_predictions_to_db"])

    # Mock the save_model_file_and_db function to also accept connection
    # This mock should be replaced by a real call if the function itself is under test.
    # For now, it returns the model_id directly to facilitate testing of the pipeline.
    mocks["save_model_file_and_db"] = AsyncMock(return_value="mock_model_id") # Return a dummy model_id
    monkeypatch.setattr("deployment.app.services.datasphere_service.save_model_file_and_db", mocks["save_model_file_and_db"])

    # Mock the update_job_status in datasphere_service to use our mocked db
    # Note: update_job_status is not available in datasphere_service module
    # The function is called through the DAL, so we don't need to patch it here

    mocks["mocked_dal"] = dal # Expose the DAL object for direct use
    yield mocks


# =================================================================================
# Reset Fixture for Test Isolation
# =================================================================================


@pytest.fixture(scope="function", autouse=True)
def reset_new_arch_mocks_between_tests():
    """
    Auto-use fixture to reset mocks between tests for new architecture.
    Only runs for tests using the new mock_datasphere_env fixture.
    """
    if not _GLOBAL_MOCKS_REGISTRY.get("_DATASPHERE_ENV_APPLIED"):
        yield
        return

    # Reset all mocks completely
    def reset_mock_completely(mock_obj):
        if isinstance(mock_obj, MagicMock):
            mock_obj.reset_mock()
            mock_obj.side_effect = None

    for mock in _GLOBAL_MOCKS_REGISTRY.values():
        reset_mock_completely(mock)

    # Restore default behaviors
    if "api_client" in _GLOBAL_MOCKS_REGISTRY:
        client_mock = _GLOBAL_MOCKS_REGISTRY["api_client"]
        client_mock.submit_job.return_value = "ds-job-default-new-arch"
        client_mock.get_job_status.return_value = "COMPLETED"

        def default_download_side_effect(ds_job_id, results_dir, **kwargs):
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "metrics.json"), "w") as f:
                f.write('{"val_loss": 0.1, "mape": 10.5}')
            with open(os.path.join(results_dir, "model.onnx"), "w") as f:
                f.write("fake onnx model data")
            with open(os.path.join(results_dir, "predictions.csv"), "w") as f:
                f.write("barcode,pred\n123456789012,10")

        client_mock.download_job_results.side_effect = default_download_side_effect

    yield


# =================================================================================
# Preserved Helper Functions (from original conftest.py)
# =================================================================================


def create_training_params(base_params=None):
    """
    Creates a complete TrainingConfig object with all required fields.

    Args:
        base_params: Optional dictionary with parameters to use as a base

    Returns:
        A valid TrainingConfig object
    """
    base_params = base_params or {}

    model_config = ModelConfig(
        num_encoder_layers=3,
        num_decoder_layers=2,
        decoder_output_dim=128,
        temporal_width_past=12,
        temporal_width_future=6,
        temporal_hidden_size_past=64,
        temporal_hidden_size_future=64,
        temporal_decoder_hidden=128,
        batch_size=base_params.get("batch_size", 32),
        dropout=base_params.get("dropout", 0.2),
        use_reversible_instance_norm=True,
        use_layer_norm=True,
    )

    optimizer_config = OptimizerConfig(
        lr=base_params.get("learning_rate", 0.001), weight_decay=0.0001
    )

    lr_shed_config = LRSchedulerConfig(T_0=10, T_mult=2)

    train_ds_config = TrainingDatasetConfig(alpha=0.05, span=12)

    return TrainingConfig(
        nn_model_config=model_config,
        optimizer_config=optimizer_config,
        lr_shed_config=lr_shed_config,
        train_ds_config=train_ds_config,
        lags=12,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    )


# =================================================================================
# Individual Fixture Alternatives (for gradual migration)
# =================================================================================


@pytest.fixture
def mock_asyncio_sleep():
    """Мокирует asyncio.sleep для ускорения тестов."""
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep


@pytest.fixture
def mock_datasphere_client():
    """Мокирует DataSphereClient для изоляции от реальных вызовов API."""
    mock_client_instance = MagicMock()
    mock_client_instance.submit_job = MagicMock(return_value="ds-job-id-123")
    mock_client_instance.get_job_status.side_effect = [
        "RUNNING",
        "RUNNING",
        "COMPLETED",
    ]
    mock_client_instance.download_job_results.return_value = None

    with patch(
        "deployment.app.services.datasphere_service.DataSphereClient",
        return_value=mock_client_instance,
    ) as mock_client:
        yield mock_client


@pytest.fixture
def mock_get_datasets():
    """Мокирует get_datasets, чтобы не генерировать реальные датасеты."""
    with patch(
        "deployment.app.services.datasphere_service.get_datasets",
        return_value=({'train': 1}, {'val': 1}),
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_save_model():
    """Мокирует save_model_file_and_db, чтобы не сохранять реальные файлы."""
    with patch(
        "deployment.app.services.datasphere_service.save_model_file_and_db",
        return_value="model-id-123",
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_save_predictions():
    """Мокирует save_predictions_to_db, чтобы не записывать в БД."""
    with patch(
        "deployment.app.services.datasphere_service.save_predictions_to_db",
        return_value="prediction-result-id-123",
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_update_job_status():
    """Мокирует update_job_status для отслеживания вызовов."""
    with patch(
        "deployment.app.services.datasphere_service.update_job_status"
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_get_job():
    """Мокирует get_job для возврата тестовых данных."""
    with patch(
        "deployment.app.services.datasphere_service.get_job",
        return_value={"job_id": "job-id-123", "status": "pending"},
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_create_model_record():
    """Мокирует create_model_record для отслеживания вызовов."""
    with patch(
        "deployment.app.services.datasphere_service.create_model_record"
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_get_active_model():
    """Мокирует get_active_model для возврата тестовых данных."""
    with patch(
        "deployment.app.services.datasphere_service.get_active_model",
        return_value={"model_id": "active-model-id", "model_path": "/path/to/model"},
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_get_active_config():
    """Мокирует get_active_config для возврата тестовых данных."""
    with patch(
        "deployment.app.services.datasphere_service.get_active_config",
        return_value={"config_id": "active-config-id", "parameters": {}},
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_create_training_result():
    """Мокирует create_training_result для отслеживания вызовов."""
    with patch(
        "deployment.app.services.datasphere_service.create_training_result",
        return_value="training-result-id-123",
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_create_prediction_result():
    """Мокирует create_prediction_result для отслеживания вызовов."""
    with patch(
        "deployment.app.services.datasphere_service.create_prediction_result",
        return_value="prediction-result-id-123",
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_get_effective_config():
    """Мокирует get_effective_config для возврата тестовых данных."""
    with patch(
        "deployment.app.services.datasphere_service.get_effective_config",
        return_value={"config_id": "effective-config-id", "parameters": {}},
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_auto_activate_best_model():
    """Мокирует auto_activate_best_model_if_enabled."""
    with patch(
        "deployment.app.services.datasphere_service.auto_activate_best_model_if_enabled"
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_auto_activate_best_config():
    """Мокирует auto_activate_best_config_if_enabled."""
    with patch(
        "deployment.app.services.datasphere_service.auto_activate_best_config_if_enabled"
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_dal(mocker):
    """
    Creates a comprehensive mock for the DataAccessLayer.
    This mock can be used in service-level tests to isolate them from the database.
    """
    dal_mock = mocker.MagicMock()

    # Mock methods that return data
    dal_mock.get_job.return_value = {"job_id": "mock-job-id", "status": "pending"}
    dal_mock.get_active_model.return_value = {"model_id": "active-model", "model_path": "/fake/path"}
    dal_mock.get_active_config.return_value = {"config_id": "active-config", "config": {}}
    dal_mock.get_prediction_results_by_month.return_value = [
        {"job_id": "mock-job-id", "result_id": "pred-res-1"}
    ]
    dal_mock.get_predictions_for_jobs.return_value = [
        {"barcode": "123", "artist": "Test Artist", "album": "Test Album", "quantile_50": 10}
    ]

    # Mock methods that perform actions (return None or an ID)
    dal_mock.create_job.return_value = "new-mock-job-id"
    dal_mock.update_job_status.return_value = None
    dal_mock.create_model_record.return_value = None
    dal_mock.create_prediction_result.return_value = "new-pred-res-id"
    dal_mock.create_training_result.return_value = "new-train-res-id"

    return dal_mock
