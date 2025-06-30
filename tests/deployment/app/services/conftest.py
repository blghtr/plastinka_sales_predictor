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

# Global mocks registry for session optimization without scope mismatch
_GLOBAL_MOCKS_REGISTRY = {}

# Safe imports - these don't conflict with ML frameworks
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)

# =================================================================================
# NEW ARCHITECTURE: Real Filesystem + Temporary Directories
# =================================================================================


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
            "job_dir": os.path.join(temp_dir, "datasphere_job"),
            "config_path": os.path.join(temp_dir, "datasphere_job", "config.yaml"),
        }

        # Create all directories
        for key, dir_path in workspace.items():
            if key != "temp_dir" and key != "config_path":
                os.makedirs(dir_path, exist_ok=True)
            elif key == "config_path":
                os.makedirs(os.path.dirname(dir_path), exist_ok=True)
                with open(dir_path, "w") as f:
                    f.write(
                        "# Fake DataSphere job config\nname: test_job\ntype: python"
                    )

        yield workspace


@pytest.fixture
def file_operations_fs():
    """
    Function-scoped pyfakefs ONLY for testing pure file operations.

    USE CASES:
    - Unit tests for shutil operations
    - pathlib operations testing
    - File manipulation without ML dependencies

    DO NOT USE FOR:
    - DataSphere SDK tests
    - PyTorch model loading
    - Any tests that import ML frameworks
    """
    from pyfakefs.fake_filesystem_unittest import Patcher

    with Patcher() as patcher:
        yield patcher.fs


@pytest.fixture(scope="session")
def session_monkeypatch():
    """Session-scoped monkeypatch for performance optimization."""
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture
def mock_datasphere_env(temp_workspace, monkeypatch):
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

    # DataSphere client settings
    mock_settings.datasphere = MagicMock()
    mock_settings.datasphere.project_id = "test-project-id-new-arch"
    mock_settings.datasphere.folder_id = "test-folder-id-new-arch"
    mock_settings.datasphere.max_polls = 3
    mock_settings.datasphere.poll_interval = 0.1
    mock_settings.datasphere.client_submit_timeout_seconds = 60
    mock_settings.datasphere.client_status_timeout_seconds = 30
    mock_settings.datasphere.client_download_timeout_seconds = 600
    mock_settings.datasphere.client_init_timeout_seconds = 60
    mock_settings.datasphere.client_cancel_timeout_seconds = 60
    mock_settings.datasphere.client = {
        "project_id": "test-project",
        "folder_id": "test-folder",
        "oauth_token": "test-token",
    }

    mocks["settings"] = mock_settings

    # --- Patch get_settings function ---
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.get_settings", lambda: mock_settings
    )
    monkeypatch.setattr("deployment.app.config.get_settings", lambda: mock_settings)

    # --- DataSphere Client Mock ---
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
    mocks["client"] = mock_client

    # --- Database Mocks ---
    mock_db_conn = MagicMock()
    mock_db_conn.cursor.return_value = mock_db_conn
    mock_db_conn.execute.return_value = mock_db_conn
    mock_db_conn.executemany.return_value = mock_db_conn
    mock_db_conn.fetchone.return_value = {
        "id": 1,
        "multiindex_id": "test-multiindex-new",
    }
    mocks["db_connection"] = mock_db_conn

    mock_get_db_connection = MagicMock(return_value=mock_db_conn)
    mocks["get_db_connection"] = mock_get_db_connection

    # --- Database Function Mocks ---
    for func_name in [
        "get_or_create_multiindex_id",
        "update_job_status",
        "create_training_result",
        "execute_many",
        "save_predictions_to_db",
        "get_job",
        "create_model_record",
    ]:
        mock = MagicMock(name=func_name)
        mocks[func_name] = mock

    # --- DataSphere Service Function Mocks ---
    def mock_get_datasets_side_effect(output_dir, **kwargs):
        """Create dataset files in REAL filesystem."""
        os.makedirs(output_dir, exist_ok=True)
        for filename in ["features.pkl", "train.dill", "val.dill", "full.dill"]:
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(f"fake {filename} data")

    mock_get_datasets = MagicMock(side_effect=mock_get_datasets_side_effect)
    mocks["get_datasets"] = mock_get_datasets

    async def mock_archive_side_effect(job_id, input_dir, archive_dir=None):
        """Create archive file in REAL filesystem."""
        archive_dir = archive_dir or temp_workspace["temp_dir"]
        archive_path = os.path.join(archive_dir, f"{os.path.basename(input_dir)}.zip")
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        with open(archive_path, "w") as f:
            f.write("fake zip content")
        return archive_path

    mock_archive_input_directory = AsyncMock(side_effect=mock_archive_side_effect)
    mocks["_archive_input_directory"] = mock_archive_input_directory

    # --- Apply Patches ---
    patch_targets = [
        ("deployment.app.services.datasphere_service.get_datasets", mock_get_datasets),
        (
            "deployment.app.services.datasphere_service._archive_input_directory",
            mock_archive_input_directory,
        ),
        (
            "deployment.app.services.datasphere_service.get_db_connection",
            mock_get_db_connection,
        ),
    ]

    for target, mock in patch_targets:
        monkeypatch.setattr(target, mock)

    # --- Registry for Reset Fixture ---
    mocks["_DATASPHERE_ENV_APPLIED"] = True
    _GLOBAL_MOCKS_REGISTRY.update(mocks)

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
    if "client" in _GLOBAL_MOCKS_REGISTRY:
        client_mock = _GLOBAL_MOCKS_REGISTRY["client"]
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
        return_value=({"train": 1}, {"val": 1}),
    ) as mock_func:
        yield mock_func
