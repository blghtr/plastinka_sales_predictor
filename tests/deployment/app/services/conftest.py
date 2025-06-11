"""
Конфигурационный файл pytest для тестов сервисов.
Содержит КЛЮЧЕВУЮ фикстуру mock_service_env для полного мокирования окружения сервиса.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import importlib
from pathlib import Path
import os
import pickle
from copy import deepcopy

# The service module will be imported INSIDE the fixture after mocks are set up.
# This is critical to ensure it's loaded in a fully patched environment.

# We can still import other dependencies that are safe.
from deployment.app.models.api_models import TrainingConfig, ModelConfig, OptimizerConfig, LRSchedulerConfig, TrainingDatasetConfig, JobStatus

# =================================================================================
# Core Fixtures
# =================================================================================

@pytest.fixture(scope="function")
def mock_service_env(monkeypatch, fs):
    """
    A comprehensive fixture to mock the entire environment for datasphere_service.
    This includes settings, file system, database, and other dependencies.
    It ensures the service module is loaded AFTER patches are applied.
    """
    mocks = {}

    # --- Environment and Settings Mocks ---
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("CONFIG_PATH", "configs/test_config.yml")
    monkeypatch.setenv("SECRET_PATH", "configs/test_secret.yml")
    monkeypatch.setenv("AUTO_SELECT_BEST_MODEL", "false")
    monkeypatch.setenv("DATASPHERE_MAX_POLLS", "3")
    monkeypatch.setenv("DATASPHERE_POLL_INTERVAL", "0.1")
    
    # Import config module here to get a fresh copy of settings
    from deployment.app import config as app_config_module
    
    # Create a new instance of the settings class to force it to re-read
    # the environment variables we've just set with monkeypatch.
    settings = app_config_module.AppSettings()

    # Add required DataSphere IDs to prevent initialization errors
    settings.datasphere.project_id = "test-project-id-from-conftest"
    settings.datasphere.folder_id = "test-folder-id-from-conftest"
    
    # --- Filesystem Mocks (pyfakefs) ---
    fake_ds_input_dir = settings.datasphere_input_dir
    fake_ds_output_dir = settings.datasphere_output_dir
    fake_job_config_path = settings.datasphere_job_config_path
    
    fs.makedirs(fake_ds_input_dir, exist_ok=True)
    fs.makedirs(fake_ds_output_dir, exist_ok=True)
    fs.makedirs("logs", exist_ok=True)
    
    # Create fake config file with proper structure matching real config.yaml
    fake_config_content = """# Configuration for Yandex DataSphere Job
name: test-job
desc: Test job for unit testing

# --- Compute Resources ---
cloud_instance_types:
  - g1.1

attach_project_disk: true

# --- Job Inputs ---
# inputs: 

# --- Job Outputs ---
outputs:
  - output.zip: OUTPUT

# --- Environment Configuration ---
env:
  python:
    type: manual
    version: '3.10'
    local-paths:
      - plastinka_sales_predictor/
    requirements-file: requirements.txt

# --- Execution Command ---
cmd: python plastinka_sales_predictor/datasphere_job/train_and_predict.py \\
    --input ${INPUT} \\
    --output ${OUTPUT}"""
    fs.create_file(fake_job_config_path, contents=fake_config_content)
    
    # Note: job_config_path is now computed automatically, no need to set it
    mocks['settings'] = settings
    # Also add fs to the mocks dict for direct access in tests
    mocks['fs'] = fs

    # --- Import and Reload the Service Module ---
    # This is the key step: import the module now that the environment is patched.
    # We must reload it to make sure it sees the patched environment, especially pyfakefs.
    from deployment.app.services import datasphere_service
    
    # Get the module directly from sys.modules to ensure we have the correct object
    # that importlib.reload expects, avoiding identity comparison issues.
    module_name = "deployment.app.services.datasphere_service"
    if module_name in sys.modules:
        ds_module = sys.modules[module_name]
        importlib.reload(ds_module)
    else:
        # This case should ideally not be hit if the import above works,
        # but as a fallback, we assign the imported module.
        ds_module = datasphere_service

    mocks['service_module'] = ds_module
    
    # --- Patch dependencies directly onto the newly loaded module ---
    monkeypatch.setattr(ds_module, 'settings', settings)

    # --- Core Service Dependency Mocks ---

    # 1. Mock the DataSphere client
    mock_client = MagicMock()
    mocks['client'] = mock_client
    # Patch the class within the reloaded module's namespace
    monkeypatch.setattr(ds_module, "DataSphereClient", MagicMock(return_value=mock_client))

    # 2. Mock the Database functions
    # Патчим функции там, где они используются - в модуле ds_module
    
    # --- Additional database mocks to resolve foreign key constraints ---
    
    # 2.1 Mock get_db_connection
    mock_db_conn = MagicMock()
    # The mock should return itself for cursor() calls
    mock_db_conn.cursor.return_value = mock_db_conn
    # And itself for execute() calls to allow chaining
    mock_db_conn.execute.return_value = mock_db_conn
    # Make executemany work too
    mock_db_conn.executemany.return_value = mock_db_conn
    # fetchone should return a dictionary-like object with required keys
    mock_db_conn.fetchone.return_value = {"id": 1, "multiindex_id": "test-multiindex-1"}
    mocks['db_connection'] = mock_db_conn
    
    # Patch get_db_connection to return our mock
    mock_get_db_connection = MagicMock(return_value=mock_db_conn)
    mocks['get_db_connection'] = mock_get_db_connection
    monkeypatch.setattr(ds_module, 'get_db_connection', mock_get_db_connection)
    
    # 2.2 Mock get_or_create_multiindex_id
    mock_get_or_create_multiindex_id = MagicMock(return_value="test-multiindex-1")
    mocks['get_or_create_multiindex_id'] = mock_get_or_create_multiindex_id
    monkeypatch.setattr(ds_module, 'get_or_create_multiindex_id', mock_get_or_create_multiindex_id)
    
    # 2.3 Mock update_job_status
    mock_update_job_status = MagicMock()
    mocks['update_job_status'] = mock_update_job_status
    monkeypatch.setattr(ds_module, 'update_job_status', mock_update_job_status)
    
    # 2.4 Mock create_training_result
    mock_create_training_result = MagicMock(return_value="test-result-id")
    mocks['create_training_result'] = mock_create_training_result
    monkeypatch.setattr(ds_module, 'create_training_result', mock_create_training_result)
    
    # 2.5 Mock execute_many for batch inserts
    mock_execute_many = MagicMock()
    mocks['execute_many'] = mock_execute_many
    monkeypatch.setattr(ds_module, 'execute_many', mock_execute_many)
    
    # 2.6 Mock save_predictions_to_db to skip database operations
    mock_save_predictions_to_db = MagicMock(
        return_value={"result_id": "test-prediction-result", "predictions_count": 5}
    )
    mocks['save_predictions_to_db'] = mock_save_predictions_to_db
    monkeypatch.setattr(ds_module, 'save_predictions_to_db', mock_save_predictions_to_db)
    
    # 2.7 Mock get_job to prevent database access during error handling
    mock_get_job = MagicMock(return_value={"job_id": "test-job", "status": "RUNNING"})
    mocks['get_job'] = mock_get_job
    monkeypatch.setattr(ds_module, 'get_job', mock_get_job)
    
    # 3. Mock the high-level data preparation function `get_datasets`
    # Это ключевой фикс, который отделяет тесты сервиса от реализации подготовки данных.
    def mock_get_datasets_side_effect(output_dir, **kwargs):
        # Реальная функция сохраняет файлы, симулируем это поведение.
        # Используем fs для работы в мокированной файловой системе.
        target_dir = Path(output_dir)
        fs.makedirs(target_dir, exist_ok=True)
        
        # Создаем все файлы, которые ожидает _verify_datasphere_job_inputs
        fs.create_file(target_dir / 'features.pkl')
        fs.create_file(target_dir / 'train.dill')
        fs.create_file(target_dir / 'val.dill')
        fs.create_file(target_dir / 'full.dill')

    mock_get_datasets = MagicMock(side_effect=mock_get_datasets_side_effect)
    mocks['get_datasets'] = mock_get_datasets
    # Патчим там, где функция вызывается - в модуле ds_module
    monkeypatch.setattr(ds_module, 'get_datasets', mock_get_datasets)

    # --- Utility Mocks ---
    fixed_uuid_hex = "fixeduuid000"
    fixed_uuid_part = fixed_uuid_hex[:12]
    mock_uuid_obj = MagicMock()
    mock_uuid_obj.hex = fixed_uuid_hex
    mocks['fixed_uuid_part'] = fixed_uuid_part
    
    mock_uuid4 = MagicMock(return_value=mock_uuid_obj)
    mocks['uuid4'] = mock_uuid4
    # We also need to patch uuid where the service module can see it
    monkeypatch.setattr(ds_module, 'uuid', MagicMock(uuid4=mock_uuid4))

    # --- Mock RetryMonitor to prevent test hangs ---
    # Create mock for RetryMonitor class and instance
    mock_retry_monitor_module = MagicMock()
    mock_retry_monitor_class = MagicMock()
    mock_retry_monitor_instance = MagicMock()
    
    # Configure the instance with required methods and properties
    mock_retry_monitor_instance.get_statistics.return_value = {
        "total_retries": 0,
        "successful_retries": 0,
        "exhausted_retries": 0,
        "successful_after_retry": 0,
        "high_failure_operations": [],
        "alerted_operations": [],
        "alert_thresholds": {},
        "operation_stats": {},
        "exception_stats": {},
        "timestamp": "2023-01-01T00:00:00"
    }
    
    # Set up the class to return our instance
    mock_retry_monitor_class.return_value = mock_retry_monitor_instance
    
    # Configure the module with required attributes
    mock_retry_monitor_module.RetryMonitor = mock_retry_monitor_class
    mock_retry_monitor_module.retry_monitor = mock_retry_monitor_instance
    mock_retry_monitor_module.DEFAULT_PERSISTENCE_PATH = None
    
    # Mock all commonly used functions
    mock_retry_monitor_module.record_retry = MagicMock()
    mock_retry_monitor_module.get_retry_statistics = MagicMock(
        return_value=mock_retry_monitor_instance.get_statistics.return_value
    )
    mock_retry_monitor_module.reset_retry_statistics = MagicMock()
    mock_retry_monitor_module.get_high_failure_operations = MagicMock(return_value=set())
    mock_retry_monitor_module.set_alert_threshold = MagicMock()
    mock_retry_monitor_module.register_alert_handler = MagicMock()
    mock_retry_monitor_module.get_alerted_operations = MagicMock(return_value=set())
    mock_retry_monitor_module.log_alert_handler = MagicMock()
    
    # Apply the mock to sys.modules
    monkeypatch.setitem(sys.modules, 'deployment.app.utils.retry_monitor', mock_retry_monitor_module)
    
    # Store in our mocks dictionary for reference
    mocks['retry_monitor'] = mock_retry_monitor_module

    # --- Other Mocks (retained from previous state) ---
    mock_create_model_record = MagicMock(return_value="test_model_id")
    mocks['create_model_record'] = mock_create_model_record
    monkeypatch.setattr(ds_module, 'create_model_record', mock_create_model_record)

    mock_archive_input_directory = AsyncMock()
    mocks['_archive_input_directory'] = mock_archive_input_directory
    monkeypatch.setattr(ds_module, '_archive_input_directory', mock_archive_input_directory)
    
    yield mocks

# =================================================================================
# Helper Functions for Tests
# =================================================================================
def create_training_params(**kwargs) -> TrainingConfig:
    """
    Factory function to create TrainingConfig instances for tests,
    making it easy to override specific parameters.
    """
    # Provide sensible defaults that can be overridden by kwargs
    defaults = {
        "nn_model_config": ModelConfig(),
        "optimizer_config": OptimizerConfig(),
        "lr_shed_config": LRSchedulerConfig(),
        "train_ds_config": TrainingDatasetConfig(),
        "lags": 12,
        "quantiles": [0.05, 0.5, 0.95]
    }
    
    # Deep update dictionary logic
    for key, value in kwargs.items():
        if isinstance(value, dict) and key in defaults:
            defaults[key].__dict__.update(value)
        else:
            defaults[key] = value

    return TrainingConfig(**defaults)

@pytest.fixture
def mock_asyncio_sleep():
    """Мокирует asyncio.sleep для ускорения тестов."""
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep

@pytest.fixture
def mock_datasphere_client():
    """Мокирует DataSphereClient для изоляции от реальных вызовов API."""
    mock_client_instance = MagicMock()
    mock_client_instance.submit_job = MagicMock(return_value="ds-job-id-123") 
    mock_client_instance.get_job_status.side_effect = ["RUNNING", "RUNNING", "COMPLETED"] 
    mock_client_instance.download_job_results.return_value = None
    
    # Патчим в месте использования
    with patch("deployment.app.services.datasphere_service.DataSphereClient", return_value=mock_client_instance) as mock_client:
        yield mock_client

@pytest.fixture
def mock_get_datasets():
    """Мокирует get_datasets, чтобы не генерировать реальные датасеты."""
    with patch("deployment.app.services.datasphere_service.get_datasets", return_value=({"train": 1}, {"val": 1})) as mock_func:
        yield mock_func
        
@pytest.fixture
def mock_shutil_rmtree(fs, mock_service_env):
    """
    Мокирует shutil.rmtree, чтобы избежать удаления базовой директории в pyfakefs.
    Зависит от mock_service_env для получения корректных путей.
    """
    reloaded_settings = mock_service_env["settings"]
    base_input_dir_to_preserve = Path(reloaded_settings.datasphere_input_dir).resolve()

    def selective_rmtree_side_effect(path_to_delete, ignore_errors=False, onerror=None):
        path_to_delete_resolved = Path(path_to_delete).resolve()

        if path_to_delete_resolved == base_input_dir_to_preserve:
            return  # Пропускаем удаление
        else:
            # Для остальных путей вызываем оригинальную логику, но в контексте pyfakefs
            # pyfakefs уже патчит shutil, поэтому просто вызываем fs.remove*
            try:
                if fs.isdir(str(path_to_delete)):
                    fs.remove_object(str(path_to_delete)) 
                elif fs.isfile(str(path_to_delete)):
                     fs.remove(str(path_to_delete))
            except FileNotFoundError:
                if not ignore_errors:
                    raise
    
    with patch("shutil.rmtree", side_effect=selective_rmtree_side_effect) as mock_func:
        yield mock_func

@pytest.fixture
def mock_shutil_make_archive_fs(fs): 
    """Мокирует shutil.make_archive для работы с pyfakefs."""
    def _fake_make_archive(base_name, format, root_dir=None, base_dir=None, **kwargs):
        archive_path = f"{base_name}.{format}"
        # Убедимся, что директория, которую мы архивируем, существует в fake fs
        if not fs.exists(str(root_dir)):
            raise FileNotFoundError(f"Source directory for make_archive {root_dir} does not exist in fake fs")
        # Создаем фейковый архивный файл
        fs.create_file(archive_path, contents="dummy zip content")
        return archive_path

    with patch("shutil.make_archive", side_effect=_fake_make_archive) as mock_func:
        yield mock_func 