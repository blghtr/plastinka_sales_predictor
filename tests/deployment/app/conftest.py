import pytest
import os
import uuid
import json
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from contextlib import ExitStack
import sqlite3 # Added for temp_db
import tempfile # Added for temp_db
from datetime import datetime, timedelta, timezone
import pandas as pd # Added for temp_db (SAMPLE_PREDICTIONS to_csv)

# Corrected top-level imports for JobType and JobStatus
from deployment.app.models.api_models import JobType, JobStatus

# Other imports from api_models, ensure JobType/Status are not duplicated if they were here
from deployment.app.models.api_models import (
    JobResponse, JobDetails, JobsList, # Removed JobStatus, JobType from here if they were
    DataUploadResponse,
    ModelConfig, 
    OptimizerConfig, 
    LRSchedulerConfig, 
    TrainingDatasetConfig,
    TrainingParams, TrainingResponse,
    PredictionParams, PredictionResponse, ReportParams, ReportResponse,
    ReportType
)

# Imports for temp_db
from deployment.app.db.schema import init_db, SCHEMA_SQL
from deployment.app.db.database import get_db_connection, create_job # create_job added for temp_db
# Import dict_factory for consistent row_factory usage
from deployment.app.db.database import dict_factory

# Re-add import for DataSphereClient if it was removed and is needed elsewhere
from deployment.datasphere.client import DataSphereClient

# Constants for tests (moved from datasphere/conftest.py)
TEST_DS_JOB_ID = "ds_job_" + str(uuid.uuid4())
TEST_MODEL_ID = "model_" + str(uuid.uuid4()) # Added for temp_db

# Sample predictions data for tests (Added for temp_db)
SAMPLE_PREDICTIONS = {
    'barcode': ['123456789012', '123456789012', '987654321098', '987654321098', '555555555555'],
    'artist': ['Artist A', 'Artist A', 'Artist B', 'Artist B', 'Artist C'],
    'album': ['Album X', 'Album X', 'Album Y', 'Album Y', 'Album Z'],
    'cover_type': ['Standard', 'Standard', 'Deluxe', 'Deluxe', 'Limited'],
    'price_category': ['A', 'A', 'B', 'B', 'C'],
    'release_type': ['Studio', 'Studio', 'Live', 'Live', 'Compilation'],
    'recording_decade': ['2010s', '2010s', '2000s', '2000s', '1990s'],
    'release_decade': ['2020s', '2020s', '2010s', '2010s', '2000s'],
    'style': ['Rock', 'Rock', 'Pop', 'Pop', 'Jazz'],
    'record_year': [2015, 2015, 2007, 2007, 1995],
    '0.05': [10.5, 12.3, 5.2, 7.8, 3.1],
    '0.25': [15.2, 18.7, 8.9, 11.3, 5.7],
    '0.5': [21.4, 24.8, 12.6, 15.9, 7.5],
    '0.75': [28.3, 32.1, 17.8, 20.4, 10.2],
    '0.95': [35.7, 40.2, 23.1, 27.5, 15.8]
}

# Helper function to create a complete TrainingParams object (moved from datasphere/conftest.py)
def create_training_params(base_params=None):
    base_params = base_params or {}
    model_config = ModelConfig(
        num_encoder_layers=3, num_decoder_layers=2, decoder_output_dim=128,
        temporal_width_past=12, temporal_width_future=6,
        temporal_hidden_size_past=64, temporal_hidden_size_future=64,
        temporal_decoder_hidden=128, batch_size=base_params.get('batch_size', 32),
        dropout=base_params.get('dropout', 0.2), use_reversible_instance_norm=True,
        use_layer_norm=True
    )
    optimizer_config = OptimizerConfig(lr=base_params.get('learning_rate', 0.001), weight_decay=0.0001)
    lr_shed_config = LRSchedulerConfig(T_0=10, T_mult=2)
    train_ds_config = TrainingDatasetConfig(alpha=0.05, span=12)
    return TrainingParams(
        model_config=model_config, optimizer_config=optimizer_config,
        lr_shed_config=lr_shed_config, train_ds_config=train_ds_config,
        lags=12, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )

# Mocks for various services (moved from datasphere/conftest.py)
def mock_get_active_parameter_set(connection=None):
    return {
        "parameter_set_id": 1,
        "parameters": {
            "input_chunk_length": 12, "output_chunk_length": 6, "hidden_size": 64,
            "lstm_layers": 2, "dropout": 0.2, "batch_size": 32, "max_epochs": 10,
            "learning_rate": 0.001
        },
        "default_metric_name": "mape", "default_metric_value": 15.3
    }

def mock_get_datasets(start_date=None, end_date=None, config=None, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'dummy_dataset.json'), 'w') as f:
            f.write('{"dummy": "dataset"}')
    return {"train_dataset": MagicMock(), "test_dataset": MagicMock()}

def mock_create_model_record(*args, **kwargs):
    return None

@pytest.fixture
def mock_datasphere():
    """
    Fixture для мокирования DataSphere клиента и связанных сервисов.
    (Moved from datasphere/conftest.py)
    """
    mock_client = MagicMock(spec=DataSphereClient)
    mock_client.submit_job.return_value = TEST_DS_JOB_ID
    mock_client.get_job_status.return_value = "SUCCESS"
    
    # This internal helper for download_job_results is complex and refers to 'patches'
    # which is defined later. We need to ensure this structure is maintained or simplified.
    # For now, let's keep its structure but be mindful.
    
    # Store patches locally first before assigning to the global 'patches' dictionary in conftest
    # This avoids NameError if download_results_side_effect is defined before 'patches'
    # _patches_for_download_side_effect = {} # Removed as it was unused

    def download_results_side_effect(job_id, output_dir, with_logs=False, with_diagnostics=False):
        metrics_path = os.path.join(output_dir, 'metrics.json')
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        model_path = os.path.join(output_dir, 'model.onnx')
        log_paths = [os.path.join(output_dir, 'logs.txt')] if with_logs else []
        diag_paths = [os.path.join(output_dir, 'diagnostics.json')] if with_diagnostics else []
        
        original_exists = os.path.exists
        
        def patched_exists(path):
            if path in [metrics_path, predictions_path, model_path, output_dir] + log_paths + diag_paths:
                return True
            return original_exists(path) # Call original for other paths
        
        # Use the locally scoped _patches_for_download_side_effect if needed for os_path_exists
        # However, the fixture structure applies patches later via ExitStack.
        # The original code relied on 'patches.get('os_path_exists').side_effect = patched_exists'
        # This implies 'os_path_exists' patch object should exist.
        # Let's assume 'os_path_exists_patch_obj' will be available when this side_effect is called.
        # This might require slight reordering or ensuring the patch object is correctly referenced.
        # For now, we'll try to get it from the main `patches` dict, assuming it's populated.
        # This is tricky because this side_effect is defined before `patches` dict is fully populated with patch objects.
        # A cleaner way might be to pass the patcher (mocker) or the specific patch object into this side_effect if needed.
        # Or, more simply, just directly patch os.path.exists within this side_effect if it's only for this scope.

        # Simpler approach: Let the global os.path.exists mock (defined later in `patches`) handle this.
        # This side_effect for download_job_results primarily ensures no real file ops happen.
        return None
    
    mock_client.download_job_results.side_effect = download_results_side_effect
    
    patches = {
        'client': patch('deployment.app.services.datasphere_service.DataSphereClient', return_value=mock_client),
        'get_active_parameter_set': patch('deployment.app.services.datasphere_service.get_active_parameter_set', mock_get_active_parameter_set),
        'get_datasets': patch('deployment.app.services.datasphere_service.get_datasets', mock_get_datasets),
        'create_model_record': patch('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record),
        'os_path_exists': patch('os.path.exists', return_value=True),
    }
    
    mock_get_job_parameters = AsyncMock()
    mock_get_job_parameters.return_value = (create_training_params(), 1)
    patches['_get_job_parameters'] = patch('deployment.app.services.datasphere_service._get_job_parameters', mock_get_job_parameters)
    
    mock_settings = MagicMock()
    mock_settings.datasphere.train_job.input_dir = 'input_dir' # Relative to test execution
    mock_settings.datasphere.train_job.output_dir = 'output_dir' # Relative to test execution
    mock_settings.datasphere.train_job.job_config_path = 'config.yaml'
    mock_settings.datasphere.max_polls = 2
    mock_settings.datasphere.poll_interval = 0.1
    patches['settings'] = patch('deployment.app.services.datasphere_service.settings', mock_settings)
    
    mock_prepare_datasets = AsyncMock()
    mock_prepare_datasets.return_value = None
    patches['_prepare_job_datasets'] = patch('deployment.app.services.datasphere_service._prepare_job_datasets', mock_prepare_datasets)
    
    mock_archive_input = AsyncMock()
    mock_archive_input.return_value = "input.zip"
    patches['_archive_input_directory'] = patch('deployment.app.services.datasphere_service._archive_input_directory', mock_archive_input)
    
    mock_save_model = AsyncMock()
    mock_save_model.return_value = "test_model_" + str(uuid.uuid4())
    patches['save_model_file_and_db'] = patch('deployment.app.services.datasphere_service.save_model_file_and_db', mock_save_model)
    
    mock_create_training = MagicMock()
    mock_create_training.return_value = "test_training_result_" + str(uuid.uuid4())
    patches['create_training_result'] = patch('deployment.app.services.datasphere_service.create_training_result', mock_create_training)
    
    def mock_save_predictions(predictions_path, job_id, model_id, direct_db_connection=None):
        return {"result_id": "test_result_" + str(uuid.uuid4()), "predictions_count": 5}
    patches['save_predictions_to_db'] = patch('deployment.app.services.datasphere_service.save_predictions_to_db', side_effect=mock_save_predictions)
    
    mock_file_content = "job_name: test_job\nparams:\n  param1: value1\n"
    mock_metrics_content = json.dumps({"mape": 15.3, "rmse": 5.7, "mae": 3.2, "r2": 0.85})
    
    mock_open_instance = mock_open(read_data=mock_file_content)
    def mock_open_side_effect(file_path, *args, **kwargs):
        if str(file_path).endswith('metrics.json'): # Ensure file_path is string for endswith
            mock_metrics = mock_open(read_data=mock_metrics_content)
            return mock_metrics(file_path, *args, **kwargs)
        return mock_open_instance(file_path, *args, **kwargs)
    patches['open'] = patch('builtins.open', side_effect=mock_open_side_effect)
    
    def mock_makedirs_func(path, exist_ok=False): # Renamed to avoid conflict with 'os.makedirs'
        return None
    patches['makedirs'] = patch('os.makedirs', side_effect=mock_makedirs_func)
    
    exit_stack = ExitStack()
    patched_objects = {}
    for name, p in patches.items():
        patched_objects[name] = exit_stack.enter_context(p)
    
    # Create temporary input_dir and output_dir for the scope of the fixture
    # These are relative to where pytest is run.
    # For more robustness, consider using tmp_path fixture from pytest.
    # Path("input_dir").mkdir(exist_ok=True)
    # Path("output_dir").mkdir(exist_ok=True)
    # The original code used os.makedirs('input_dir', exist_ok=True)
    # This is now mocked by patches['makedirs'], so no actual directories are created by the fixture itself.
    # The mock for os.makedirs (mock_makedirs_func) does nothing.
    # If tests rely on these dirs existing for other reasons, this might need adjustment
    # or specific tests should handle their own directory setup if not covered by mocks.

    try:
        yield patched_objects
    finally:
        exit_stack.close()
        # Cleanup of physical dirs 'input_dir', 'output_dir' is not strictly needed
        # if all operations using them are mocked or if they are created in a temp location by tests.
        # The original fixture tried to shutil.rmtree them, which could fail.
        # Given os.makedirs is mocked, these dirs aren't created by the fixture's setup code.
        # If a test *needs* real dirs, it should use pytest's tmp_path. 

# Helper function: sets up a temporary directory and database file, initializes schema.
# This function itself does NOT manage the connection object's lifecycle.
def setup_temp_db_environment(): # Renamed to clarify its role
    """
    Sets up a temporary directory, database path, and initializes the schema.
    Yields db_path and the temp_dir object.
    """
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "test.db")
    config_path = os.path.join(temp_dir.name, "config.yaml") # If needed for settings override

    # Create a minimal config.yaml if settings depend on it for DB path
    # This might be necessary if settings.db.path is read from a file
    # For simplicity, assuming settings.db.path can be patched directly for tests if needed,
    # or that the DB_PATH in database.py is the primary source for tests not using settings overrides for path.

    # Patch DB_PATH for the duration of this setup if get_db_connection or init_db relies on it
    # This is safer than globally patching if not all tests need this specific path.
    # However, init_db takes db_path directly. get_db_connection in database.py uses settings.db.path.
    # For test isolation, init_db(db_path, ...) is preferred.

    init_db(db_path) # Corrected: Initialize schema directly on the path

    yield {
        "db_path": db_path,
        "temp_dir_obj": temp_dir, # So it can be cleaned up by the caller fixture
        "config_path": config_path # If used
    }
    # temp_dir.cleanup() # Cleanup is handled by the fixture that calls this

@pytest.fixture
def temp_db():
    """
    Fixture for a temporary SQLite database on disk, providing a managed connection.
    The connection uses dict_factory and is closed automatically after the test.
    """
    # Use a generator to manage the TemporaryDirectory lifecycle
    env_setup_gen = setup_temp_db_environment()
    db_env_info = next(env_setup_gen)
    
    db_path = db_env_info["db_path"]
    temp_dir_obj = db_env_info["temp_dir_obj"]
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = dict_factory  # Use dict_factory from database.py
        
        # For WAL mode if desired, though might complicate file locking with temp dirs
        # conn.execute("PRAGMA journal_mode=WAL;")

        yield {
            "conn": conn, # The managed connection
            "db_path": db_path,
            "config_path": db_env_info["config_path"]
            # temp_dir_obj is not yielded, its cleanup is managed here
        }
    finally:
        if conn:
            conn.close()
        if temp_dir_obj: # Ensure temp_dir_obj exists before cleanup
            temp_dir_obj.cleanup()



@pytest.fixture
def in_memory_db():
    """
    Fixture for an in-memory SQLite database, providing a managed connection.
    The connection uses dict_factory and is closed automatically.
    """
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = dict_factory # Use consistent dict_factory
        conn.executescript(SCHEMA_SQL) # Initialize schema for in-memory DB
        conn.commit() # Commit schema changes
        yield {
            "conn": conn,
            "db_path": ":memory:" # Indicate it's an in-memory DB
        }
    finally:
        if conn:
            conn.close()

@pytest.fixture
def setup_db_with_data(temp_db): # Now depends on the revised temp_db
    """
    Sets up the database with extensive test data using a managed connection from temp_db.
    Also creates temporary model files.
    """
    db_info = temp_db # db_info now contains {'conn': ..., 'db_path': ...}
    conn = db_info["conn"] # Use the managed connection from temp_db
    db_path = db_info["db_path"]

    # This fixture should NOT manage the lifecycle of 'conn' (open/close)
    # as temp_db is now responsible for that.

    model_files_temp_dir = tempfile.TemporaryDirectory()
    model_dir_path = model_files_temp_dir.name

    try:
        cursor = conn.cursor()
        
        # Populate data using 'conn'
        # (Sample data population logic remains the same as in the user's file)
        # ... (Ensure all INSERT statements use 'cursor' from 'conn') ...

        # Example of data population (ensure all parts use 'cursor'):
        # Jobs
        sample_jobs = [
            # (job_id, job_type, status, created_at, updated_at, parameters, result_id, error_message, progress)
            ("job-1", JobType.TRAINING.value, JobStatus.COMPLETED.value, 
             (datetime.now() - timedelta(days=2)).isoformat(), 
             (datetime.now() - timedelta(days=2, hours=-1)).isoformat(), 
             json.dumps({"param_set_id": "param-1", "model_id_to_train": "model-1-trained"}), 
             "res-train-1", None, 100),
            ("job-2", JobType.PREDICTION.value, JobStatus.COMPLETED.value, 
             (datetime.now() - timedelta(days=1)).isoformat(), 
             (datetime.now() - timedelta(days=1, hours=-1)).isoformat(), 
             json.dumps({"model_id": "model-1", "data_date": "2023-01-15"}), 
             "res-pred-1", None, 100),
            ("job-3", JobType.DATA_UPLOAD.value, JobStatus.FAILED.value, 
             datetime.now().isoformat(), 
             datetime.now().isoformat(), 
             json.dumps({"source_file": "sales_2023.csv"}), 
             None, "Data upload failed", 50), # Added error message
            ("job-4", "custom_job_type", "running", 
             datetime.now().isoformat(), 
             datetime.now().isoformat(), 
             "{}", 
             None, None, 25) # No result_id or error_message for running job
        ]
        cursor.executemany("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)", sample_jobs)

        # Parameter Sets
        # Schema: parameter_set_id, parameters, created_at, is_active
        sample_param_sets = [
            ("param-1", json.dumps({"lr": 0.01, "epochs": 100, "feature_set": "A"}), 
             (datetime.now() - timedelta(days=3)).isoformat(), 1), # param_id, params, created_at, is_active
            ("param-2", json.dumps({"lr": 0.005, "epochs": 150, "feature_set": "B"}), 
             (datetime.now() - timedelta(days=2)).isoformat(), 0)  # param_id, params, created_at, is_active
        ]
        cursor.executemany("INSERT INTO parameter_sets VALUES (?,?,?,?)", sample_param_sets)

        # Models - create dummy model files
        model_path_1 = os.path.join(model_dir_path, "model-1.onnx")
        model_path_2 = os.path.join(model_dir_path, "model-2.onnx")
        with open(model_path_1, "w") as f: f.write("dummy model 1 data")
        with open(model_path_2, "w") as f: f.write("dummy model 2 data")

        sample_models = [
            ("model-1", "job-1", model_path_1, (datetime.now() - timedelta(days=2)).isoformat(), json.dumps({"desc": "Initial active model", "version": "1.0"}), 1),
            ("model-2", "another-job-id", model_path_2, (datetime.now() - timedelta(days=1)).isoformat(), json.dumps({"desc": "Challenger model", "version": "1.1"}), 0),
        ]
        cursor.executemany("INSERT INTO models VALUES (?,?,?,?,?,?)", sample_models)

        # Training Results
        sample_training_results = [
            ("res-train-1", "job-1", "model-1", "param-1", json.dumps({"mape": 10.5, "rmse": 1.2}), json.dumps({"lr": 0.01, "epochs": 100}), 3600),
             # Add a result for param-2 / model-2 to test get_best_... functions
            (str(uuid.uuid4()), "job-that-created-model-2", "model-2", "param-2", json.dumps({"mape": 9.8, "rmse": 1.1}), json.dumps({"lr": 0.005, "epochs": 150}), 4000),
        ]
        cursor.executemany("INSERT INTO training_results VALUES (?,?,?,?,?,?,?)", sample_training_results)
        
        # Prediction Results
        # Create a dummy predictions file
        predictions_file_path = os.path.join(model_dir_path, "predictions_job_2.csv")
        pd.DataFrame(SAMPLE_PREDICTIONS).to_csv(predictions_file_path, index=False)
        
        sample_prediction_results = [
            ("res-pred-1", "job-2", "model-1", 
             datetime.now().isoformat(),  # prediction_date
             predictions_file_path, 
             json.dumps({"avg_sales_pred": 100.5}))
        ]
        cursor.executemany("INSERT INTO prediction_results VALUES (?,?,?,?,?,?)", sample_prediction_results)
        
        # Data Upload Results
        sample_data_upload_results = [
            (str(uuid.uuid4()), "job-3", 0, "[]", 1) # Failed job, 0 records
        ]
        cursor.executemany("INSERT INTO data_upload_results VALUES (?,?,?,?,?)", sample_data_upload_results)

        # Processing Runs
        sample_processing_runs = [
            (1, (datetime.now() - timedelta(days=5)).isoformat(), "COMPLETED", "2023-01-01", "file1.csv,file2.csv", (datetime.now() - timedelta(days=5, hours=-2)).isoformat())
        ]
        cursor.executemany("INSERT INTO processing_runs (run_id, start_time, status, cutoff_date, source_files, end_time) VALUES (?,?,?,?,?,?)", sample_processing_runs)
        
        # Job Status History (example for job-1)
        # Schema: id (auto), job_id, status, progress, status_message, updated_at
        sample_job_history = [
            ("job-1", "pending", 0.0, "Job submitted", (datetime.now() - timedelta(days=2, minutes=5)).isoformat()),
            ("job-1", "running", 50.0, "Processing data", (datetime.now() - timedelta(days=2, minutes=2)).isoformat()),
            ("job-1", "completed", 100.0, "Job finished successfully", (datetime.now() - timedelta(days=2, hours=-1)).isoformat()),
        ]
        cursor.executemany("INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES (?,?,?,?,?)", sample_job_history)

        conn.commit()
        cursor.close()

        yield {
            "conn": conn, # Pass the managed connection from temp_db
            "db_path": db_path,
            "model_path_1": model_path_1,
            "model_path_2": model_path_2,
            "model_files_temp_dir_obj": model_files_temp_dir # For cleanup by this fixture
        }
    finally:
        # DO NOT close 'conn' here, temp_db fixture handles it.
        # Clean up the separate temporary directory for model files
        if 'model_files_temp_dir' in locals(): # ensure it was created
            model_files_temp_dir.cleanup()

@pytest.fixture
def sample_job_data():
    """Sample job data for testing"""
    return {
        "job_id": str(uuid.uuid4()),
        "job_type": "training",
        "parameters": {
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }

@pytest.fixture
def sample_model_data():
    """Sample model data for testing"""
    return {
        "model_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "model_path": "/path/to/model.onnx",
        "created_at": datetime.now(),
        "metadata": {
            "framework": "pytorch",
            "version": "1.9.0"
        }
    }

@pytest.fixture
def sample_parameter_set():
    """Sample parameter set for testing"""
    return {
        "input_chunk_length": 12,
        "output_chunk_length": 6,
        "hidden_size": 64,
        "lstm_layers": 2,
        "dropout": 0.2,
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 0.001
    } 