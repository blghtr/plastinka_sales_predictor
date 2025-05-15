import pytest
import os
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import uuid
import json
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime, timedelta
from contextlib import ExitStack

from deployment.app.db.database import (
    get_db_connection, 
    create_job, 
    get_job, 
    get_recent_models, 
    get_active_model,
    get_prediction_result,
    update_job_status,
    execute_query
)
from deployment.app.db.schema import init_db, SCHEMA_SQL
from deployment.app.models.api_models import (
    JobStatus, JobType, TrainingParams, 
    ModelConfig, OptimizerConfig, LRSchedulerConfig,
    TrainingDatasetConfig
)
from deployment.app.services.datasphere_service import run_job, save_predictions_to_db
from deployment.datasphere.client import DataSphereClient, DataSphereClientError

# Constants for tests
TEST_DS_JOB_ID = "ds_job_" + str(uuid.uuid4())
TEST_MODEL_ID = "model_" + str(uuid.uuid4())

# Sample predictions data for tests
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

# Helper function to create a complete TrainingParams object
def create_training_params(base_params=None):
    """
    Creates a complete TrainingParams object with all required fields.
    
    Args:
        base_params: Optional dictionary with parameters to use as a base
        
    Returns:
        A valid TrainingParams object
    """
    base_params = base_params or {}
    
    # Create model config
    model_config = ModelConfig(
        num_encoder_layers=3,
        num_decoder_layers=2,
        decoder_output_dim=128,
        temporal_width_past=12,
        temporal_width_future=6,
        temporal_hidden_size_past=64,
        temporal_hidden_size_future=64,
        temporal_decoder_hidden=128,
        batch_size=base_params.get('batch_size', 32),
        dropout=base_params.get('dropout', 0.2),
        use_reversible_instance_norm=True,
        use_layer_norm=True
    )
    
    # Create optimizer config
    optimizer_config = OptimizerConfig(
        lr=base_params.get('learning_rate', 0.001),
        weight_decay=0.0001
    )
    
    # Create LR scheduler config
    lr_shed_config = LRSchedulerConfig(
        T_0=10,
        T_mult=2
    )
    
    # Create training dataset config
    train_ds_config = TrainingDatasetConfig(
        alpha=0.05,
        span=12
    )
    
    # Create complete TrainingParams
    return TrainingParams(
        model_config=model_config,
        optimizer_config=optimizer_config,
        lr_shed_config=lr_shed_config,
        train_ds_config=train_ds_config,
        lags=12,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )

# ==============================================
# Mocks for various services
# ==============================================

def mock_get_active_parameter_set(connection=None):
    """Mock implementation of get_active_parameter_set to return test parameters"""
    return {
        "parameter_set_id": 1,
        "parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "hidden_size": 64,
            "lstm_layers": 2,
            "dropout": 0.2,
            "batch_size": 32,
            "max_epochs": 10,
            "learning_rate": 0.001
        },
        "default_metric_name": "mape",
        "default_metric_value": 15.3
    }

def mock_get_datasets(start_date=None, end_date=None, config=None, output_dir=None):
    """Mock implementation of get_datasets to return test datasets"""
    # Create some dummy files in the output_dir to simulate dataset generation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Create a dummy dataset file
        with open(os.path.join(output_dir, 'dummy_dataset.json'), 'w') as f:
            f.write('{"dummy": "dataset"}')
    
    return {
        "train_dataset": MagicMock(),
        "test_dataset": MagicMock()
    }

def mock_create_model_record(*args, **kwargs):
    """Mock for create_model_record (заглушка для тестов)"""
    return None

# ==============================================
# Setup functions for different testing strategies
# ==============================================

def setup_db_mocks():
    """
    Настраивает моки для функций базы данных.
    
    Returns:
        ExitStack context manager для использования через `with setup_db_mocks():`
    """
    # Базовый мок соединения
    mock_conn = MagicMock()
    
    # Мок для get_db_connection
    get_db_connection_patch = patch('deployment.app.db.database.get_db_connection', return_value=mock_conn)
    
    # Мок для update_job_status
    update_job_status_patch = patch('deployment.app.db.database.update_job_status')
    
    # Мок для get_job
    mock_job = {'job_id': 'test-job-id', 'status': 'completed', 'progress': 100}
    get_job_patch = patch('deployment.app.db.database.get_job', return_value=mock_job)
    
    # Мок для create_job
    create_job_patch = patch('deployment.app.db.database.create_job', return_value='test-job-id')
    
    # Мок для get_active_model
    mock_model = {
        'model_id': 'test-model-id', 
        'job_id': 'test-job-id', 
        'is_active': True,
        'metadata': json.dumps({
            'file_size_bytes': 1024,
            'downloaded_from_ds_job': 'test-ds-job',
            'original_path': '/tmp/model.onnx'
        })
    }
    get_active_model_patch = patch('deployment.app.db.database.get_active_model', return_value=mock_model)
    
    # Мок для get_prediction_result
    mock_result = {'result_id': 'test-result-id', 'job_id': 'test-job-id'}
    get_prediction_result_patch = patch('deployment.app.db.database.get_prediction_result', return_value=mock_result)
    
    # Мок для execute_query
    def mock_execute_query(query, params=(), fetchall=False, connection=None):
        if 'job_status_history' in query and fetchall:
            return [
                {'job_id': 'test-job-id', 'status': 'pending', 'progress': 0, 'status_message': 'Initializing'},
                {'job_id': 'test-job-id', 'status': 'running', 'progress': 50, 'status_message': 'Running'},
                {'job_id': 'test-job-id', 'status': 'completed', 'progress': 100, 'status_message': 'Completed'}
            ]
        elif fetchall:
            return [{'id': 1}, {'id': 2}]
        else:
            return {'id': 1}
    
    execute_query_patch = patch('deployment.app.db.database.execute_query', side_effect=mock_execute_query)
    
    # Для тестирования ошибок
    def get_job_with_status(job_id, status='completed', connection=None):
        return {'job_id': job_id, 'status': status, 'progress': 100 if status == 'completed' else 0}
    
    # Создаем контекстный менеджер для всех патчей
    exit_stack = ExitStack()
    
    # Применяем все патчи
    mock_patches = {
        'get_db_connection': exit_stack.enter_context(get_db_connection_patch),
        'update_job_status': exit_stack.enter_context(update_job_status_patch),
        'get_job': exit_stack.enter_context(get_job_patch),
        'create_job': exit_stack.enter_context(create_job_patch),
        'get_active_model': exit_stack.enter_context(get_active_model_patch),
        'get_prediction_result': exit_stack.enter_context(get_prediction_result_patch),
        'execute_query': exit_stack.enter_context(execute_query_patch),
    }
    
    # Дополнительные утилиты для тестов
    mock_patches['get_job_with_status'] = get_job_with_status
    mock_patches['mock_conn'] = mock_conn
    mock_patches['exit_stack'] = exit_stack
    
    # Возвращаем словарь с патчами вместо использования yield
    return mock_patches

def setup_temp_db():
    """
    Настраивает временную тестовую БД на диске для интеграционных тестов.
    Создает временный файл БД с необходимыми таблицами и тестовыми данными.
    
    Yields:
        dict: Словарь с параметрами настроенной БД
    """
    # Создаём временную директорию для тестовых файлов
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test.db')
    
    # Инициализируем схему БД
    init_db(db_path)
    
    # Проверяем, что все необходимые таблицы созданы
    debug_conn = sqlite3.connect(db_path)
    debug_cursor = debug_conn.cursor()
    debug_cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
        "('dim_multiindex_mapping', 'fact_predictions', 'prediction_results', 'models', 'jobs', 'job_status_history')"
    )
    table_count = debug_cursor.fetchone()[0]
    debug_conn.close()
    
    # Убеждаемся, что таблицы созданы
    assert table_count >= 5, f"Database should have at least 5 required tables, but found {table_count}"
    
    # Создаем соединение для тестов
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Вставляем тестовые данные
    parameter_set_id = 1
    params = {
        "input_chunk_length": 12,
        "output_chunk_length": 6,
        "hidden_size": 64,
        "lstm_layers": 2,
        "dropout": 0.2,
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 0.001
    }
    
    # Вставляем тестовый parameter_set
    cursor.execute(
        """INSERT INTO parameter_sets 
           (parameter_set_id, parameters, is_active, created_at) 
           VALUES (?, ?, ?, ?)""",
        (parameter_set_id, json.dumps(params), 1, datetime.now().isoformat())
    )
    
    # Вставляем тестовые multiindex_mapping
    sample_multiindices = [
        ('123456789012', 'Artist A', 'Album X', 'Standard', 'A', 'Studio', '2010s', '2020s', 'Rock', 2015),
        ('987654321098', 'Artist B', 'Album Y', 'Deluxe', 'B', 'Live', '2000s', '2010s', 'Pop', 2007),
        ('555555555555', 'Artist C', 'Album Z', 'Limited', 'C', 'Compilation', '1990s', '2000s', 'Jazz', 1995)
    ]
    
    for idx, (barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year) in enumerate(sample_multiindices, start=1):
        cursor.execute(
            """INSERT INTO dim_multiindex_mapping
               (multiindex_id, barcode, artist, album, cover_type, price_category, release_type,
                recording_decade, release_decade, style, record_year)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (idx, barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, record_year)
        )
    
    # Создаем тестовый job
    job_id = "test_job_" + str(uuid.uuid4())
    cursor.execute(
        """INSERT INTO jobs 
           (job_id, job_type, status, created_at, updated_at) 
           VALUES (?, ?, ?, ?, ?)""",
        (job_id, JobType.TRAINING.value, JobStatus.PENDING.value, datetime.now().isoformat(), datetime.now().isoformat())
    )
    
    # Создаем тестовую модель
    model_id = "test_model_" + str(uuid.uuid4())
    cursor.execute(
        """INSERT INTO models 
           (model_id, job_id, model_path, created_at, is_active) 
           VALUES (?, ?, ?, ?, ?)""",
        (model_id, job_id, "/test/path/model.onnx", datetime.now().isoformat(), 1)
    )
    
    # Применяем изменения и закрываем соединение
    conn.commit()
    conn.close()
    
    # Сохраняем оригинальный путь к БД (если он установлен)
    original_db_path = os.environ.get('DATABASE_PATH', None)
    
    # Устанавливаем путь к тестовой БД
    os.environ['DATABASE_PATH'] = db_path
    
    # Создаем временный файл с предсказаниями для тестов
    predictions_path = os.path.join(temp_dir.name, 'predictions.csv')
    pd.DataFrame(SAMPLE_PREDICTIONS).to_csv(predictions_path, index=False)
    
    # Создаем файл конфигурации для DataSphere
    config_path = os.path.join(temp_dir.name, 'config.yaml')
    with open(config_path, 'w') as f:
        f.write("job_name: test_job\nparams:\n  param1: value1\n")
    
    # Формируем словарь с параметрами настройки
    setup_data = {
        "temp_dir": temp_dir,
        "db_path": db_path,
        "predictions_path": predictions_path,
        "config_path": config_path,
        "job_id": job_id,
        "model_id": model_id,
        "parameter_set_id": parameter_set_id,
        "params": params,
        "original_db_path": original_db_path
    }
    
    # Возвращаем настройки
    try:
        yield setup_data
    finally:
        # Восстанавливаем оригинальный путь к БД
        if original_db_path is not None:
            os.environ['DATABASE_PATH'] = original_db_path
        else:
            os.environ.pop('DATABASE_PATH', None)
        
        # Закрываем все соединения
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
        except Exception:
            pass
        
        # Удаляем временную директорию
        try:
            temp_dir.cleanup()
        except PermissionError:
            # На Windows иногда не удается удалить директорию сразу
            pass

# ==============================================
# Pytest fixtures
# ==============================================

@pytest.fixture
def mocked_db():
    """
    Fixture для полностью моковой БД.
    Возвращает словарь моков для использования в тестах.
    """
    # Создаем контекстный менеджер и применяем патчи
    mocks = setup_db_mocks()
    
    # Возвращаем моки и заботимся о закрытии контекстного менеджера
    try:
        yield mocks
    finally:
        # Закрываем все патчи после использования фикстуры
        if 'exit_stack' in mocks:
            mocks['exit_stack'].close()

@pytest.fixture
def temp_db():
    """
    Fixture для реальной временной БД на диске.
    Возвращает словарь с параметрами настроенной БД.
    """
    yield from setup_temp_db()

@pytest.fixture
def mock_datasphere():
    """
    Fixture для мокирования DataSphere клиента и связанных сервисов.
    """
    # Настройка мока DataSphere клиента
    mock_client = MagicMock(spec=DataSphereClient)
    mock_client.submit_job.return_value = TEST_DS_JOB_ID
    mock_client.get_job_status.return_value = "SUCCESS"
    
    # Mock для download_job_results
    def download_results_side_effect(job_id, output_dir, with_logs=False, with_diagnostics=False):
        # Не делаем реальных операций с файлами, просто имитируем успешное
        # скачивание и наличие файлов
        # Вместо создания файлов, просто имитируем что они существуют
        # Мы не будем физически создавать директории, т.к. это может вызвать проблемы
        # на разных ОС, особенно в конвейере CI

        # Вместо этого мокаем os.path.exists чтобы он возвращал True для путей файлов результатов
        # Возвращаем моки путей к файлам как будто они существуют
        metrics_path = os.path.join(output_dir, 'metrics.json')
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        model_path = os.path.join(output_dir, 'model.onnx')
        
        # Если запрашиваются логи, добавляем пути логов
        log_paths = []
        if with_logs:
            log_paths.append(os.path.join(output_dir, 'logs.txt'))
        
        # Если запрашивается диагностика, добавляем пути диагностики
        diag_paths = []
        if with_diagnostics:
            diag_paths.append(os.path.join(output_dir, 'diagnostics.json'))
        
        # Мокаем проверки существования этих файлов в основном коде
        # Обновляем side_effect для os.path.exists
        original_exists = os.path.exists
        
        def patched_exists(path):
            # Для путей к нашим файлам возвращаем True
            if path in [metrics_path, predictions_path, model_path, output_dir] + log_paths + diag_paths:
                return True
            # Для всех остальных вызываем оригинальную функцию
            return original_exists(path)
        
        # Обновляем мок
        patches.get('os_path_exists').side_effect = patched_exists
        
        # Не делаем реальных операций, вместо этого просто возвращаем, что скачивание успешно
        return None
    
    mock_client.download_job_results.side_effect = download_results_side_effect
    
    # Создаем патчи для всех зависимостей
    patches = {
        'client': patch('deployment.app.services.datasphere_service.DataSphereClient', return_value=mock_client),
        'get_active_parameter_set': patch('deployment.app.services.datasphere_service.get_active_parameter_set', mock_get_active_parameter_set),
        'get_datasets': patch('deployment.app.services.datasphere_service.get_datasets', mock_get_datasets),
        'create_model_record': patch('deployment.app.services.datasphere_service.create_model_record', mock_create_model_record),
        'os_path_exists': patch('os.path.exists', return_value=True),  # Всегда считаем, что файл существует
    }
    
    # Настройка мока _get_job_parameters
    mock_get_job_parameters = AsyncMock()
    mock_get_job_parameters.return_value = (create_training_params(), 1)
    patches['_get_job_parameters'] = patch('deployment.app.services.datasphere_service._get_job_parameters', mock_get_job_parameters)
    
    # Настройка мока settings
    mock_settings = MagicMock()
    mock_settings.datasphere.train_job.input_dir = 'input_dir'
    mock_settings.datasphere.train_job.output_dir = 'output_dir'
    mock_settings.datasphere.train_job.job_config_path = 'config.yaml'
    mock_settings.datasphere.max_polls = 2
    mock_settings.datasphere.poll_interval = 0.1
    patches['settings'] = patch('deployment.app.services.datasphere_service.settings', mock_settings)
    
    # Настройка мока _prepare_job_datasets
    mock_prepare_datasets = AsyncMock()
    mock_prepare_datasets.return_value = None
    patches['_prepare_job_datasets'] = patch('deployment.app.services.datasphere_service._prepare_job_datasets', mock_prepare_datasets)
    
    # Настройка мока _archive_input_directory чтобы избежать создания реального input.zip
    mock_archive_input = AsyncMock()
    mock_archive_input.return_value = "input.zip"  # Просто возвращаем имя файла, не создавая его
    patches['_archive_input_directory'] = patch('deployment.app.services.datasphere_service._archive_input_directory', mock_archive_input)
    
    # Настройка мока save_model_file_and_db чтобы избежать работы с реальным файлом модели
    mock_save_model = AsyncMock()
    mock_save_model.return_value = "test_model_" + str(uuid.uuid4())
    patches['save_model_file_and_db'] = patch('deployment.app.services.datasphere_service.save_model_file_and_db', mock_save_model)
    
    # Настройка мока create_training_result
    mock_create_training = AsyncMock()
    mock_create_training.return_value = None
    patches['create_training_result'] = patch('deployment.app.services.datasphere_service.create_training_result', mock_create_training)
    
    # Настройка мока для save_predictions_to_db
    def mock_save_predictions(predictions_path, job_id, model_id, direct_db_connection=None):
        """Мок для save_predictions_to_db чтобы избежать чтения реального CSV файла"""
        return {
            "result_id": "test_result_" + str(uuid.uuid4()),
            "predictions_count": 5
        }
    
    patches['save_predictions_to_db'] = patch('deployment.app.services.datasphere_service.save_predictions_to_db', 
                                                side_effect=mock_save_predictions)
    
    # Настройка mock_open для чтения конфига и результатов
    mock_file_content = "job_name: test_job\nparams:\n  param1: value1\n"
    mock_metrics_content = json.dumps({"mape": 15.3, "rmse": 5.7, "mae": 3.2, "r2": 0.85})
    
    # Создаем патч для open, который будет обрабатывать разные файлы по-разному
    mock_open_instance = mock_open(read_data=mock_file_content)
    
    def mock_open_side_effect(file_path, *args, **kwargs):
        # Для metrics.json возвращаем метрики
        if file_path.endswith('metrics.json'):
            mock_metrics = mock_open(read_data=mock_metrics_content)
            return mock_metrics(file_path, *args, **kwargs)
        # Для YAML конфига и всего остального возвращаем стандартное содержимое
        return mock_open_instance(file_path, *args, **kwargs)
    
    patches['open'] = patch('builtins.open', side_effect=mock_open_side_effect)
    
    # Добавляем патч для os.makedirs, чтобы не было ошибки с несуществующими директориями
    def mock_makedirs(path, exist_ok=False):
        # Имитируем создание директории без фактического создания
        return None
    
    patches['makedirs'] = patch('os.makedirs', side_effect=mock_makedirs)
    
    # Применяем все патчи через ExitStack
    exit_stack = ExitStack()
    patched_objects = {}
    
    for name, p in patches.items():
        patched_objects[name] = exit_stack.enter_context(p)
    
    # Создаем временные директории для тестов
    os.makedirs('input_dir', exist_ok=True)
    os.makedirs('output_dir', exist_ok=True)
    
    # Возвращаем моки и патчи
    try:
        yield patched_objects
    finally:
        exit_stack.close()
        
        # Удаляем временные директории
        try:
            import shutil
            if os.path.exists('input_dir'):
                shutil.rmtree('input_dir')
            if os.path.exists('output_dir'):
                shutil.rmtree('output_dir')
        except Exception as e:
            print(f"Cleanup error: {e}")

# ==============================================
# Utility functions for assertions
# ==============================================

def verify_predictions_saved(connection, result, expected_data):
    """
    Проверяет, что предсказания корректно сохранены в базе данных.
    
    Args:
        connection: Соединение с БД
        result: Результат сохранения предсказаний
        expected_data: Ожидаемые данные (словарь как SAMPLE_PREDICTIONS)
    """
    cursor = connection.cursor()
    
    # Проверяем, что результат содержит валидный result_id
    assert result is not None
    assert "result_id" in result
    assert "predictions_count" in result
    assert result["predictions_count"] == len(expected_data["barcode"])
    
    # Проверяем таблицу prediction_results
    cursor.execute("SELECT * FROM prediction_results WHERE result_id = ?", (result["result_id"],))
    prediction_result = cursor.fetchone()
    assert prediction_result is not None
    
    # Проверяем, что в fact_predictions сохранены все предсказания
    cursor.execute("SELECT COUNT(*) as count FROM fact_predictions WHERE result_id = ?", (result["result_id"],))
    count = cursor.fetchone()["count"]
    assert count == len(expected_data["barcode"])
    
    # Проверяем значения квантилей для первой записи
    cursor.execute("""
        SELECT p.*, m.barcode, m.artist, m.album FROM fact_predictions p
        JOIN dim_multiindex_mapping m ON p.multiindex_id = m.multiindex_id
        WHERE m.barcode = ? AND m.artist = ? AND m.album = ?
        AND p.result_id = ?
    """, (expected_data["barcode"][0], expected_data["artist"][0], expected_data["album"][0], result["result_id"]))
    
    record = cursor.fetchone()
    assert record is not None
    
    # Проверяем каждую квантиль
    quantiles = ['05', '25', '50', '75', '95']
    for i, q in enumerate(quantiles):
        db_value = float(record[f"quantile_{q}"])
        expected_value = expected_data[f"0.{q}"][0]
        assert pytest.approx(db_value, 0.01) == expected_value, f"Quantile {q} mismatch" 