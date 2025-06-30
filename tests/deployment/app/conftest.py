"""
Основные фикстуры для тестирования на уровне deployment/app.
"""
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import date, datetime

import pytest

from deployment.app.db.database import dict_factory
from deployment.app.db.schema import init_db
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
    if isinstance(obj, (datetime, date)):
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
                "use_layer_norm": True
            },
            "optimizer_config": {
                "lr": 0.001,
                "weight_decay": 0.0001
            },
            "lr_shed_config": {
                "T_0": 10,
                "T_mult": 2
            },
            "train_ds_config": {
                "alpha": 0.05,
                "span": 12
            },
            "lags": 12,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
            "model_id": "default_model",
            "additional_params": {
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2022-12-31"
            }
        }
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
        "metadata": {
            "framework": "pytorch",
            "version": "1.9.0"
        }
    }

@pytest.fixture
def sample_config():
    """
    Sample config for testing
    """
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

@pytest.fixture
def in_memory_db():
    """
    УНИФИЦИРОВАННАЯ фикстура для SQLite в памяти.
    Создает и инициализирует схему БД в оперативной памяти (':memory:').
    Используется для быстрых unit-тестов.
    """
    conn = sqlite3.connect(':memory:')
    init_db(connection=conn)
    conn.row_factory = dict_factory
    yield {"conn": conn}
    conn.close()

@pytest.fixture
def file_based_db():
    """
    УНИФИЦИРОВАННАЯ фикстура для реальной временной БД на диске.
    Создает полнофункциональную БД с несколькими моделями, конфигами и результатами.
    """
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test.db')
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # --- Параметр-сеты (configs) ---
        config_id_1 = 'param-1'
        config_id_2 = 'param-2'
        config_data_1 = TrainingConfig(
            model_config=ModelConfig(num_encoder_layers=2, batch_size=32),
            optimizer_config=OptimizerConfig(lr=0.01),
            lr_shed_config=LRSchedulerConfig(T_0=10),
            train_ds_config=TrainingDatasetConfig(alpha=0.1)
        ).model_dump(mode='json')
        config_data_2 = TrainingConfig(
            model_config=ModelConfig(num_encoder_layers=4, batch_size=64),
            optimizer_config=OptimizerConfig(lr=0.005),
            lr_shed_config=LRSchedulerConfig(T_0=20),
            train_ds_config=TrainingDatasetConfig(alpha=0.05)
        ).model_dump(mode='json')

        cursor.execute(
            """INSERT INTO configs (config_id, parameters, is_active, created_at) VALUES (?, ?, ?, ?)""",
            (config_id_1, json.dumps(config_data_1), 1, datetime.now().isoformat())
        )
        cursor.execute(
            """INSERT INTO configs (config_id, parameters, is_active, created_at) VALUES (?, ?, ?, ?)""",
            (config_id_2, json.dumps(config_data_2), 0, datetime.now().isoformat())
        )

        # --- Модели ---
        job_id = 'job-test'
        model_id_1 = 'model-1'
        model_id_2 = 'model-2'
        model_path_1 = os.path.join(temp_dir.name, 'model_1.onnx')
        model_path_2 = os.path.join(temp_dir.name, 'model_2.onnx')
        with open(model_path_1, 'w') as f: f.write('dummy model 1')
        with open(model_path_2, 'w') as f: f.write('dummy model 2')

        cursor.execute(
            """INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)""",
            (model_id_1, job_id, model_path_1, datetime.now().isoformat(), 1)
        )
        cursor.execute(
            """INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)""",
            (model_id_2, job_id, model_path_2, datetime.now().isoformat(), 0)
        )

        # --- Результаты тренировки (для метрик) ---
        result_id_1 = "tr-result-1"
        result_id_2 = "tr-result-2"
        metrics_data_1 = {"mape": 9.9, "val_loss": 0.08}
        metrics_data_2 = {"mape": 9.8, "val_loss": 0.03}
        cursor.execute(
            """INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics) VALUES (?, ?, ?, ?, ?)""",
            (result_id_1, job_id, model_id_1, config_id_1, json.dumps(metrics_data_1, default=json_default_serializer))
        )
        cursor.execute(
            """INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics) VALUES (?, ?, ?, ?, ?)""",
            (result_id_2, job_id, model_id_2, config_id_2, json.dumps(metrics_data_2, default=json_default_serializer))
        )

        # --- Запись о job ---
        cursor.execute(
            """INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)""",
            (job_id, "training", "completed", datetime.now().isoformat(), datetime.now().isoformat())
        )

        conn.commit()

        setup_data = {
            "temp_dir_path": temp_dir.name,
            "db_path": db_path,
            "conn": conn,
            "job_id": job_id,
            "model_id_1": model_id_1,
            "model_id_2": model_id_2,
            "config_id_1": config_id_1,
            "config_id_2": config_id_2,
        }

        yield setup_data

    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            temp_dir.cleanup()
        except Exception:
            pass

@pytest.fixture
def sample_job_data():
    """
    Sample job data for testing
    """
    return {
        "job_id": str(uuid.uuid4()),
        "job_type": "training",
        "parameters": {
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
