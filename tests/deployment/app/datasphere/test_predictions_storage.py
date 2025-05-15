import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from pathlib import Path
import json
from unittest.mock import patch, MagicMock, mock_open

from deployment.app.db.database import get_db_connection, execute_query
from deployment.app.db.schema import init_db, SCHEMA_SQL
from deployment.app.services.datasphere_service import save_predictions_to_db
from tests.deployment.app.datasphere.conftest import verify_predictions_saved, SAMPLE_PREDICTIONS

@patch('deployment.app.services.datasphere_service.get_db_connection')
@patch('deployment.app.db.schema.init_db')
@patch('sqlite3.Connection.executescript')
def test_save_predictions_to_db(mock_executescript, mock_init_db, mock_get_db, temp_db):
    """Test saving predictions from CSV to database"""
    # Настраиваем мок для использования тестовой БД
    conn = sqlite3.connect(temp_db["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    # Отключаем выполнение схемы в тестах
    mock_executescript.return_value = None
    
    # Вызываем тестируемую функцию с прямым соединением
    result = save_predictions_to_db(
        predictions_path=temp_db["predictions_path"],
        job_id=temp_db["job_id"],
        model_id=temp_db["model_id"],
        direct_db_connection=conn
    )
    
    # Используем общую функцию проверки результатов
    verify_predictions_saved(conn, result, SAMPLE_PREDICTIONS)
    conn.close()

@patch('deployment.app.services.datasphere_service.get_db_connection')
def test_save_predictions_to_db_invalid_path(mock_get_db, temp_db):
    """Test handling of invalid prediction file path"""
    conn = sqlite3.connect(temp_db["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    with pytest.raises(FileNotFoundError):
        save_predictions_to_db(
            predictions_path="/nonexistent/path/predictions.csv",
            job_id=temp_db["job_id"],
            model_id=temp_db["model_id"],
            direct_db_connection=conn
        )
    conn.close()
        
@patch('deployment.app.services.datasphere_service.get_db_connection')
def test_save_predictions_to_db_invalid_format(mock_get_db, temp_db):
    """Test handling of invalid prediction file format"""
    conn = sqlite3.connect(temp_db["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    # Создаем файл с неправильным форматом
    invalid_path = os.path.join(temp_db["temp_dir"].name, 'invalid.csv')
    with open(invalid_path, 'w') as f:
        f.write("This is not a valid CSV file")
        
    with pytest.raises(ValueError):
        save_predictions_to_db(
            predictions_path=invalid_path,
            job_id=temp_db["job_id"],
            model_id=temp_db["model_id"],
            direct_db_connection=conn
        )
    conn.close()

@patch('deployment.app.services.datasphere_service.get_db_connection')
def test_save_predictions_to_db_missing_columns(mock_get_db, temp_db):
    """Test handling of prediction file with missing required columns"""
    conn = sqlite3.connect(temp_db["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    # Создаем CSV с отсутствующими колонками квантилей
    missing_data = SAMPLE_PREDICTIONS.copy()
    # Удаляем некоторые колонки
    del missing_data['0.05']
    del missing_data['0.25']
    
    # Создаем и сохраняем DataFrame
    missing_cols_path = os.path.join(temp_db["temp_dir"].name, 'missing_cols.csv')
    pd.DataFrame(missing_data).to_csv(missing_cols_path, index=False)
    
    with pytest.raises(ValueError):
        save_predictions_to_db(
            predictions_path=missing_cols_path,
            job_id=temp_db["job_id"],
            model_id=temp_db["model_id"],
            direct_db_connection=conn
        )
    conn.close() 