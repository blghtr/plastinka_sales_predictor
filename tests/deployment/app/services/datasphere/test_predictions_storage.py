import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from deployment.app.services.datasphere_service import save_predictions_to_db
from tests.deployment.app.services.datasphere.conftest import verify_predictions_saved


def test_save_predictions_to_db(
    monkeypatch, mock_init_db, mock_get_db, temp_db, sample_predictions_data
):
    """Test saving predictions from CSV to database"""
    # Вызываем тестируемую функцию с DAL
    result = save_predictions_to_db(
        predictions_path=temp_db["predictions_path"],
        job_id=temp_db["job_id"],
        model_id=temp_db["model_id"],
        dal=temp_db["dal"],
    )

    # Используем общую функцию проверки результатов
    verify_predictions_saved(temp_db["dal"], result, sample_predictions_data)


def test_save_predictions_to_db_invalid_path(monkeypatch, mock_get_db, temp_db):
    """Test handling of invalid prediction file path"""
    with pytest.raises(FileNotFoundError):
        save_predictions_to_db(
            predictions_path="/nonexistent/path/predictions.csv",
            job_id=temp_db["job_id"],
            model_id=temp_db["model_id"],
            dal=temp_db["dal"],
        )


def test_save_predictions_to_db_invalid_format(monkeypatch, mock_get_db, temp_db):
    """Test handling of invalid prediction file format"""
    # Создаем файл с неправильным форматом
    invalid_path = os.path.join(temp_db["temp_dir_path"], "invalid.csv")
    with open(invalid_path, "w") as f:
        f.write("This is not a valid CSV file")

    with pytest.raises(ValueError):
        save_predictions_to_db(
            predictions_path=invalid_path,
            job_id=temp_db["job_id"],
            model_id=temp_db["model_id"],
            dal=temp_db["dal"],
        )


def test_save_predictions_to_db_missing_columns(
    temp_db, sample_predictions_data
):
    """Test handling of prediction file with missing required columns"""
    # Создаем CSV с отсутствующими колонками квантилей
    missing_data = sample_predictions_data.copy()
    # Удаляем некоторые колонки
    del missing_data["0.05"]
    del missing_data["0.25"]

    # Создаем и сохраняем DataFrame
    missing_cols_path = os.path.join(temp_db["temp_dir_path"], "missing_cols.csv")
    pd.DataFrame(missing_data).to_csv(missing_cols_path, index=False)

    with pytest.raises(ValueError):
        save_predictions_to_db(
            predictions_path=missing_cols_path,
            job_id=temp_db["job_id"],
            model_id=temp_db["model_id"],
            dal=temp_db["dal"],
        )


def test_save_predictions_to_db_db_connection_failure(temp_db, monkeypatch):
    """Test handling of database connection failure during predictions save"""
    # Simulate DAL insert_predictions raising an exception
    with monkeypatch.context() as m:
        m.setattr(temp_db["dal"], "insert_predictions", MagicMock(side_effect=Exception("Simulated DAL error")))

        with pytest.raises(Exception, match="Simulated DAL error"):
            save_predictions_to_db(
                predictions_path=temp_db["predictions_path"],
                job_id=temp_db["job_id"],
                model_id=temp_db["model_id"],
                dal=temp_db["dal"],
            )
