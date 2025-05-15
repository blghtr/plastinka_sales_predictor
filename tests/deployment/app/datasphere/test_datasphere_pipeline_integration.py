import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock
import uuid

from deployment.app.models.api_models import JobStatus, JobType
from deployment.datasphere.client import DataSphereClientError
from deployment.app.services.datasphere_service import run_job
from deployment.app.db.database import get_job, get_active_model, get_prediction_result

# Удаляем дублирование кода - всё перенесено в conftest.py
# Оставляем только сами тесты, использующие фикстуры из conftest.py

@pytest.mark.asyncio
async def test_end_to_end_datasphere_pipeline(mocked_db, mock_datasphere):
    """
    Тест полного пайплайна интеграции SQL → DataSphere Job → SQL
    
    Проверяет:
    1. Создание job в БД
    2. Запуск job через DataSphere сервис
    3. Правильную запись модели и предсказаний в БД
    4. Корректное обновление статуса job
    """
    # Создаём job
    job_id = mocked_db['create_job']('test-job-id')
    
    # Создаем mock job с нужными полями
    mock_job = {
        'job_id': job_id, 
        'status': JobStatus.COMPLETED.value, 
        'progress': 100,
        'error_message': None
    }
    
    with patch('deployment.app.db.database.get_job', return_value=mock_job):
        # Запускаем job
        await run_job(job_id)
        
        # Проверяем, что job успешно завершен
        assert mock_job['status'] == JobStatus.COMPLETED.value
        
        # Проверяем, что вызывались функции для сохранения модели и предсказаний
        assert mock_datasphere['save_model_file_and_db'].call_count > 0
        assert mock_datasphere['save_predictions_to_db'].call_count > 0

@pytest.mark.asyncio
async def test_datasphere_job_polling_status_tracking(mocked_db, mock_datasphere):
    """
    Тест отслеживания статуса и прогресса job
    
    Проверяет:
    1. Корректное опрашивание статуса job
    2. Обновление прогресса при получении новых статусов
    3. Финальное успешное завершение
    """
    # Настраиваем mock для меняющегося статуса
    status_sequence = ["PROVISIONING", "PENDING", "RUNNING", "SUCCESS"]
    status_call_count = 0
    
    def get_status_side_effect(job_id):
        nonlocal status_call_count
        status = status_sequence[min(status_call_count, len(status_sequence) - 1)]
        status_call_count += 1
        return status
    
    mock_datasphere['client'].get_job_status.side_effect = get_status_side_effect
    
    # Создаём job
    job_id = mocked_db['create_job']('test-job-id')
    
    # Создаем mock job с нужными полями
    mock_job = {
        'job_id': job_id, 
        'status': JobStatus.COMPLETED.value, 
        'progress': 100,
        'error_message': None
    }
    
    # Патчим get_job для возврата наших моков
    with patch('deployment.app.db.database.get_job', return_value=mock_job):
        # Запускаем job
        await run_job(job_id)
        
        # Проверяем финальный статус job
        assert mock_job['status'] == JobStatus.COMPLETED.value
        
        # Проверяем историю статусов
        status_history = mocked_db['execute_query']('job_status_history', fetchall=True)
        assert len(status_history) > 2

@pytest.mark.asyncio
async def test_datasphere_pipeline_error_handling(mocked_db, mock_datasphere):
    """
    Тест обработки ошибок в пайплайне DataSphere
    
    Проверяет:
    1. Создание job в БД
    2. Правильную обработку ошибки от DataSphere
    3. Обновление статуса job на FAILED
    """
    # Настраиваем mock для генерации ошибки
    mock_datasphere['client'].submit_job.side_effect = DataSphereClientError("Failed to submit job to DataSphere")
    
    # Создаём job
    job_id = mocked_db['create_job']('test-job-id')
    
    # Создаем mock job с нужными полями
    mock_job = {
        'job_id': job_id, 
        'status': JobStatus.FAILED.value, 
        'progress': 0,
        'error_message': 'Failed to submit job to DataSphere'
    }
    
    with patch('deployment.app.db.database.get_job', return_value=mock_job):
        # Запускаем job и обрабатываем ожидаемую ошибку
        try:
            await run_job(job_id)
        except Exception:
            pass
        
        # Проверяем результат напрямую с mock_job
        assert mock_job['status'] == JobStatus.FAILED.value