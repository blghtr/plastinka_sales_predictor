
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.models.api_models import JobStatus, JobType

# Этот ключ определён в tests/deployment/app/api/conftest.py
TEST_X_API_KEY = "test_x_api_key_conftest"

class TestToyApiJobs:
    """
    Рабочий пример для демонстрации исправленной тестовой инфраструктуры.
    """

    def test_get_job_status_success(self, api_client: TestClient, in_memory_db: DataAccessLayer):
        """
        Проверяет успешное получение статуса созданной задачи.
        Демонстрирует правильное использование фикстур.
        """
        # Arrange: Создаём задачу в БД через DAL, который мы получили из фикстуры.
        # Метод create_job возвращает ID созданной задачи.
        job_id = in_memory_db.create_job(
            job_type=JobType.TRAINING,
            parameters={"test_param": "test_value"}
        )
        # Обновляем статус, чтобы проверить, что вернётся именно он
        in_memory_db.update_job_status(job_id, JobStatus.COMPLETED, progress=100.0)

        # Act: Делаем запрос к API, используя тот же экземпляр БД
        response = api_client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert: Проверяем результат
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.COMPLETED.value
        assert data["progress"] == 100.0

    def test_create_training_job_success(self, api_client: TestClient, in_memory_db: DataAccessLayer, monkeypatch):
        """
        Проверяет успешное создание задачи обучения.
        Демонстрирует подготовку данных (активный конфиг) и мокирование фоновых задач.
        """
        # Arrange:
        # 1. Создаём активную конфигурацию, необходимую для эндпоинта
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        in_memory_db.create_or_get_config(config_data, is_active=True)

        # 2. Мокируем BackgroundTasks, чтобы избежать реального запуска фоновых задач
        mock_add_task = MagicMock()
        monkeypatch.setattr("fastapi.BackgroundTasks.add_task", mock_add_task)

        # 3. Готовим данные для POST-запроса
        payload = {
            "dataset_start_date": "2023-01-01",
            "dataset_end_date": "2023-06-30",
        }

        # Act: Выполняем запрос на создание задачи
        response = api_client.post(
            "/api/v1/jobs/training",
            json=payload,
            headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert:
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == JobStatus.PENDING.value

        # Проверяем, что задача действительно была создана в БД
        job_id = data["job_id"]
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        assert job_from_db["job_type"] == JobType.TRAINING.value

        # Проверяем, что фоновая задача была добавлена
        mock_add_task.assert_called_once()
