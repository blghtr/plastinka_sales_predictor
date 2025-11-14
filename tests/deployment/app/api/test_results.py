
from datetime import datetime

import pytest

TEST_X_API_KEY = "test_x_api_key_conftest"
TEST_BEARER_TOKEN = "test_admin_token"


class TestResultsApi:
    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_training_results(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - Create real data in the database
        job_id = await dal.create_job(job_type="training", parameters={})
        await dal.update_job_status(job_id, "completed")
        model_id = "model1"
        config_id = await dal.create_or_get_config({
            "nn_model_config": {
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
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
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12
        })

        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=False,
        )

        metrics = {"acc": 0.9, "val_loss": 0.1}
        result_id = await dal.create_training_result(
            job_id=job_id,
            model_id=model_id,
            config_id=config_id,
            metrics=metrics,
            duration=100,
        )

        # Act
        response = await async_api_client.get("/api/v1/results/training", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["result_id"] == result_id
        assert data[0]["job_id"] == job_id
        assert data[0]["model_id"] == model_id
        assert data[0]["config_id"] == config_id
        assert data[0]["metrics"] == metrics
        assert data[0]["duration"] == 100

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_training_result_by_id(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - Create real data in the database
        job_id = await dal.create_job(job_type="training", parameters={})
        await dal.update_job_status(job_id, "completed")
        model_id = "model1"
        config_id = await dal.create_or_get_config({
            "nn_model_config": {
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
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
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12
        })

        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=False,
        )

        metrics = {"acc": 0.9, "val_loss": 0.1}
        result_id = await dal.create_training_result(
            job_id=job_id,
            model_id=model_id,
            config_id=config_id,
            metrics=metrics,
            duration=100,
        )

        # Act
        response = await async_api_client.get(f"/api/v1/results/training/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["result_id"] == result_id
        assert data["job_id"] == job_id
        assert data["model_id"] == model_id
        assert data["config_id"] == config_id
        assert data["metrics"] == metrics
        assert data["duration"] == 100

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_training_result_not_found(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - No data in database
        result_id = "not-found"

        # Act
        response = await async_api_client.get(f"/api/v1/results/training/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 404

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_tuning_results(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - Create real tuning data in the database
        job_id = await dal.create_job(job_type="tuning", parameters={})
        await dal.update_job_status(job_id, "completed")
        config_id = await dal.create_or_get_config({
            "nn_model_config": {
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
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
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12
        })

        metrics = {"val_loss": 0.1, "train_loss": 0.2}
        result_id = await dal.create_tuning_result(
            job_id=job_id,
            config_id=config_id,
            metrics=metrics,
            duration=200,
        )

        # Act
        response = await async_api_client.get("/api/v1/results/tuning", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["result_id"] == result_id
        assert data[0]["job_id"] == job_id
        assert data[0]["config_id"] == config_id
        assert data[0]["metrics"] == metrics
        assert data[0]["duration"] == 200

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_tuning_result_by_id(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - Create real tuning data in the database
        job_id = await dal.create_job(job_type="tuning", parameters={})
        await dal.update_job_status(job_id, "completed")
        config_id = await dal.create_or_get_config({
            "nn_model_config": {
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
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
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12
        })

        metrics = {"val_loss": 0.1, "train_loss": 0.2}
        result_id = await dal.create_tuning_result(
            job_id=job_id,
            config_id=config_id,
            metrics=metrics,
            duration=200,
        )

        # Act
        response = await async_api_client.get(f"/api/v1/results/tuning/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["result_id"] == result_id
        assert data["job_id"] == job_id
        assert data["config_id"] == config_id
        assert data["metrics"] == metrics
        assert data["duration"] == 200

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_tuning_result_not_found(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - No data in database
        result_id = "not-found"

        # Act
        response = await async_api_client.get(f"/api/v1/results/tuning/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 404

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_tuning_results_with_query_params(self, async_api_client, dal, auth_header_name, auth_token):
        # Arrange - Create real tuning data in the database
        job_id = await dal.create_job(job_type="tuning", parameters={})
        await dal.update_job_status(job_id, "completed")
        config_id = await dal.create_or_get_config({
            "nn_model_config": {
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
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
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12
        })

        metrics = {"val_loss": 0.1, "train_loss": 0.2}
        result_id = await dal.create_tuning_result(
            job_id=job_id,
            config_id=config_id,
            metrics=metrics,
            duration=200,
        )

        # Act
        response = await async_api_client.get(
            "/api/v1/results/tuning?metric_name=val_loss&higher_is_better=false&limit=5",
            headers={auth_header_name: auth_token}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["result_id"] == result_id
