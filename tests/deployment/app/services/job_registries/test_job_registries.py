import importlib
import json
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import deployment.app.services.job_registries.input_preparator_registry as input_prep_reg
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)

# =====================
# Fixtures
# =====================

@pytest.fixture
def dummy_training_config():
    # Аналог create_training_params из conftest.py
    model_config = ModelConfig(
        num_encoder_layers=3,
        num_decoder_layers=2,
        decoder_output_dim=128,
        temporal_width_past=12,
        temporal_width_future=6,
        temporal_hidden_size_past=64,
        temporal_hidden_size_future=64,
        temporal_decoder_hidden=128,
        batch_size=32,
        dropout=0.2,
        use_reversible_instance_norm=True,
        use_layer_norm=True,
    )
    optimizer_config = OptimizerConfig(lr=0.001, weight_decay=0.0001)
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

# =====================
# Tests for prepare_training_inputs
# =====================

@pytest.mark.asyncio
async def test_prepare_training_inputs_creates_config_json(temp_workspace, dummy_training_config, caplog):
    logging.getLogger('deployment.app.services.job_registries.input_preparator_registry').propagate = True
    caplog.set_level('INFO')
    target_dir = Path(temp_workspace["input_dir"])
    job_id = "job-123"
    job_config = MagicMock()

    # Create a mock DAL for testing
    mock_dal = MagicMock()
    await input_prep_reg.prepare_training_inputs(job_id, dummy_training_config, target_dir, job_config, mock_dal)
    config_json_path = target_dir / "config.json"
    assert config_json_path.exists(), "config.json was not created"
    with open(config_json_path, encoding="utf-8") as f:
        data = json.load(f)
    dummy_training_config_dict = dummy_training_config.model_dump()
    dummy_training_config_dict['model_config'] = dummy_training_config_dict.pop('nn_model_config')
    assert data == dummy_training_config_dict
    assert f"[{job_id}] Saving training config to {config_json_path}" in caplog.text

# =====================
# Tests for prepare_tuning_inputs
# =====================

@pytest.mark.asyncio
async def test_prepare_tuning_inputs_creates_files_and_merges_params(temp_workspace, dummy_training_config, caplog, monkeypatch):
    logging.getLogger('deployment.app.services.job_registries.input_preparator_registry').propagate = True
    caplog.set_level('INFO')
    target_dir = Path(temp_workspace["input_dir"])
    job_id = "job-456"
    job_config = MagicMock()
    job_config.additional_params = {"mode": "fast", "time_budget_s": 123}

    # Mock get_settings to return an object with the correct structure
    mock_tuning_settings = MagicMock()
    mock_tuning_settings.model_dump.return_value = {"foo": "bar"}
    mock_tuning_settings.seed_configs_limit = 2

    mock_settings_obj = MagicMock()
    mock_settings_obj.tuning = mock_tuning_settings
    mock_settings_obj.default_metric = "mape"
    mock_settings_obj.default_metric_higher_is_better = False

    monkeypatch.setattr("deployment.app.services.job_registries.input_preparator_registry.get_settings", lambda: mock_settings_obj)
    # Create a mock DAL with get_top_configs method
    mock_dal = MagicMock()
    mock_dal.get_top_configs.return_value = [{"config": {"a": 1}}, {"config": {"b": 2}}]

    await input_prep_reg.prepare_tuning_inputs(job_id, dummy_training_config, target_dir, job_config, mock_dal)

    tuning_json_path = target_dir / "tuning_settings.json"
    initial_cfgs_path = target_dir / "initial_configs.json"
    assert tuning_json_path.exists()
    assert initial_cfgs_path.exists()
    with open(tuning_json_path, encoding="utf-8") as f:
        tuning_data = json.load(f)
    assert tuning_data["mode"] == "fast"
    assert tuning_data["time_budget_s"] == 123
    with open(initial_cfgs_path, encoding="utf-8") as f:
        cfgs = json.load(f)
    assert cfgs == [{"a": 1}, {"b": 2}]
    assert f"[{job_id}] Tuning settings saved to {tuning_json_path}" in caplog.text
    assert f"[{job_id}] initial_configs.json saved with 2 configs" in caplog.text

@pytest.mark.asyncio
async def test_prepare_tuning_inputs_empty_top_configs_logs_warning(temp_workspace, dummy_training_config, caplog, monkeypatch):
    logging.getLogger('deployment.app.services.job_registries.input_preparator_registry').propagate = True
    caplog.set_level('INFO')
    target_dir = Path(temp_workspace["input_dir"])
    job_id = "job-789"
    job_config = MagicMock()
    job_config.additional_params = {}

    # Mock get_settings to return an object with the correct structure
    mock_tuning_settings = MagicMock()
    mock_tuning_settings.model_dump.return_value = {"foo": "bar"}
    mock_tuning_settings.seed_configs_limit = 2

    mock_settings_obj = MagicMock()
    mock_settings_obj.tuning = mock_tuning_settings
    mock_settings_obj.default_metric = "mape"
    mock_settings_obj.default_metric_higher_is_better = False

    monkeypatch.setattr("deployment.app.services.job_registries.input_preparator_registry.get_settings", lambda: mock_settings_obj)
    # Create a mock DAL with get_top_configs method
    mock_dal = MagicMock()
    mock_dal.get_top_configs.return_value = []

    await input_prep_reg.prepare_tuning_inputs(job_id, dummy_training_config, target_dir, job_config, mock_dal)

    initial_cfgs_path = target_dir / "initial_configs.json"
    assert initial_cfgs_path.exists()
    with open(initial_cfgs_path, encoding="utf-8") as f:
        cfgs = json.load(f)
    assert cfgs == []
    assert f"[{job_id}] No starter configs found for initial_configs.json" in caplog.text

# =====================
# Tests for process_training_results
# =====================

@pytest.mark.asyncio
async def test_process_training_results_happy_path(temp_workspace, caplog, monkeypatch):
    import logging
    logging.getLogger('deployment.app.services.job_registries.result_processor_registry').propagate = True
    caplog.set_level('INFO')
    job_id = "job-321"
    ds_job_id = "ds-321"
    results_dir = temp_workspace["output_dir"]
    config = MagicMock()
    metrics_data = {"mape": 1.0}
    output_files = {"model": os.path.join(results_dir, "model.onnx"), "predictions": os.path.join(results_dir, "predictions.csv"), "metrics": metrics_data}
    polls = 5
    poll_interval = 1.0
    config_id = "cfg-1"
    Path(output_files["model"]).write_text("fake model")
    Path(output_files["predictions"]).write_text("barcode,pred\n123,10")
    mock_save_model = AsyncMock()
    mock_save_preds = MagicMock()
    mock_create_tr = MagicMock()
    mock_update_status = MagicMock()
    mock_cleanup = AsyncMock()
    monkeypatch.setattr("deployment.app.services.datasphere_service.save_model_file_and_db", mock_save_model)
    monkeypatch.setattr("deployment.app.services.datasphere_service.save_predictions_to_db", mock_save_preds)
    monkeypatch.setattr("deployment.app.db.database.create_training_result", mock_create_tr)
    monkeypatch.setattr("deployment.app.db.database.update_job_status", mock_update_status)
    monkeypatch.setattr("deployment.app.services.datasphere_service._perform_model_cleanup", mock_cleanup)

    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    importlib.reload(result_proc_reg)
    mock_save_model.return_value = "model-xyz"
    mock_save_preds.return_value = {"result_id": "res-1", "predictions_count": 1}
    mock_create_tr.return_value = "tr-1"

    # Create a mock DAL for testing
    mock_dal = MagicMock()
    mock_dal.get_job.return_value = {"parameters": '{"prediction_month": "2023-01-01"}'}
    await result_proc_reg.process_training_results(
        job_id=job_id,
        ds_job_id=ds_job_id,
        results_dir=results_dir,
        config=config,
        metrics_data=metrics_data,
        output_files=output_files,
        polls=polls,
        poll_interval=poll_interval,
        config_id=config_id,
        dal=mock_dal,
    )
    mock_save_model.assert_awaited_once_with(
        job_id=job_id,
        model_path=output_files["model"],
        ds_job_id=ds_job_id,
        config=config,
        metrics_data=metrics_data,
        dal=mock_dal,
    )
    mock_save_preds.assert_called_once_with(
        predictions_path=output_files["predictions"],
        job_id=job_id,
        model_id="model-xyz",
        dal=mock_dal,
    )
    # The functions are now called through the DAL, so we check the DAL method calls
    mock_dal.create_training_result.assert_called_once()
    mock_dal.update_job_status.assert_called()
    mock_cleanup.assert_awaited_once()
    assert "Job completed. DS Job ID: ds-321." in caplog.text

@pytest.mark.asyncio
async def test_process_training_results_missing_model_logs_warning(temp_workspace, caplog, monkeypatch):
    import logging
    logging.getLogger('deployment.app.services.job_registries.result_processor_registry').propagate = True
    caplog.set_level('INFO')
    job_id = "job-322"
    ds_job_id = "ds-322"
    results_dir = temp_workspace["output_dir"]
    config = MagicMock()
    metrics_data = {"mape": 1.0}
    output_files = {"model": None, "predictions": os.path.join(results_dir, "predictions.csv"), "metrics": metrics_data}
    polls = 5
    poll_interval = 1.0
    config_id = "cfg-2"
    Path(output_files["predictions"]).write_text("barcode,pred\n123,10")
    mock_save_model = AsyncMock()
    mock_save_preds = MagicMock()
    mock_create_tr = MagicMock()
    mock_update_status = MagicMock()
    mock_cleanup = AsyncMock()
    monkeypatch.setattr("deployment.app.services.datasphere_service.save_model_file_and_db", mock_save_model)
    monkeypatch.setattr("deployment.app.services.datasphere_service.save_predictions_to_db", mock_save_preds)
    monkeypatch.setattr("deployment.app.db.database.create_training_result", mock_create_tr)
    monkeypatch.setattr("deployment.app.db.database.update_job_status", mock_update_status)
    monkeypatch.setattr("deployment.app.services.datasphere_service._perform_model_cleanup", mock_cleanup)
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    importlib.reload(result_proc_reg)

    # Create a mock DAL for testing
    mock_dal = MagicMock()
    mock_dal.get_job.return_value = {"parameters": '{"prediction_month": "2023-01-01"}'}
    await result_proc_reg.process_training_results(
        job_id=job_id,
        ds_job_id=ds_job_id,
        results_dir=results_dir,
        config=config,
        metrics_data=metrics_data,
        output_files=output_files,
        polls=polls,
        poll_interval=poll_interval,
        config_id=config_id,
        dal=mock_dal,
    )
    mock_save_model.assert_not_awaited()
    mock_save_preds.assert_not_called()
    # The functions are now called through the DAL, so we check the DAL method calls
    mock_dal.create_training_result.assert_called_once()
    mock_dal.update_job_status.assert_called()
    mock_cleanup.assert_not_awaited()
    assert "Model file not found in results." in caplog.text

# =====================
# Tests for process_tuning_results
# =====================

def make_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.mark.asyncio
async def test_process_tuning_results_happy_path(temp_workspace, caplog, monkeypatch):
    import logging
    logging.getLogger('deployment.app.services.job_registries.result_processor_registry').propagate = True
    caplog.set_level('INFO')
    job_id = "job-555"
    results_dir = temp_workspace["output_dir"]
    best_cfgs = [{"a": 1}, {"b": 2}]
    metrics = [{"score": 0.1}, {"score": 0.2}]
    make_json_file(os.path.join(results_dir, "best_configs.json"), best_cfgs)
    make_json_file(os.path.join(results_dir, "metrics.json"), metrics)
    output_files = {"configs": os.path.join(results_dir, "best_configs.json"), "metrics": metrics}
    polls = 3
    poll_interval = 1.0
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    # Create a mock DAL with the required methods
    mock_dal = MagicMock()
    mock_dal.create_or_get_config.return_value = "cfg-1"
    mock_dal.update_job_status = MagicMock()

    # Mock the database functions
    mock_create_tr = MagicMock()
    mock_update_status = MagicMock()
    monkeypatch.setattr("deployment.app.db.database.create_tuning_result", mock_create_tr)
    monkeypatch.setattr("deployment.app.db.database.update_job_status", mock_update_status)

    result_proc_reg.process_tuning_results(
        job_id=job_id,
        results_dir=results_dir,
        metrics_data=metrics,
        output_files=output_files,
        polls=polls,
        poll_interval=poll_interval,
        dal=mock_dal,
    )
    assert mock_dal.create_or_get_config.call_count == 2
    assert mock_dal.create_tuning_result.call_count == 2
    mock_dal.update_job_status.assert_called()

def test_process_tuning_results_missing_best_configs_raises_and_fails_job(temp_workspace, caplog, monkeypatch):
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    job_id = "job-556"
    results_dir = temp_workspace["output_dir"]
    metrics = [{"score": 0.1}]
    output_files = {"configs": None, "metrics": metrics}
    polls = 1
    poll_interval = 1.0
    # Create a mock DAL for testing
    mock_dal = MagicMock()
    mock_update_status = MagicMock()
    monkeypatch.setattr("deployment.app.db.database.update_job_status", mock_update_status)
    with pytest.raises(RuntimeError):
        result_proc_reg.process_tuning_results(
            job_id=job_id,
            results_dir=results_dir,
            metrics_data=metrics,
            output_files=output_files,
            polls=polls,
            poll_interval=poll_interval,
            dal=mock_dal,
        )
    mock_dal.update_job_status.assert_called()
    assert "best_configs.json (role: configs) not found in results" in caplog.text

def test_process_tuning_results_invalid_metrics_type_fails_job(temp_workspace, caplog, monkeypatch):
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    job_id = "job-557"
    results_dir = temp_workspace["output_dir"]
    best_cfgs = [{"a": 1}]
    make_json_file(os.path.join(results_dir, "best_configs.json"), best_cfgs)
    output_files = {"configs": os.path.join(results_dir, "best_configs.json"), "metrics": {"score": 0.1}}
    polls = 1
    poll_interval = 1.0
    # Create a mock DAL for testing
    mock_dal = MagicMock()
    mock_update_status = MagicMock()
    monkeypatch.setattr("deployment.app.db.database.update_job_status", mock_update_status)
    with pytest.raises(RuntimeError):
        result_proc_reg.process_tuning_results(
            job_id=job_id,
            results_dir=results_dir,
            metrics_data={"score": 0.1},
            output_files=output_files,
            polls=polls,
            poll_interval=poll_interval,
            dal=mock_dal,
        )
    mock_dal.update_job_status.assert_called()
    assert "metrics.json must contain a list of metric dicts" in caplog.text

def test_process_tuning_results_metrics_length_mismatch_fails_job(temp_workspace, caplog, monkeypatch):
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    job_id = "job-558"
    results_dir = temp_workspace["output_dir"]
    best_cfgs = [{"a": 1}, {"b": 2}]
    metrics = [{"score": 0.1}]
    make_json_file(os.path.join(results_dir, "best_configs.json"), best_cfgs)
    make_json_file(os.path.join(results_dir, "metrics.json"), metrics)
    output_files = {"configs": os.path.join(results_dir, "best_configs.json"), "metrics": metrics}
    polls = 1
    poll_interval = 1.0
    # Create a mock DAL for testing
    mock_dal = MagicMock()
    mock_update_status = MagicMock()
    monkeypatch.setattr("deployment.app.db.database.update_job_status", mock_update_status)
    with pytest.raises(RuntimeError):
        result_proc_reg.process_tuning_results(
            job_id=job_id,
            results_dir=results_dir,
            metrics_data=metrics,
            output_files=output_files,
            polls=polls,
            poll_interval=poll_interval,
            dal=mock_dal,
        )
    mock_dal.update_job_status.assert_called()
    assert "metrics.json list length" in caplog.text
