import json
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import logging
import importlib

import deployment.app.services.job_registries.input_preparator_registry as input_prep_reg
from deployment.app.models.api_models import TrainingConfig, ModelConfig, OptimizerConfig, LRSchedulerConfig, TrainingDatasetConfig

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
    
    await input_prep_reg.prepare_training_inputs(job_id, dummy_training_config, target_dir, job_config)
    config_json_path = target_dir / "config.json"
    assert config_json_path.exists(), "config.json was not created"
    with open(config_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == dummy_training_config.model_dump()
    assert f"[{job_id}] Saving training config to {config_json_path}" in caplog.text

# =====================
# Tests for prepare_tuning_inputs
# =====================

@pytest.mark.asyncio
async def test_prepare_tuning_inputs_creates_files_and_merges_params(temp_workspace, dummy_training_config, caplog):
    logging.getLogger('deployment.app.services.job_registries.input_preparator_registry').propagate = True
    caplog.set_level('INFO')
    target_dir = Path(temp_workspace["input_dir"])
    job_id = "job-456"
    job_config = MagicMock()
    job_config.additional_params = {"mode": "fast", "time_budget_s": 123}
    
    # Mock get_settings and get_top_configs
    fake_tuning = MagicMock()
    fake_tuning.model_dump.return_value = {"foo": "bar"}
    fake_tuning.seed_configs_limit = 2
    fake_settings = MagicMock()
    fake_settings.tuning = fake_tuning
    fake_settings.default_metric = "mape"
    fake_settings.default_metric_higher_is_better = False
    
    with patch("deployment.app.services.job_registries.input_preparator_registry.get_settings", return_value=fake_settings), \
         patch("deployment.app.services.job_registries.input_preparator_registry.get_top_configs", return_value=[{"a": 1}, {"b": 2}]):
        await input_prep_reg.prepare_tuning_inputs(job_id, dummy_training_config, target_dir, job_config)
    tuning_json_path = target_dir / "tuning_settings.json"
    initial_cfgs_path = target_dir / "initial_configs.json"
    assert tuning_json_path.exists()
    assert initial_cfgs_path.exists()
    with open(tuning_json_path, "r", encoding="utf-8") as f:
        tuning_data = json.load(f)
    assert tuning_data["mode"] == "fast"
    assert tuning_data["time_budget_s"] == 123
    with open(initial_cfgs_path, "r", encoding="utf-8") as f:
        cfgs = json.load(f)
    assert cfgs == [{"a": 1}, {"b": 2}]
    assert f"[{job_id}] Tuning settings saved to {tuning_json_path}" in caplog.text
    assert f"[{job_id}] initial_configs.json saved with 2 configs" in caplog.text

@pytest.mark.asyncio
async def test_prepare_tuning_inputs_empty_top_configs_logs_warning(temp_workspace, dummy_training_config, caplog):
    logging.getLogger('deployment.app.services.job_registries.input_preparator_registry').propagate = True
    caplog.set_level('INFO')
    target_dir = Path(temp_workspace["input_dir"])
    job_id = "job-789"
    job_config = MagicMock()
    job_config.additional_params = {}
    fake_tuning = MagicMock()
    fake_tuning.model_dump.return_value = {"foo": "bar"}
    fake_tuning.seed_configs_limit = 2
    fake_settings = MagicMock()
    fake_settings.tuning = fake_tuning
    fake_settings.default_metric = "mape"
    fake_settings.default_metric_higher_is_better = False
    with patch("deployment.app.services.job_registries.input_preparator_registry.get_settings", return_value=fake_settings), \
         patch("deployment.app.services.job_registries.input_preparator_registry.get_top_configs", return_value=[]):
        await input_prep_reg.prepare_tuning_inputs(job_id, dummy_training_config, target_dir, job_config)
    initial_cfgs_path = target_dir / "initial_configs.json"
    assert initial_cfgs_path.exists()
    with open(initial_cfgs_path, "r", encoding="utf-8") as f:
        cfgs = json.load(f)
    assert cfgs == []
    assert f"[{job_id}] No starter configs found for initial_configs.json" in caplog.text

# =====================
# Tests for process_training_results
# =====================

@pytest.mark.asyncio
async def test_process_training_results_happy_path(temp_workspace, caplog):
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
    with patch("deployment.app.services.datasphere_service.save_model_file_and_db", new_callable=AsyncMock) as mock_save_model, \
         patch("deployment.app.services.datasphere_service.save_predictions_to_db") as mock_save_preds, \
         patch("deployment.app.db.database.create_training_result") as mock_create_tr, \
         patch("deployment.app.db.database.update_job_status") as mock_update_status, \
         patch("deployment.app.services.datasphere_service._perform_model_cleanup", new_callable=AsyncMock) as mock_cleanup:
        import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
        importlib.reload(result_proc_reg)
        mock_save_model.return_value = "model-xyz"
        mock_save_preds.return_value = {"result_id": "res-1", "predictions_count": 1}
        mock_create_tr.return_value = "tr-1"
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
        )
        mock_save_model.assert_awaited_once_with(
            job_id=job_id,
            model_path=output_files["model"],
            ds_job_id=ds_job_id,
            config=config,
            metrics_data=metrics_data,
        )
        mock_save_preds.assert_called_once_with(
            predictions_path=output_files["predictions"],
            job_id=job_id,
            model_id="model-xyz",
        )
        mock_create_tr.assert_called_once()
        mock_update_status.assert_called()
        mock_cleanup.assert_awaited_once()
        assert "Job completed. DS Job ID: ds-321." in caplog.text

@pytest.mark.asyncio
async def test_process_training_results_missing_model_logs_warning(temp_workspace, caplog):
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
    with patch("deployment.app.services.datasphere_service.save_model_file_and_db", new_callable=AsyncMock) as mock_save_model, \
         patch("deployment.app.services.datasphere_service.save_predictions_to_db") as mock_save_preds, \
         patch("deployment.app.db.database.create_training_result") as mock_create_tr, \
         patch("deployment.app.db.database.update_job_status") as mock_update_status, \
         patch("deployment.app.services.datasphere_service._perform_model_cleanup", new_callable=AsyncMock) as mock_cleanup:
        import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
        importlib.reload(result_proc_reg)
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
        )
        mock_save_model.assert_not_awaited()
        mock_save_preds.assert_not_called()
        mock_create_tr.assert_called_once()
        mock_update_status.assert_called()
        mock_cleanup.assert_not_awaited()
        assert "Model file not found in results." in caplog.text

# =====================
# Tests for process_tuning_results
# =====================

def make_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.mark.asyncio
async def test_process_tuning_results_happy_path(temp_workspace, caplog):
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
    with patch.object(result_proc_reg, "create_or_get_config", return_value="cfg-1") as mock_create_cfg, \
         patch.object(result_proc_reg, "create_tuning_result") as mock_create_tr, \
         patch.object(result_proc_reg, "update_job_status") as mock_update_status:
        result_proc_reg.process_tuning_results(
            job_id=job_id,
            results_dir=results_dir,
            metrics_data=metrics,
            output_files=output_files,
            polls=polls,
            poll_interval=poll_interval,
        )
        assert mock_create_cfg.call_count == 2
        assert mock_create_tr.call_count == 2
        mock_update_status.assert_called()

def test_process_tuning_results_missing_best_configs_raises_and_fails_job(temp_workspace, caplog):
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    job_id = "job-556"
    results_dir = temp_workspace["output_dir"]
    metrics = [{"score": 0.1}]
    output_files = {"configs": None, "metrics": metrics}
    polls = 1
    poll_interval = 1.0
    with patch.object(result_proc_reg, "update_job_status") as mock_update_status:
        with pytest.raises(RuntimeError):
            result_proc_reg.process_tuning_results(
                job_id=job_id,
                results_dir=results_dir,
                metrics_data=metrics,
                output_files=output_files,
                polls=polls,
                poll_interval=poll_interval,
            )
        mock_update_status.assert_called()
        assert "best_configs.json (role: configs) not found in results" in caplog.text

def test_process_tuning_results_invalid_metrics_type_fails_job(temp_workspace, caplog):
    import deployment.app.services.job_registries.result_processor_registry as result_proc_reg
    job_id = "job-557"
    results_dir = temp_workspace["output_dir"]
    best_cfgs = [{"a": 1}]
    make_json_file(os.path.join(results_dir, "best_configs.json"), best_cfgs)
    output_files = {"configs": os.path.join(results_dir, "best_configs.json"), "metrics": {"score": 0.1}}
    polls = 1
    poll_interval = 1.0
    with patch.object(result_proc_reg, "update_job_status") as mock_update_status:
        with pytest.raises(RuntimeError):
            result_proc_reg.process_tuning_results(
                job_id=job_id,
                results_dir=results_dir,
                metrics_data={"score": 0.1},
                output_files=output_files,
                polls=polls,
                poll_interval=poll_interval,
            )
        mock_update_status.assert_called()
        assert "metrics.json must contain a list of metric dicts" in caplog.text

def test_process_tuning_results_metrics_length_mismatch_fails_job(temp_workspace, caplog):
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
    with patch.object(result_proc_reg, "update_job_status") as mock_update_status:
        with pytest.raises(RuntimeError):
            result_proc_reg.process_tuning_results(
                job_id=job_id,
                results_dir=results_dir,
                metrics_data=metrics,
                output_files=output_files,
                polls=polls,
                poll_interval=poll_interval,
            )
        mock_update_status.assert_called()
        assert "metrics.json list length" in caplog.text 