import os
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from deployment.app.config import DataRetentionSettings
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.data_retention import (
    cleanup_old_historical_data,
    cleanup_old_models,
    cleanup_old_predictions,
    run_cleanup_job,
)


@pytest.fixture
async def test_data_retention_env(dal, tmp_path):
    """Set up test environment for data retention tests"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Mock settings
    mock_retention_settings = DataRetentionSettings(
        sales_retention_days=365,
        stock_retention_days=365,
        prediction_retention_days=30,
        models_to_keep=2,
        inactive_model_retention_days=15,
        cleanup_enabled=True,
    )
    
    with patch("deployment.app.db.data_retention.get_settings") as mock_get_settings:
        mock_settings_object = MagicMock()
        mock_settings_object.data_retention = mock_retention_settings
        mock_settings_object.default_metric = "val_MIC"
        mock_settings_object.default_metric_higher_is_better = True
        mock_settings_object.models_dir = str(model_dir)
        mock_get_settings.return_value = mock_settings_object

        yield {
            "dal": dal,
            "model_dir": model_dir,
            "mock_settings": mock_settings_object,
        }


async def _create_test_models(dal, model_dir):
    """Create test model records and files"""
    now = datetime.now()

    # Create model files
    model_files = [
        ("model1.pt", now - timedelta(days=5)),
        ("model2.pt", now - timedelta(days=10)),
        ("model3.pt", now - timedelta(days=20)),
        ("model4.pt", now - timedelta(days=30)),
        ("inactive_model1.pt", now - timedelta(days=10)),
        ("inactive_model2.pt", now - timedelta(days=20)),
    ]

    for filename, _ in model_files:
        (model_dir / filename).write_text("Test model content")

    # Create jobs first (required for foreign key constraints)
    job_ids = []
    for i in range(1, 7):
        job_id = await dal.create_job(job_type=f"training_job_{i}", parameters={"test": "params"})
        job_ids.append(job_id)

    # Create config first (required for foreign key constraints)
    config_id = await dal.create_or_get_config(
        {
            "nn_model_config": {
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "decoder_output_dim": 64,
                "temporal_width_past": 10,
                "temporal_width_future": 5,
                "temporal_hidden_size_past": 64,
                "temporal_hidden_size_future": 64,
                "temporal_decoder_hidden": 64,
                "batch_size": 32,
                "dropout": 0.1,
                "use_reversible_instance_norm": True,
                "use_layer_norm": True,
            },
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.5, "span": 12},
            "lags": 12,
        },
        is_active=True,
    )

    # Insert model records using DAL
    await dal.create_model_record(
        model_id="model1",
        job_id=job_ids[0],
        model_path=str(model_dir / "model1.pt"),
        created_at=(now - timedelta(days=5)),
        metadata={"size": 1000},
        is_active=True,
    )
    await dal.create_model_record(
        model_id="model2",
        job_id=job_ids[1],
        model_path=str(model_dir / "model2.pt"),
        created_at=(now - timedelta(days=10)),
        metadata={"size": 1000},
        is_active=True,
    )
    await dal.create_model_record(
        model_id="model3",
        job_id=job_ids[2],
        model_path=str(model_dir / "model3.pt"),
        created_at=(now - timedelta(days=20)),
        metadata={"size": 1000},
        is_active=True,
    )
    await dal.create_model_record(
        model_id="model4",
        job_id=job_ids[3],
        model_path=str(model_dir / "model4.pt"),
        created_at=(now - timedelta(days=30)),
        metadata={"size": 1000},
        is_active=True,
    )
    await dal.create_model_record(
        model_id="inactive_model1",
        job_id=job_ids[4],
        model_path=str(model_dir / "inactive_model1.pt"),
        created_at=(now - timedelta(days=10)),
        metadata={"size": 1000},
        is_active=False,
    )
    await dal.create_model_record(
        model_id="inactive_model2",
        job_id=job_ids[5],
        model_path=str(model_dir / "inactive_model2.pt"),
        created_at=(now - timedelta(days=20)),
        metadata={"size": 1000},
        is_active=False,
    )

    # Associate models with config set and add metrics using DAL
    await dal.create_training_result(
        job_id=job_ids[0],
        model_id="model1",
        config_id=config_id,
        metrics={"val_MIC": 0.95},
        duration=3600,
    )
    await dal.create_training_result(
        job_id=job_ids[1],
        model_id="model2",
        config_id=config_id,
        metrics={"val_MIC": 0.90},
        duration=3600,
    )
    await dal.create_training_result(
        job_id=job_ids[2],
        model_id="model3",
        config_id=config_id,
        metrics={"val_MIC": 0.85},
        duration=3600,
    )
    await dal.create_training_result(
        job_id=job_ids[3],
        model_id="model4",
        config_id=config_id,
        metrics={"val_MIC": 0.80},
        duration=3600,
    )

async def _create_test_predictions(dal, model_dir):
    """Create test prediction records"""
    now = datetime.now()

    # Create jobs and results first
    job1_id = await dal.create_job(job_type="prediction_job_1", parameters={"test": "params"})
    job2_id = await dal.create_job(job_type="prediction_job_2", parameters={"test": "params"})

    # Create models if they don't exist
    try:
        await dal.create_model_record(
            model_id="model1",
            job_id=job1_id,
            model_path=str(model_dir / "model1.pt"),
            created_at=now,
            is_active=True,
        )
    except Exception:  # May already exist
        pass
    try:
        await dal.create_model_record(
            model_id="model2",
            job_id=job2_id,
            model_path=str(model_dir / "model2.pt"),
            created_at=now,
            is_active=True,
        )
    except Exception:  # May already exist
        pass

    from datetime import date
    # Convert to date object (first day of month)
    prediction_month = date(now.year, now.month, 1)
    
    result1_id = await dal.create_prediction_result(
        job_id=job1_id,
        prediction_month=prediction_month,
        model_id="model1",
        output_path="/tmp/test1",
        summary_metrics={"mape": 10.5},
    )
    result2_id = await dal.create_prediction_result(
        job_id=job2_id,
        prediction_month=prediction_month,
        model_id="model2",
        output_path="/tmp/test2",
        summary_metrics={"mape": 11.5},
    )

    # Create predictions using insert_predictions (which expects DataFrame)
    # Recent predictions (10 records)
    recent_dates = [now - timedelta(days=i) for i in range(10)]
    recent_df = pd.DataFrame({
        'barcode': [f'barcode{i}' for i in range(10)],
        'artist': [f'artist{i}' for i in range(10)],
        'album': [f'album{i}' for i in range(10)],
        'cover_type': ['vinyl'] * 10,
        'price_category': ['medium'] * 10,
        'release_type': ['studio'] * 10,
        'recording_decade': ['1990s'] * 10,
        'release_decade': ['1990s'] * 10,
        'style': ['rock'] * 10,
        'recording_year': [1995] * 10,
        '0.05': [10.5] * 10,
        '0.25': [15.2] * 10,
        '0.5': [20.1] * 10,
        '0.75': [25.8] * 10,
        '0.95': [30.3] * 10,
    })
    await dal.insert_predictions(
        result_id=result1_id,
        model_id="model1",
        prediction_month=recent_dates[0].date(),
        df=recent_df,
    )

    # Old predictions (10 records)
    old_dates = [now - timedelta(days=40 + i) for i in range(10)]
    old_df = pd.DataFrame({
        'barcode': [f'barcode{i+100}' for i in range(10)],
        'artist': [f'artist{i+100}' for i in range(10)],
        'album': [f'album{i+100}' for i in range(10)],
        'cover_type': ['vinyl'] * 10,
        'price_category': ['medium'] * 10,
        'release_type': ['studio'] * 10,
        'recording_decade': ['1990s'] * 10,
        'release_decade': ['1990s'] * 10,
        'style': ['rock'] * 10,
        'recording_year': [1995] * 10,
        '0.05': [11.5] * 10,
        '0.25': [16.2] * 10,
        '0.5': [21.1] * 10,
        '0.75': [26.8] * 10,
        '0.95': [31.3] * 10,
    })
    await dal.insert_predictions(
        result_id=result2_id,
        model_id="model2",
        prediction_month=old_dates[0].date(),
        df=old_df,
    )

async def _create_test_historical_data(dal):
    """Create test historical data records directly."""
    now = datetime.now()

    # 1. Create multi-index entries
    multiindex_tuples = [
        (
            f"barcode_hist_{i}",
            f"artist_hist_{i}",
            f"album_hist_{i}",
            "CD",
            "low",
            "live",
            "2010s",
            "2010s",
            "electronic",
            2015,
        )
        for i in range(200)
    ]
    multiindex_ids = await dal.get_or_create_multiindex_ids_batch(multiindex_tuples)

    sales_params = []
    movement_params = []

    # 2. Generate data points
    # Recent data (100 records)
    for i in range(10):  # 10 dates
        date_obj = (now - timedelta(days=i * 30)).date()
        for j in range(10):  # 10 products per date
            midx_id = multiindex_ids[i * 10 + j]
            sales_params.append((midx_id, date_obj, 10.0 + j))
            movement_params.append((midx_id, date_obj, 1.0 + j))

    # Old data (100 records)
    for i in range(10):  # 10 dates
        date_obj = (now - timedelta(days=800 + i * 30)).date()
        for j in range(10):  # 10 products per date
            midx_id = multiindex_ids[100 + i * 10 + j]
            sales_params.append((midx_id, date_obj, 5.0 + j))
            movement_params.append((midx_id, date_obj, 1.0 + j))

    # 3. Insert data directly
    await dal.insert_features_batch("fact_sales", sales_params)
    await dal.insert_features_batch("fact_stock_movement", movement_params)


@pytest.mark.asyncio
async def test_cleanup_old_predictions(test_data_retention_env):
    """Test cleaning up old predictions"""
    dal = test_data_retention_env["dal"]
    model_dir = test_data_retention_env["model_dir"]
    
    await _create_test_predictions(dal, model_dir)

    before_count = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_predictions", fetchall=False
    )
    assert before_count["count"] == 20

    count = await cleanup_old_predictions(days_to_keep=30, dal=dal)

    after_count = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_predictions", fetchall=False
    )

    assert count == 10
    assert after_count["count"] == 10


@pytest.mark.asyncio
async def test_cleanup_old_historical_data(test_data_retention_env):
    """Test cleaning up old historical data"""
    dal = test_data_retention_env["dal"]
    
    await _create_test_historical_data(dal)

    sales_before = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_sales", fetchall=False
    )
    changes_before = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_stock_movement", fetchall=False
    )

    assert sales_before["count"] == 200
    assert changes_before["count"] == 200

    result = await cleanup_old_historical_data(
        sales_days_to_keep=365, stock_days_to_keep=365, dal=dal
    )

    sales_after = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_sales", fetchall=False
    )
    changes_after = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_stock_movement", fetchall=False
    )

    assert result["sales"] == 100
    assert result["stock_movement"] == 100
    assert sales_after["count"] == 100
    assert changes_after["count"] == 100

@pytest.mark.asyncio
async def test_cleanup_old_models(test_data_retention_env):
    """Test cleaning up old models, including check for linked predictions."""
    dal = test_data_retention_env["dal"]
    model_dir = test_data_retention_env["model_dir"]
    
    await _create_test_models(dal, model_dir)

    # Add predictions linked to models that would otherwise be deleted
    job_id = await dal.create_job(job_type="prediction_job_test", parameters={})
    from datetime import date
    result_id = await dal.create_prediction_result(
        job_id=job_id,
        prediction_month=date(2025, 1, 1),  # Use date object instead of string
        model_id="model4",
        output_path="/tmp/pred_test1",
        summary_metrics={"test": 1.0},
    )
    
    # Create a DataFrame for insert_predictions
    pred_df_model4 = pd.DataFrame({
        'barcode': ["p_bc"], 'artist': ["p_art"], 'album': ["p_alb"],
        'cover_type': ["a"], 'price_category': ["b"], 'release_type': ["c"],
        'recording_decade': ["d"], 'release_decade': ["e"], 'style': ["f"],
        'recording_year': [2000],
        '0.05': [1], '0.25': [2], '0.5': [3], '0.75': [4], '0.95': [5]
    })
    await dal.insert_predictions(
        result_id=result_id,
        model_id="model4",
        prediction_month=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        df=pred_df_model4,
    )

    job_id2 = await dal.create_job(job_type="prediction_job_test2", parameters={})
    result_id2 = await dal.create_prediction_result(
        job_id=job_id2,
        prediction_month=date(2025, 1, 1),  # Use date object instead of string
        model_id="inactive_model2",
        output_path="/tmp/pred_test2",
        summary_metrics={"test": 1.0},
    )
    
    pred_df_inactive_model2 = pd.DataFrame({
        'barcode': ["p_bc"], 'artist': ["p_art"], 'album': ["p_alb"],
        'cover_type': ["a"], 'price_category': ["b"], 'release_type': ["c"],
        'recording_decade': ["d"], 'release_decade': ["e"], 'style': ["f"],
        'recording_year': [2000],
        '0.05': [1], '0.25': [2], '0.5': [3], '0.75': [4], '0.95': [5]
    })
    await dal.insert_predictions(
        result_id=result_id2,
        model_id="inactive_model2",
        prediction_month=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        df=pred_df_inactive_model2,
    )

    before_count = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM models", fetchall=False
    )
    assert before_count["count"] == 6

    deleted_model_ids = await cleanup_old_models(
        models_to_keep=2, inactive_days_to_keep=15, dal=dal
    )

    after_count = await dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM models", fetchall=False
    )

    # Only model3 should be deleted. model4 and inactive_model2 are protected by predictions.
    assert len(deleted_model_ids) == 1
    assert "model3" in deleted_model_ids
    assert after_count["count"] == 5

    remaining_models_q = await dal.execute_raw_query(
        "SELECT model_id FROM models", fetchall=True
    )
    remaining_models = {row["model_id"] for row in remaining_models_q}
    assert "model1" in remaining_models
    assert "model2" in remaining_models
    assert "inactive_model1" in remaining_models
    assert "model4" in remaining_models
    assert "inactive_model2" in remaining_models


@pytest.mark.asyncio
@patch("deployment.app.db.data_retention.cleanup_old_predictions")
@patch("deployment.app.db.data_retention.cleanup_old_models")
@patch("deployment.app.db.data_retention.cleanup_old_historical_data")
async def test_run_cleanup_job(
    mock_cleanup_historical, mock_cleanup_models, mock_cleanup_predictions, test_data_retention_env
):
    """Test running the complete cleanup job"""
    dal = test_data_retention_env["dal"]
    
    mock_cleanup_predictions.return_value = 10
    mock_cleanup_models.return_value = ["model3"]
    mock_cleanup_historical.return_value = {"sales": 50, "stock_movement": 50}

    await run_cleanup_job(dal=dal)

    mock_cleanup_predictions.assert_called_once()
    mock_cleanup_models.assert_called_once()
    mock_cleanup_historical.assert_called_once()