import os
import pytest
import json
import sqlite3
import tempfile
import pandas as pd
from datetime import datetime
import uuid
from pathlib import Path
import sys
from unittest.mock import MagicMock, PropertyMock
import numpy as np

# This is a more robust way to handle global mocks.
# It's a fixture that can be explicitly used by tests that need it,
# and it automatically handles setup and teardown, preventing contamination.
@pytest.fixture(scope="function")
def mock_global_utils(monkeypatch):
    """
    Mocks global utility modules (misc, retry_monitor) for a single test function.
    This prevents test contamination that can occur with module-level sys.modules patching.
    """
    # --- Mock for deployment.app.utils.misc ---
    mock_misc = MagicMock()
    mock_misc.camel_to_snake.side_effect = lambda s: s.lower()
    monkeypatch.setitem(sys.modules, 'deployment.app.utils.misc', mock_misc)

    # --- Mock for deployment.app.utils.retry_monitor ---
    mock_retry_monitor_module = MagicMock()
    mock_retry_monitor_class = MagicMock()
    mock_retry_monitor_instance = MagicMock()
    mock_retry_monitor_instance.get_statistics.return_value = {
        "total_retries": 0, "successful_retries": 0, "exhausted_retries": 0,
        "successful_after_retry": 0, "high_failure_operations": [],
        "alerted_operations": [], "alert_thresholds": {}, "operation_stats": {},
        "exception_stats": {}, "timestamp": "2021-01-01T00:00:00"
    }
    mock_retry_monitor_class.return_value = mock_retry_monitor_instance
    mock_retry_monitor_module.RetryMonitor = mock_retry_monitor_class
    mock_retry_monitor_module.retry_monitor = mock_retry_monitor_instance
    mock_retry_monitor_module.DEFAULT_PERSISTENCE_PATH = None
    mock_retry_monitor_module.record_retry = MagicMock()
    mock_retry_monitor_module.get_retry_statistics = MagicMock(return_value=mock_retry_monitor_instance.get_statistics.return_value)
    mock_retry_monitor_module.reset_retry_statistics = MagicMock()
    mock_retry_monitor_module.get_high_failure_operations = MagicMock(return_value=set())
    mock_retry_monitor_module.set_alert_threshold = MagicMock()
    mock_retry_monitor_module.register_alert_handler = MagicMock()
    mock_retry_monitor_module.get_alerted_operations = MagicMock(return_value=set())
    mock_retry_monitor_module.log_alert_handler = MagicMock()
    monkeypatch.setitem(sys.modules, 'deployment.app.utils.retry_monitor', mock_retry_monitor_module)
    
    # Yield the mocks in a dictionary in case any test needs to access them
    yield {
        "misc": mock_misc,
        "retry_monitor": mock_retry_monitor_module
    }
    
    # monkeypatch automatically handles teardown, restoring original modules

# Add the project root to sys.path to fix imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deployment.app.models.api_models import (
    JobStatus, JobType, TrainingConfig,
    ModelConfig, OptimizerConfig, LRSchedulerConfig,
    TrainingDatasetConfig
)

# Common test data fixtures
@pytest.fixture
def sample_predictions_data():
    """Returns a sample of prediction data for use in various tests"""
    return {
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
@pytest.fixture
def create_training_params_fn():
    """
    Returns a function to create a TrainingParams object with specified parameters.
    Common utility for all tests.
    """
    def _create_training_params(base_params=None):
        base_params = base_params or {}
        
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
        
        optimizer_config = OptimizerConfig(
            lr=base_params.get('learning_rate', 0.001),
            weight_decay=0.0001
        )
        
        lr_shed_config = LRSchedulerConfig(
            T_0=10,
            T_mult=2
        )
        
        train_ds_config = TrainingDatasetConfig(
            alpha=0.05,
            span=12
        )
        
        return TrainingConfig(
            model_config=model_config,
            optimizer_config=optimizer_config,
            lr_shed_config=lr_shed_config,
            train_ds_config=train_ds_config,
            lags=12,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
    
    return _create_training_params

