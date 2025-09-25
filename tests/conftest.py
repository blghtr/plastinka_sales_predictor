"""
Глобальные фикстуры для тестов plastinka_sales_predictior

- Устанавливает переменные окружения для тестов (scope='session').
- Предоставляет session_monkeypatch для глобального патчинга.
- sample_predictions_data, create_training_params_fn — для генерации тестовых данных.
- Все фикстуры имеют scope='function' или 'session' только если это безопасно.
- Изоляция между тестами обеспечивается явным cleanup и использованием временных директорий/БД.
- Все моки сбрасываются между тестами.
"""

import pytest

from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)


# Common test data fixtures
@pytest.fixture(scope="session", autouse=True)
def set_test_environment_variables():
    """
    Set up global environment variables for testing, specifically for the encryption key.
    This fixture runs once per session and ensures the key is available to all tests.
    """
    pass

@pytest.fixture(scope="session")
def session_monkeypatch():
    """Session-scoped monkeypatch fixture.
    This fixture is moved here to be globally available within the tests directory.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()

@pytest.fixture
def sample_predictions_data():
    """Returns a sample of prediction data for use in various tests"""
    return {
        "barcode": [
            "123456789012",
            "123456789012",
            "987654321098",
            "987654321098",
            "555555555555",
        ],
        "artist": ["Artist A", "Artist A", "Artist B", "Artist B", "Artist C"],
        "album": ["Album X", "Album X", "Album Y", "Album Y", "Album Z"],
        "cover_type": ["Standard", "Standard", "Deluxe", "Deluxe", "Limited"],
        "price_category": ["A", "A", "B", "B", "C"],
        "release_type": ["Studio", "Studio", "Live", "Live", "Compilation"],
        "recording_decade": ["2010s", "2010s", "2000s", "2000s", "1990s"],
        "release_decade": ["2020s", "2020s", "2010s", "2010s", "2000s"],
        "style": ["Rock", "Rock", "Pop", "Pop", "Jazz"],
        "recording_year": [2015, 2015, 2007, 2007, 1995],
        "0.05": [10.5, 12.3, 5.2, 7.8, 3.1],
        "0.25": [15.2, 18.7, 8.9, 11.3, 5.7],
        "0.5": [21.4, 24.8, 12.6, 15.9, 7.5],
        "0.75": [28.3, 32.1, 17.8, 20.4, 10.2],
        "0.95": [35.7, 40.2, 23.1, 27.5, 15.8],
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
            batch_size=base_params.get("batch_size", 32),
            dropout=base_params.get("dropout", 0.2),
            use_reversible_instance_norm=True,
            use_layer_norm=True,
        )

        optimizer_config = OptimizerConfig(
            lr=base_params.get("learning_rate", 0.001), weight_decay=0.0001
        )

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

    return _create_training_params
