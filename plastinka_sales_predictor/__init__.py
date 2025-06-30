"""
Plastinka Sales Predictor - A machine learning model for sales prediction.
"""

__version__ = "0.1.0"

from .callbacks import DartsCheckpointCallback
from .data_preparation import GlobalLogMinMaxScaler, PlastinkaTrainingTSDataset
from .logger_setup import configure_logger
from .losses import WQuantileRegression
from .metrics import DEFAULT_METRICS, MIC, MIWS, MIWS_MIC_Ratio
from .training_utils import (
    extract_early_stopping_callback,
    get_model,
    prepare_for_training,
    train_model,
)
from .tuning_utils import (
    flatten_config,
    load_fixed_params,
    train_fn,
    trial_name_creator,
)
from .tuning_utils import train_model as train_model_tune

__all__ = [
    "MIWS",
    "MIC",
    "MIWS_MIC_Ratio",
    "DEFAULT_METRICS",
    "WQuantileRegression",
    "DartsCheckpointCallback",
    "PlastinkaTrainingTSDataset",
    "GlobalLogMinMaxScaler",
    "configure_logger",
    "train_model",
    "prepare_for_training",
    "get_model",
    "train_model_tune",
    "train_fn",
    "trial_name_creator",
    "flatten_config",
    "load_fixed_params",
    "extract_early_stopping_callback",
]
