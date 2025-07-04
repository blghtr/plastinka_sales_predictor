"""
Plastinka Sales Predictor - A machine learning model for sales prediction.
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNF

try:
    __version__ = _pkg_version(__name__)
except _PkgNF:
    __version__ = "0.0.0+local"  # fallback для editable-install или запуска из исходников

# Always available imports (no ML dependencies)
from .data_preparation import GlobalLogMinMaxScaler, PlastinkaTrainingTSDataset
from .logger_setup import configure_logger

# Conditional imports for ML components (only if dependencies are available)
try:
    from .callbacks import DartsCheckpointCallback
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
    
    # ML components are available
    _ML_AVAILABLE = True
except ImportError as e:
    # ML components not available (e.g., in deployment environment)
    import warnings
    warnings.warn(
        f"ML components not available: {e}. "
        "Only data preparation functions will be available. "
        "Install with 'uv sync --extra ml' for full ML functionality.",
        ImportWarning
    )
    
    # Create placeholder objects to avoid AttributeError
    DartsCheckpointCallback = None
    WQuantileRegression = None
    DEFAULT_METRICS = None
    MIC = None
    MIWS = None
    MIWS_MIC_Ratio = None
    extract_early_stopping_callback = None
    get_model = None
    prepare_for_training = None
    train_model = None
    flatten_config = None
    load_fixed_params = None
    train_fn = None
    trial_name_creator = None
    train_model_tune = None
    
    _ML_AVAILABLE = False

# Base exports (always available)
_BASE_ALL = [
    "PlastinkaTrainingTSDataset",
    "GlobalLogMinMaxScaler", 
    "configure_logger",
]

# ML exports (only if ML components are available)
_ML_ALL = [
    "MIWS",
    "MIC", 
    "MIWS_MIC_Ratio",
    "DEFAULT_METRICS",
    "WQuantileRegression",
    "DartsCheckpointCallback",
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

# Dynamic __all__ based on available components
__all__ = _BASE_ALL + (_ML_ALL if _ML_AVAILABLE else [])
