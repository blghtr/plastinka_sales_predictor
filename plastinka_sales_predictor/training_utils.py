import importlib
import uuid
import torch
from plastinka_sales_predictor import (
    PlastinkaTrainingTSDataset,
    setup_dataset,
    DEFAULT_METRICS,
    WQuantileRegression,
    configure_logger
)
from warnings import filterwarnings
from copy import deepcopy
from darts.models.forecasting.tide_model import TiDEModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    StochasticWeightAveraging,
    LearningRateMonitor
)
import logging
from typing_extensions import (
    TYPE_CHECKING,
    Optional,
    Dict,
    Tuple,
    Any,
    Union,
    List
)
filterwarnings('ignore')
if TYPE_CHECKING:
    from torchmetrics import Metric, MetricCollection
    from plastinka_sales_predictor.data_preparation import PlastinkaBaseTSDataset
    from darts.utils.likelihood_models import Likelihood
    from torch.optim.lr_scheduler import LRScheduler


logger = configure_logger(
    child_logger_name='train',
)


def prepare_for_training(
        config: Dict[str, Any],
        ds: 'PlastinkaBaseTSDataset', 
        val_ds: Optional['PlastinkaBaseTSDataset'] = None
) -> Tuple[
    'PlastinkaBaseTSDataset',
    Optional['PlastinkaBaseTSDataset'],
    List[pl.Callback],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    'Likelihood',
    Optional[str]
]:
    logger.info("Preparing for training")
    config = deepcopy(config)

    try:
        model_id = config.get('model_id', None)
        lags = config['lags']
        config['model_config']['input_chunk_length'] = lags

        model_config = config['model_config']
        optimizer_config = config['optimizer_config']
        ds_config = config['train_ds_config']
        swa_config = config['swa_config']

        lr_shed_config = config['lr_shed_config']
        _ = lr_shed_config.setdefault('interval', 'step')
        lr_scheduler_name = lr_shed_config.pop(
            'class_name', 'CosineAnnealingWarmRestarts'
        )
        lr_shed_module = importlib.import_module(
            'torch.optim.lr_scheduler'
        )
        lr_scheduler_cls = getattr(lr_shed_module, lr_scheduler_name)

        weights_config = config['weights_config']
        quantiles = config['quantiles']
        setup_dataset(
            ds=ds,
            input_chunk_length=lags,
            output_chunk_length=1,
            span=ds_config['span'],
            weights_alpha=ds_config['alpha']
        )

        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            StochasticWeightAveraging(
                **swa_config
            )
        ]
        
        if val_ds is not None:
            callbacks.extend([
                EarlyStopping(
                    monitor="val_loss",
                    patience=20,
                    min_delta=0.001,
                    mode="min",
                ),
            ])

        likelihood = WQuantileRegression(
            quantiles=quantiles,
            sigma_left_factor=weights_config['sigma_left'],
            sigma_right_factor=weights_config['sigma_right']
        )
    except KeyError:
        logger.error(
            "KeyError in prepare_for_training: "
            "required keys are missing in config",
            exc_info=True
        )
        raise

    except Exception:
        logger.error(
            "Error in prepare_for_training",
            exc_info=True
        )
        raise

    logging.info("Training is prepared")

    return (
        ds,
        val_ds,
        callbacks,
        lr_scheduler_cls,
        lr_shed_config,
        optimizer_config,
        model_config,
        likelihood,
        model_id
    )


def train_tide(
        ds: 'PlastinkaTrainingTSDataset',
        val_ds: Optional['PlastinkaTrainingTSDataset'] = None,
        callbacks: Optional[List[pl.Callback]] = None,
        lr_scheduler_cls: Optional['LRScheduler'] = None,
        lr_shed_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        likelihood: Optional['Likelihood'] = None,
        model_id: Optional[str] = None,
        random_state: Optional[int] = 42,
        torch_metrics: Optional[Union['Metric', 'MetricCollection']] = None,
) -> TiDEModel:
    if not model_id:
        model_id = str(uuid.uuid4())

    if torch_metrics is None:
        torch_metrics = DEFAULT_METRICS

    full_train = val_ds is None

    try:
        model = TiDEModel(
            output_chunk_length=1,
            optimizer_kwargs=optimizer_config,
            pl_trainer_kwargs={
                "callbacks": callbacks,
                "enable_progress_bar": False,
                "gradient_clip_val": 0.5,
                "precision": '32-true',
                "accelerator": get_device 
            },
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_shed_config,
            random_state=random_state,
            work_dir=f'logs/tide_{model_id}_fully_trained={full_train}',
            likelihood=likelihood,
            log_tensorboard=True,
            torch_metrics=torch_metrics,
            save_checkpoints=full_train,
            **model_config,
        )

        model = model.fit_from_dataset(ds, val_ds)
    except Exception:
        logger.error("Error training model.")
        raise
        
    return model


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

