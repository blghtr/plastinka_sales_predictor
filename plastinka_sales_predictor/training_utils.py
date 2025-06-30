import importlib
import logging
import uuid
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
from warnings import filterwarnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from typing_extensions import Any

from plastinka_sales_predictor import (
    DEFAULT_METRICS,
    PlastinkaTrainingTSDataset,
    WQuantileRegression,
    configure_logger,
)
from plastinka_sales_predictor.model import CustomTiDEModel as TiDEModel

filterwarnings("ignore")
if TYPE_CHECKING:
    from darts.utils.likelihood_models import Likelihood
    from torch.optim.lr_scheduler import LRScheduler
    from torchmetrics import Metric, MetricCollection

    from plastinka_sales_predictor.data_preparation import PlastinkaBaseTSDataset


logger = configure_logger(
    child_logger_name="train",
)

THIS_DIR = Path(__file__).parent


def prepare_for_training(
    config: dict[str, Any],
    ds: "PlastinkaBaseTSDataset",
    val_ds: Optional["PlastinkaBaseTSDataset"] = None,
    callbacks: list[pl.Callback] | None = None,
) -> tuple[
    "PlastinkaBaseTSDataset",
    Optional["PlastinkaBaseTSDataset"],
    list[pl.Callback],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    "Likelihood",
    str | None,
]:
    logger.info("Preparing for training")
    config = deepcopy(config)

    try:
        lags = config["lags"]
        config["model_config"]["input_chunk_length"] = lags

        model_id = config.setdefault("model_id", str(uuid.uuid4()))
        model_config = config["model_config"]
        optimizer_config = config["optimizer_config"]
        ds_config = config["train_ds_config"]
        swa_config = config["swa_config"]

        lr_shed_config = config["lr_shed_config"]
        _ = lr_shed_config.setdefault("interval", "step")
        lr_scheduler_name = lr_shed_config.pop(
            "class_name", "CosineAnnealingWarmRestarts"
        )
        lr_shed_module = importlib.import_module("torch.optim.lr_scheduler")
        lr_scheduler_cls = getattr(lr_shed_module, lr_scheduler_name)

        weights_config = config["weights_config"]
        quantiles = config.pop("quantiles", None)
        ds.setup_dataset(
            input_chunk_length=lags,
            output_chunk_length=1,
            span=ds_config["span"],
            weights_alpha=ds_config["alpha"],
            copy=False,
        )

        if callbacks is None:
            callbacks = []

        callbacks.extend(
            [
                LearningRateMonitor(logging_interval="step"),
                StochasticWeightAveraging(**swa_config),
            ]
        )

        if val_ds is not None:
            val_ds.setup_dataset(
                input_chunk_length=lags,
                output_chunk_length=1,
                span=ds_config["span"],
                weights_alpha=ds_config["alpha"],
                copy=False,
            )

            callbacks.extend(
                [
                    EarlyStopping(
                        monitor="val_loss",
                        patience=15,
                        min_delta=0.001,
                        mode="min",
                    ),
                ]
            )

        likelihood = WQuantileRegression(
            quantiles=quantiles,
            sigma_left_factor=weights_config["sigma_left"],
            sigma_right_factor=weights_config["sigma_right"],
        )

    except KeyError:
        logger.error(
            "KeyError in prepare_for_training: required keys are missing in config",
            exc_info=True,
        )
        raise

    except Exception:
        logger.error("Error in prepare_for_training", exc_info=True)
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
        model_id,
    )


def get_model(
    optimizer_config: dict[str, Any],
    callbacks: list[pl.Callback],
    lr_scheduler_cls: "LRScheduler",
    lr_shed_config: dict[str, Any],
    random_state: int,
    work_dir: str,
    model_name: str,
    save_checkpoints: bool,
    likelihood: "Likelihood",
    torch_metrics: Union["Metric", "MetricCollection"],
    model_config: dict[str, Any],
):
    return TiDEModel(
        output_chunk_length=1,
        optimizer_kwargs=optimizer_config,
        pl_trainer_kwargs={
            "callbacks": callbacks,
            "enable_progress_bar": False,
            "gradient_clip_val": 0.5,
            "precision": "32-true",
            "accelerator": get_device(),
        },
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_shed_config,
        random_state=random_state,
        work_dir=work_dir,
        model_name=model_name,
        likelihood=likelihood,
        log_tensorboard=True,
        torch_metrics=torch_metrics,
        save_checkpoints=save_checkpoints,
        force_reset=True,
        **model_config,
    )


def train_model(
    ds: "PlastinkaTrainingTSDataset",
    val_ds: Optional["PlastinkaTrainingTSDataset"] = None,
    callbacks: list[pl.Callback] | None = None,
    lr_scheduler_cls: Optional["LRScheduler"] = None,
    lr_shed_config: dict[str, Any] | None = None,
    optimizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    likelihood: Optional["Likelihood"] = None,
    model_id: str | None = None,
    model_name: str | None = "TiDE",
    random_state: int | None = 42,
    torch_metrics: Union["Metric", "MetricCollection"] | None = None,
) -> TiDEModel:
    if torch_metrics is None:
        torch_metrics = DEFAULT_METRICS

    save_checkpoints = val_ds is None
    work_dir = THIS_DIR.parent / f"logs/{model_name}_{model_id}"
    try:
        model = get_model(
            optimizer_config=optimizer_config,
            callbacks=callbacks,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_shed_config=lr_shed_config,
            random_state=random_state,
            work_dir=work_dir,
            model_name=model_name,
            save_checkpoints=save_checkpoints,
            likelihood=likelihood,
            torch_metrics=torch_metrics,
            model_config=model_config,
        )

        # Принудительно выставляем флаг использования статических ковариат
        # Это необходимо, поскольку при использовании fit_from_dataset() вместо fit()
        # автоматическая проверка наличия static_covariates в _setup_for_fit_from_dataset() не выполняется
        if model.supports_static_covariates and model.considers_static_covariates:
            model._uses_static_covariates = True

        logger.info("Принудительно установлен флаг _uses_static_covariates = True")

        model = model.fit_from_dataset(ds, val_ds)

    except Exception:
        logger.error("Error training model.")
        raise

    return model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def extract_early_stopping_callback(trainer: pl.Trainer) -> EarlyStopping:
    for callback in trainer.callbacks:
        if isinstance(callback, EarlyStopping):
            return callback
    return None
