import json
import os
from copy import deepcopy
from pathlib import Path
from warnings import filterwarnings

import pytorch_lightning as pl
import torch
from darts.models.forecasting.torch_forecasting_model import (
    CHECKPOINTS_FOLDER,
    INIT_MODEL_NAME,
)
from pytorch_lightning.callbacks import ModelCheckpoint

import ray
from plastinka_sales_predictor import (
    DEFAULT_METRICS,
    DartsCheckpointCallback,
    apply_config,
    configure_logger,
    get_model,
    split_dataset,
)

filterwarnings("ignore")


logger = configure_logger(child_logger_name="tune")


def load_fixed_params(json_path=None):
    """
    Load base config from a JSON file.
    Args:
        json_path: Path to the JSON file with fixed parameters

    Returns:
        dict: base_config with only fixed params based on tunable params
    """
    fixed = {}
    if json_path is not None:
        with open(json_path) as f:
            fixed = json.load(f)
    else:
        logger.warning("No JSON file provided, using empty config")
    return fixed


def merge_configs(update_config, base_config, keys=None):
    """
    Merge update_config with base_config.
    update_config takes precedence over base_config.
    """
    base_config = deepcopy(base_config)
    base_keys = set(base_config.keys())
    update_keys = set(update_config.keys())
    if keys is None:
        keys = base_keys | update_keys
        logger.debug("No keys provided, using all keys from base_config and update_config")

    if not isinstance(keys, set):
        try:
            keys = set(keys)
        except TypeError:
            raise TypeError(f"Keys must be hashable, got {type(keys)}")

    if len(keys & base_keys) == 0:
        logger.debug("No keys in base_config, returning full update_config")
        return update_config

    keys = list(keys & update_keys)
    if len(keys) == 0:
        logger.warning("Nothing to merge, returning base_config")
        return base_config

    for key in keys:
        if (
            isinstance(update_config[key], dict)
            and isinstance(base_config.get(key, {}), dict)
        ):
            base_config[key] = merge_configs(
                update_config[key],
                base_config.get(key, {})
            )
        else:
            base_config[key] = update_config[key]

    return base_config


def train_model(config, ds, val_ds, random_state=42, model_name="TiDE"):
    def load_from_checkpoint(ckpt_dir: str, best=True, **kwargs):
        def _get_checkpoint_fname():
            path = Path(ckpt_dir)

            prefixes = []
            if best:
                prefixes.append("best-*")
            prefixes.append("last-*")

            for i in range(len(prefixes)):
                prefix = prefixes[i]
                checklist = list(path.glob(prefix))
                if i == len(prefixes) - 1 and len(checklist) == 0:
                    raise FileNotFoundError(
                        f"There is no file matching prefix {prefix} in {ckpt_dir}"
                    )
                if len(checklist) > 0:
                    break

            file_name = max(checklist, key=os.path.getctime).name
            return file_name

        file_name = None
        trial_dir = Path(ckpt_dir).parent
        ckpt_trial_id = str(trial_dir.parts[-1])[6:]
        base_model_path = trial_dir / ckpt_trial_id / model_name / INIT_MODEL_NAME
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(
                "Could not find base model save file"
                f" `{INIT_MODEL_NAME}` in {base_model_path}."
            )
        print(f"! Loading base model from {base_model_path}")
        model = torch.load(base_model_path, weights_only=False)

        if file_name is None:
            file_name = _get_checkpoint_fname()

        file_path = Path(ckpt_dir, file_name)

        model.model = model._load_from_checkpoint(file_path, **kwargs)

        loss_fn = model.model_params.get("loss_fn")
        if loss_fn is not None:
            model.model.criterion = loss_fn
            model.model.train_criterion = deepcopy(loss_fn)
            model.model.val_criterion = deepcopy(loss_fn)

        torch_metrics = model.model.configure_torch_metrics(
            model.model_params.get("torch_metrics")
        )
        model.model.train_metrics = torch_metrics.clone(prefix="train_")
        model.model.val_metrics = torch_metrics.clone(prefix="val_")

        model._fit_called = True
        model.load_ckpt_path = file_path

        return model

    def inject_callback():
        callbacks_ = model.trainer_params["callbacks"]
        for i in range(len(callbacks_)):
            if isinstance(callbacks_[i], ModelCheckpoint):
                print("✓ Injected checkpoint callback")
                callbacks_[i] = ckpt_callback
                break
        assert isinstance(callbacks_[i], DartsCheckpointCallback), (
            f"Callback {callbacks_[i]} is not injected"
        )

    pl.seed_everything(random_state)

    context = ray.tune.get_context()
    trial_id = context.get_trial_id()

    ckpt_callback = DartsCheckpointCallback(
        metrics=["val_MIWS", "val_MIC", "val_MIWS_MIC_Ratio"],
        dirpath=str(Path(trial_id, model_name, CHECKPOINTS_FOLDER)),
        monitor="val_MIWS_MIC_Ratio",
        mode="min",
        save_last=True,
        save_top_k=-1,
        filename="best-{epoch}-{val_loss:.2f}-{val_MIWS_MIC_Ratio:.2f}",
    )
    ckpt_callback.CHECKPOINT_NAME_LAST = (
        "last-{epoch}-{val_loss:.2f}-{val_MIWS_MIC_Ratio:.2f}"
    )

    (
        ds,
        val_ds,
        callbacks,
        lr_scheduler_cls,
        lr_shed_config,
        optimizer_config,
        model_config,
        likelihood,
        _,
    ) = apply_config(config=config, ds=ds, val_ds=val_ds)
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            if os.path.exists(checkpoint_dir):
                model = load_from_checkpoint(checkpoint_dir)
                inject_callback()
                print(f"✓ Resumed training from epoch {model.epochs_trained}")

    else:
        model_config["nr_epochs_val_period"] = 5
        model = get_model(
            optimizer_config=optimizer_config,
            callbacks=callbacks,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_shed_config=lr_shed_config,
            random_state=random_state,
            work_dir=trial_id,
            model_name=model_name,
            save_checkpoints=True,
            likelihood=likelihood,
            torch_metrics=DEFAULT_METRICS,
            model_config=model_config,
        )
        inject_callback()

    model.fit_from_dataset(ds, val_ds)
    return model


def train_fn(config, fixed_config, ds):
    """Ray Tune trainable wrapper.

    If *val_ds* provided, use it directly; otherwise split *ds* into
    train/val windows (legacy behaviour).
    """

    config = merge_configs(config, fixed_config)
    train_ds, val_ds = split_dataset(ds, config["lags"])
    if train_ds is not None and val_ds is not None:
        train_model(config, train_ds, val_ds)
    else:
        raise ValueError("train_ds or val_ds is None")


def trial_name_creator(trial):
    return "trial_" + str(trial.trial_id)


def flatten_config(config, parent_key=None):
    """
    Flatten a nested configuration dictionary.
    foo: bar: baz -> config/foo/bar/baz

    Args:
        config: Nested configuration dictionary
        parent_key: Parent key to prepend to the flattened keys

    Returns:
        Flattened configuration dictionary
    """
    flattened = {}
    for key, value in config.items():
        keys = []
        if parent_key:
            if isinstance(parent_key, list):
                keys.extend(parent_key)
            else:
                keys.append(parent_key)
        keys.append(key)
        if isinstance(value, dict):
            flattened.update(flatten_config(value, keys))
        else:
            flattened["/".join(keys)] = value
    return flattened
