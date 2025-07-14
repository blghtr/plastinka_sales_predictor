import json
import os
from collections import defaultdict
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
    configure_logger,
    get_model,
    prepare_for_training,
    PlastinkaBaseTSDataset,
    PlastinkaTrainingTSDataset,
)

filterwarnings("ignore")


logger = configure_logger(child_logger_name="tune")


def load_fixed_params(json_path=None, tunable_params=None):
    """
    Load fixed parameters from a JSON file and return tunable and fixed parameters.

    Args:
        json_path: Path to the JSON file with fixed parameters

    Returns:
        tuple: (tunable_params, fixed_params)
    """

    assert json_path or tunable_params, (
        "Either json_path or tunable_params must be provided."
    )
    if tunable_params is None:
        tunable_params = {}

    fixed = {}
    if json_path is not None:
        with open(json_path) as f:
            fixed = json.load(f)

    fixed_params = defaultdict(dict)
    for key in [ 
        "optimizer_config",
        "lr_shed_config",
        "train_ds_config",
        "swa_config",
        "weights_config",
        "lags",
    ]:
        if key in fixed:
            fixed_part = None
            if key in tunable_params:
                if isinstance(fixed[key], dict):
                    fixed_part = {
                        k: v
                        for k, v in fixed[key].items()
                        if k not in tunable_params[key]
                    }
                if fixed_part:
                    fixed_params[key] = fixed_part
            else:
                fixed_params[key] = fixed[key]

    return fixed_params


def merge_with_fixed_params(tunable_params, fixed_params):
    """
    Merge tunable configuration with fixed parameters.
    Tunable parameters take precedence over fixed ones.
    """
    merged = fixed_params
    for key, value in tunable_params.items():
        if key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


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
    ) = prepare_for_training(config=config, ds=ds, val_ds=val_ds)
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            if os.path.exists(checkpoint_dir):
                model = load_from_checkpoint(checkpoint_dir)
                inject_callback()
                print(f"✓ Resumed training from epoch {model.epochs_trained}")

    else:
        model_config["nr_epochs_val_period"] = 1
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


def train_fn(config, fixed_config, ds, val_ds=None):
    """Ray Tune trainable wrapper.

    If *val_ds* provided, use it directly; otherwise split *ds* into
    train/val windows (legacy behaviour).
    """

    config = merge_with_fixed_params(config, fixed_config)
    
    if val_ds is None:
        L = ds.L
        lags = config["lags"]
        length = lags + 1

        temp_train = ds.setup_dataset(
            input_chunk_length=lags, output_chunk_length=1, window=(0, L - 1), copy=True
        )
        temp_val = ds.setup_dataset(
            input_chunk_length=lags, output_chunk_length=1, window=(L - length, L), copy=True
        )

        assert temp_train.L < ds.L and temp_val.end < ds._n_time_steps
        train_model(config, temp_train, temp_val)
    else:
        # Use provided datasets directly
        train_model(config, ds, val_ds)


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
