from copy import deepcopy
from warnings import filterwarnings
from ray import tune
from ray.train import RunConfig, CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.tuner import Tuner
from pathlib import Path
import json
from typing import Any
import ray
from plastinka_sales_predictor import (
    PlastinkaTrainingTSDataset,
    train_fn,
    trial_name_creator,
    configure_logger,
    flatten_config,
    load_fixed_params
)
import os
import sys
import tempfile
import time
import zipfile
import click
import numpy as np
import pandas as pd

filterwarnings('ignore')


logger = configure_logger()

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- File constants expected inside DataSphere job input dir ---
TUNING_SETTINGS_FILE = "tuning_settings.json"
INITIAL_CONFIGS_FILE = "initial_configs.json"


model_config = {
    "num_encoder_layers": tune.choice([1, 2, 3, 4, 5]),
    "num_decoder_layers": tune.choice([1, 2, 3, 4, 5]),
    "decoder_output_dim": tune.choice([16, 32, 64, 128, 256]),
    "temporal_width_past": tune.choice([0, 1, 2, 3, 4, 5]),
    "temporal_width_future": tune.choice([0, 1, 2, 3, 4, 5]),
    "temporal_hidden_size_past": tune.choice([4, 8, 16, 32]),
    "temporal_hidden_size_future": tune.choice([4, 8, 16, 32]),
    "temporal_decoder_hidden": tune.choice([16, 32, 64, 128, 256]),
    "batch_size": tune.choice([8, 16, 32, 64, 128, 256]),
    "dropout": tune.uniform(0.3, 0.8),
    "use_reversible_instance_norm": tune.choice([False, True]),
    "use_layer_norm": tune.choice([False, True]),
    }

optimizer_config = {
    "lr": tune.loguniform(1e-10, 1e-2),
    "weight_decay": tune.loguniform(1e-10, 1e-1),
    }

lr_shed_config = {
        "T_0": tune.qrandint(10, 500, 10),
        "T_mult": tune.randint(1, 5),
            }

train_ds_config = {
    "alpha": tune.uniform(1, 7),
    "span": tune.randint(3, 16),
    }

swa_config = {
    "swa_lrs": tune.loguniform(1e-5, 1e-2),
    "swa_epoch_start": tune.uniform(0.0, 0.8),
    "annealing_epochs": tune.randint(5, 50),
    }

weights_config = {
    "sigma_left": tune.uniform(0.4, 1.0),
    "sigma_right": tune.uniform(0.8, 3.0),
    }

tunable_params = {
    "model_config": model_config,
    "optimizer_config": optimizer_config,
    "lr_shed_config": lr_shed_config,
    "train_ds_config": train_ds_config,
    "swa_config": swa_config,
    "weights_config": weights_config,
    "lags": tune.randint(3, 7),
    }

_fixed_params = {
    "model_config": {
        "n_epochs": 100  # hardcoded for now
    },
    "quantiles": QUANTILES,
}

def validate_input_directory(input_path: str) -> str:
    """If input_path is zip – extract, else ensure dir exists."""
    if zipfile.is_zipfile(input_path):
        temp_dir = tempfile.mkdtemp(prefix="plastinka_tune_input_")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_dir)
        return temp_dir
    elif os.path.isdir(input_path):
        return input_path
    else:
        logger.error(f"Invalid input path: {input_path}")
        sys.exit(1)


def create_output_archive(temp_output_dir: str, output: str, files: list[str]):
    """Zip selected files from temp_output_dir to output path."""
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fp in files:
            rel_name = os.path.basename(fp)
            zipf.write(fp, rel_name)
    logger.info(f"Output archive created: {output}")


@click.command()
@click.option(
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to input directory (or zip) containing datasets and config.json",
)
@click.option("--output", default="output.zip", help="Name of output archive")
def main(input_dir, output):
    """Run hyperparameter tuning inside DataSphere job."""

    logger.info("Starting tuning job ...")
    input_dir_real = validate_input_directory(input_dir)

    # Read tuning settings from JSON file
    tuning_settings_path = os.path.join(input_dir_real, "tuning_settings.json")
    tuning_cfg = {}
    if os.path.exists(tuning_settings_path):
        try:
            with open(tuning_settings_path, "r", encoding="utf-8") as f:
                tuning_cfg = json.load(f)
            logger.info(f"Loaded tuning settings from {tuning_settings_path}")
        except Exception as e:
            logger.warning(f"Failed to load tuning settings: {e}, using defaults")
    else:
        logger.warning("tuning_settings.json not found, using default settings")

    # Determine tuning mode from JSON (default 'light')
    mode = tuning_cfg.get("mode", "light")

    # Extract configuration values with fallbacks
    num_samples = tuning_cfg.get(
        "num_samples_light" if mode == "light" else "num_samples_full",
        50 if mode == "light" else 200,
    )
    resources = tuning_cfg.get("resources", {"cpu": 32})
    max_concurrent = tuning_cfg.get("max_concurrent", 16)
    best_configs_to_save = tuning_cfg.get("best_configs_to_save", 5)
    
    # Optional overall time budget for the entire tuning run
    time_budget_s = tuning_cfg.get("time_budget_s", 3600)

    logger.info(
        f"Using tuning configuration: mode={mode}, num_samples={num_samples}, "
        f"resources={resources}, max_concurrent={max_concurrent}"
    )

    # Load fixed parameters from active_config if present
    fixed_params_path = os.path.join(input_dir_real, "config.json")
    fixed_params = load_fixed_params(
        fixed_params_path, tunable_params
    )
    fixed_params.update(_fixed_params)
    # Load initial configs if present
    initial_configs: list[dict] = []
    init_cfg_path = os.path.join(input_dir_real, INITIAL_CONFIGS_FILE)
    if os.path.exists(init_cfg_path):
        try:
            with open(init_cfg_path, "r", encoding="utf-8") as f:
                raw_cfgs = json.load(f)
            for raw_cfg in raw_cfgs:
                cfg = deepcopy(raw_cfg)

                if "nn_model_config" in cfg:
                    cfg["model_config"] = cfg.pop("nn_model_config")
                keys = list(cfg.keys())

                for k in keys:
                    if k not in tunable_params:
                        _ = cfg.pop(k)
                if mode == "light":
                    _ = cfg.pop("model_config")
                else:
                    _ = cfg["model_config"].pop("n_epochs")

                try:
                    initial_configs.append(flatten_config(cfg))
                except Exception as fe:
                    logger.warning(f"Failed to flatten initial config: {fe}")
            
            logger.info(f"Loaded {len(initial_configs)} initial configs from {init_cfg_path}")
        except Exception as e:
            logger.warning(f"Failed to read initial_configs.json: {e}")

    if mode == "light":
        _ = tunable_params.pop("model_config")

    # dataset paths
    train_path = os.path.join(input_dir_real, "train.dill")
    val_path = os.path.join(input_dir_real, "val.dill")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error("train.dill or val.dill not found in input dir")
        sys.exit(1)

    ds = PlastinkaTrainingTSDataset.from_dill(train_path)
    val_ds = PlastinkaTrainingTSDataset.from_dill(val_path)

    # re-create train_with_parameters with new ds
    train_with_parameters = tune.with_parameters(
        train_fn, fixed_config=fixed_params, ds=ds, val_ds=val_ds
    )
    train_with_parameters = tune.with_resources(
        train_with_parameters,
        {
            k: max(1., float(v) / max_concurrent) # TODO: check if this is correct
            for k, v in resources.items()
        }
    )

    scheduler = HyperBandForBOHB(max_t=100, metric="val_MIWS_MIC_Ratio", mode="min")
    # Configure BOHB searcher with optional starter configs
    searcher = TuneBOHB(
        metric="val_MIWS_MIC_Ratio",
        mode="min",
        max_concurrent=max_concurrent,
        points_to_evaluate=initial_configs if initial_configs else None,
        seed=42,
    )

    # run tuner
    start = time.time()
    tuner = Tuner(
        train_with_parameters,
        param_space=tunable_params,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            search_alg=searcher,
            scheduler=scheduler,
            time_budget_s=time_budget_s,
            trial_dirname_creator=trial_name_creator
        ),
        run_config=RunConfig(
                name="plastinka_tuning",
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute="val_MIWS_MIC_Ratio",
                    checkpoint_score_order="min"
                ),
                sync_config=ray.train.SyncConfig(sync_artifacts=True)),
    )
    results = tuner.fit()
    _duration = int(time.time() - start)

    df = results.get_dataframe()
    if "val_MIWS_MIC_Ratio" not in df.columns:
        logger.error("Metric val_MIWS_MIC_Ratio not present in results dataframe")
        sys.exit(1)

    df_sorted = df.sort_values("val_MIWS_MIC_Ratio").head(best_configs_to_save)

    # Collect validation metric columns only
    metric_cols = [c for c in df.columns if c.startswith("val_")]

    best_configs: list[dict] = []
    metrics_list: list[dict] = []

    # Build nested configs and associated metric dicts (maintain same ordering)
    for _, row in df_sorted.iterrows():
        flat_cfg = {c: row[c] for c in df_sorted.columns if c.startswith("config/")}
        nested_cfg = unflatten_config(flat_cfg)
        best_configs.append(nested_cfg)

        metric_values = {col: _convert_jsonable(row[col]) for col in metric_cols}
        metrics_list.append(metric_values)

    # save json files
    with tempfile.TemporaryDirectory(prefix="plastinka_tune_outputs_") as temp_out:
        best_cfg_path = os.path.join(temp_out, "best_configs.json")
        metrics_path = os.path.join(temp_out, "metrics.json")
        with open(best_cfg_path, "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_list, f, indent=2)

        output_abs = os.path.abspath(output)
        create_output_archive(temp_out, output_abs, [best_cfg_path, metrics_path])

    logger.info("Tuning job finished in %s seconds", _duration)


# ---------------------------------------------------------------------------
# Helper utilities (local to this script)
# ---------------------------------------------------------------------------

def _convert_jsonable(value):
    """Convert numpy dtypes to native python for JSON serialization."""
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def unflatten_config(flat_cfg: dict[str, Any], prefix: str = "config/") -> dict:
    """Recreate nested config structure from flattened keys.

    Example:
        {
            "config/model_config/num_layers": 3,
            "config/optimizer_config/lr": 0.001,
        }
        -> {"model_config": {"num_layers": 3}, "optimizer_config": {"lr": 0.001}}

    Args:
        flat_cfg: Dictionary with flattened keys (slash-separated).
        prefix:   Leading prefix to strip (default "config/").

    Returns:
        Nested configuration dictionary.
    """
    nested: dict = {}
    for full_key, value in flat_cfg.items():
        if not full_key.startswith(prefix):
            continue

        # Skip NaNs (can appear for missing params)
        if pd.isna(value):
            continue

        key_path = full_key[len(prefix):].split("/")  # strip prefix and split
        current = nested
        for part in key_path[:-1]:
            current = current.setdefault(part, {})
        current[key_path[-1]] = _convert_jsonable(value)

    return nested


if __name__ == "__main__":
    main()