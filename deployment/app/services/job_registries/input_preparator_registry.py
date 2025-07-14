"""
Input preparators registry for DataSphere job inputs.

This module provides a unified approach to preparing different types of DataSphere job inputs
using the registry pattern.
"""
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Dict

from deployment.app.config import get_settings
from deployment.app.db.database import get_top_configs
from deployment.app.models.api_models import TrainingConfig
from deployment.app.services.job_registries.job_type_registry import JobTypeConfig

logger = logging.getLogger(__name__)


# --- Preparator Functions ---


async def prepare_training_inputs(
    job_id: str,
    config: TrainingConfig,
    target_dir: Path,
    job_config: JobTypeConfig,
):
    """Prepares inputs for standard training jobs."""
    logger.info(f"[{job_id}] Preparing inputs for training job...")
    if config:
        config = _replace_model_config_keys(config)
        config_json_path = target_dir / "config.json"
        logger.info(f"[{job_id}] Saving training config to {config_json_path}")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)


async def prepare_tuning_inputs(
    job_id: str,
    config: TrainingConfig,
    target_dir: Path,
    job_config: JobTypeConfig,
):
    """Prepares inputs for hyperparameter tuning jobs."""
    logger.info(f"[{job_id}] Preparing inputs for tuning job...")

    # Create tuning_settings.json
    tuning_json_path = target_dir / "tuning_settings.json"
    logger.info(f"[{job_id}] Creating tuning settings file: {tuning_json_path}")
    tuning_data = get_settings().tuning.model_dump()
    # Inject overrides from job additional parameters
    if "mode" in job_config.additional_params and job_config.additional_params["mode"] is not None:
        tuning_data["mode"] = job_config.additional_params["mode"]

    if "time_budget_s" in job_config.additional_params and job_config.additional_params["time_budget_s"] is not None:
        tuning_data["time_budget_s"] = job_config.additional_params["time_budget_s"]
    with open(tuning_json_path, "w", encoding="utf-8") as f:
        json.dump(tuning_data, f, indent=2)
    logger.info(f"[{job_id}] Tuning settings saved to {tuning_json_path}")

    if config:
        config = _replace_model_config_keys(config)
        config_path = target_dir / "config.json"
        logger.info(f"[{job_id}] Saving config to {config_path}")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)
        logger.info(f"[{job_id}] Config saved to {config_path}")

    # Create initial_configs.json with top historical configs
    init_cfgs: list[dict] = get_top_configs(
        limit=get_settings().tuning.seed_configs_limit,
        metric_name=get_settings().default_metric,
        higher_is_better=get_settings().default_metric_higher_is_better,
    )
    if not init_cfgs:
        logger.info(f"[{job_id}] No starter configs found for initial_configs.json")
    initial_cfgs_path = target_dir / "initial_configs.json"
    with open(initial_cfgs_path, "w", encoding="utf-8") as f:
        json.dump(init_cfgs, f, indent=2)
    logger.info(
        f"[{job_id}] initial_configs.json saved with {len(init_cfgs)} configs"
    )


# --- Registry and Helper Functions ---

# Define a type hint for preparator functions
PreparatorFunc = Callable[
    [str, TrainingConfig | None, Path, JobTypeConfig], Coroutine[Any, Any, None]
]


INPUT_PREPARATORS: Dict[str, PreparatorFunc] = {
    "training_input_preparator": prepare_training_inputs,
    "tuning_input_preparator": prepare_tuning_inputs,
}


def get_input_preparator(name: str) -> PreparatorFunc:
    """
    Get an input preparator function from the registry.

    Args:
        name: The name of the preparator.

    Returns:
        The preparator function.

    Raises:
        ValueError: If the preparator name is not found.
    """
    preparator = INPUT_PREPARATORS.get(name)
    if not preparator:
        available = ", ".join(INPUT_PREPARATORS.keys())
        raise ValueError(
            f"Unknown input preparator: {name}. Available: {available}"
        )
    return preparator


def _replace_model_config_keys(config: dict) -> dict:
    """Replace nn_model_config keys with model_config keys in the config."""
    if "nn_model_config" in config:
        config["model_config"] = config.pop("nn_model_config")
    return config