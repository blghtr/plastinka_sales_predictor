"""
Result processors registry for DataSphere job results.

This module provides a unified approach to processing different types of DataSphere job results
using the registry pattern to eliminate conditional checks and improve extensibility.
"""

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict

from deployment.app.config import get_settings
from deployment.app.db.database import (
    create_or_get_config,
    create_tuning_result,
    execute_query,
    update_job_status,
)
from deployment.app.models.api_models import JobStatus, TrainingConfig

# Note: Avoiding circular import by importing these functions inside the functions where needed

logger = logging.getLogger(__name__)


async def process_training_results(
    job_id: str,
    ds_job_id: str,
    results_dir: str,
    config: TrainingConfig,
    metrics_data: Dict[str, Any] | None,
    output_files: Dict[str, str | None],
    polls: int,
    poll_interval: float,
    config_id: str,
    **kwargs,
) -> None:
    """Process training job results - handles model and predictions."""

    model_path = output_files.get("model.onnx")
    predictions_path = output_files.get("predictions.csv")

    final_message = f"Job completed. DS Job ID: {ds_job_id}."
    warnings_list = []
    model_id_for_status = None

    try:
        # Import functions to avoid circular import
        from deployment.app.services.datasphere_service import (
            _perform_model_cleanup,
            save_model_file_and_db,
            save_predictions_to_db,
        )
        from deployment.app.db.database import create_training_result

        # Save model and predictions if they exist
        if model_path:
            model_id_for_status = await save_model_file_and_db(
                job_id=job_id,
                model_path=model_path,
                ds_job_id=ds_job_id,
                config=config,
                metrics_data=metrics_data,
            )
            logger.info(f"[{job_id}] Model saved with ID: {model_id_for_status}")

            if predictions_path:
                prediction_result_info = save_predictions_to_db(
                    predictions_path=predictions_path,
                    job_id=job_id,
                    model_id=model_id_for_status,
                )
                logger.info(
                    f"[{job_id}] Saved {prediction_result_info.get('predictions_count', 'N/A')} predictions to database with result_id: {prediction_result_info.get('result_id', 'N/A')}"
                )
            else:
                logger.warning(
                    f"[{job_id}] No predictions file found, cannot save predictions to DB."
                )
                warnings_list.append("Predictions file not found in results.")
        else:
            logger.warning(
                f"[{job_id}] Model path is None or empty. Skipping model and prediction saving."
            )
            warnings_list.append("Model file not found in results.")

        # Create a record of the training result with metrics
        training_result_id = None
        if metrics_data:
            training_result_id = create_training_result(
                job_id=job_id,
                config_id=config_id,
                metrics=metrics_data,
                model_id=model_id_for_status,
                duration=int(polls * poll_interval),
                config=config.model_dump(),
            )
            logger.info(
                f"[{job_id}] Training result record created: {training_result_id}"
            )
            final_message += f" Training Result ID: {training_result_id}."
        else:
            logger.warning(
                f"[{job_id}] No metrics data available. Training result record not created."
            )
            warnings_list.append("Metrics data not found in results.")

        # Append warnings to the status message if any
        if warnings_list:
            final_message += " Processing warnings: " + "; ".join(warnings_list)

        # Update job status to completed
        update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            progress=100,
            status_message=final_message,
            result_id=training_result_id,
        )
        logger.info(
            f"[{job_id}] Job processing completed. Final status message: {final_message}"
        )

        # Perform model cleanup only if a model was successfully created in this run
        if model_id_for_status:
            await _perform_model_cleanup(job_id, model_id_for_status)
        else:
            logger.info(
                f"[{job_id}] Skipping model cleanup as no new model was created in this job run."
            )

    except Exception as e:
        error_msg = f"Error processing training job results: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        raise


def process_tuning_results(
    job_id: str,
    results_dir: str,
    metrics_data: Dict[str, Any] | None,
    output_files: Dict[str, str | None],
    polls: int,
    poll_interval: float,
    **kwargs,
) -> None:
    """Process tuning job results - handles best configs and metrics."""

    best_configs_path = output_files.get("best_configs.json")

    if not best_configs_path or not os.path.exists(best_configs_path):
        logger.error(f"[{job_id}] best_configs.json not found in results")
        update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message="Tuning results missing best_configs.json",
        )
        raise RuntimeError("best_configs.json not found")

    # Load configs
    try:
        with open(best_configs_path, "r", encoding="utf-8") as f:
            best_cfgs: list[dict[str, Any]] = json.load(f)
    except Exception as e:
        logger.error(f"[{job_id}] Failed to parse best_configs.json: {e}")
        raise

    # Validate metrics format – must be list[dict] with same length as configs
    if not isinstance(metrics_data, list):
        error_msg = "metrics.json must contain a list of metric dicts (one per config)"
        logger.error(f"[{job_id}] {error_msg}. Got type: {type(metrics_data).__name__}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg)

    if len(metrics_data) != len(best_cfgs):
        error_msg = (
            f"metrics.json list length ({len(metrics_data)}) does not match number of configs ({len(best_cfgs)})"
        )
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg)

    metrics_list: list[dict[str, Any]] = metrics_data

    # Persist configs and tuning_results
    saved_count = 0
    for idx, cfg in enumerate(best_cfgs):
        try:
            # Store config, mark source
            cfg_copy = cfg.copy()
            cfg_id = create_or_get_config(cfg_copy, is_active=False, source="tuning")

            metric_dict = metrics_list[idx] if idx < len(metrics_list) else {}

            create_tuning_result(
                job_id=job_id,
                config_id=cfg_id,
                metrics=metric_dict,
                duration=int(polls * poll_interval),
            )
            saved_count += 1
        except Exception as cfg_e:
            logger.warning(f"[{job_id}] Failed to persist tuning config #{idx}: {cfg_e}")

    update_job_status(
        job_id,
        JobStatus.COMPLETED.value,
        progress=100,
        status_message=f"Tuning completed. Saved {saved_count} configs.",
    )


# Registry of result processors
RESULT_PROCESSORS: Dict[str, Callable] = {
    "process_training_results": process_training_results,
    "process_tuning_results": process_tuning_results,
}


async def process_job_results_unified(job_id: str, processor_name: str, **kwargs) -> None:
    """
    Unified result processing using registry pattern.

    Args:
        job_id: Job identifier
        processor_name: Name of the processor function to use
        **kwargs: Arguments to pass to the processor function

    Raises:
        ValueError: If processor_name is not found in registry
    """
    processor_func = RESULT_PROCESSORS.get(processor_name)
    if not processor_func:
        available_processors = ", ".join(RESULT_PROCESSORS.keys())
        raise ValueError(
            f"Unknown result processor: {processor_name}. "
            f"Available processors: {available_processors}"
        )

    logger.info(f"[{job_id}] Processing results using: {processor_name}")

    # Call the processor function (async or sync)
    if asyncio.iscoroutinefunction(processor_func):
        await processor_func(job_id=job_id, **kwargs)
    else:
        processor_func(job_id=job_id, **kwargs)


def register_result_processor(name: str, processor_func: Callable) -> None:
    """
    Register a new result processor.

    Args:
        name: Name of the processor
        processor_func: Function to handle results processing
    """
    RESULT_PROCESSORS[name] = processor_func
    logger.info(f"Registered result processor: {name}")


def get_available_processors() -> list[str]:
    """Get list of available result processors."""
    return list(RESULT_PROCESSORS.keys()) 