"""
Job Type Configuration Registry for DataSphere Tasks.

This module provides a unified configuration approach for different types of DataSphere jobs
(training, tuning, etc.) to eliminate conditional checks and improve code maintainability.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class JobTypeConfig:
    """Configuration for a DataSphere job type."""

    name: str
    """Name of the job type (e.g., 'train', 'tune')."""

    script_dir_getter: Callable[[Any], str]
    """Function to get the script directory from settings."""

    config_filename: str
    """Name of the config file (usually 'config.yaml')."""

    required_input_files: List[str]
    """List of required input files that must be present."""

    optional_input_files: List[str]
    """List of optional input files that may be present."""

    expected_output_files: List[str]
    """List of expected output files from the job."""

    result_processor_name: str
    """Name of the function to process job results."""

    input_preparator_name: str
    """Name of the function to prepare job inputs."""

    output_file_roles: Dict[str, str] = field(default_factory=dict)
    """Roles of output files, e.g. {'model.onnx': 'model', 'predictions.csv': 'predictions'}"""

    async_processor: bool = False
    """Whether the result processor is asynchronous."""

    additional_params: Dict[str, Any] = field(default_factory=dict)
    """Dynamic job-specific parameters (e.g., {"mode": "light"}).
    Filled/updated перед запуском run_job, не хранится в реестре глобально."""

    datasets_to_generate: List[str] = field(default_factory=list)
    """List of dataset types to generate by prepare_datasets (e.g., ["train", "val", "inference"])."""

    def get_script_dir(self, settings) -> str:
        """Get the script directory for this job type."""
        return self.script_dir_getter(settings)


# Registry of job type configurations
JOB_TYPE_CONFIGS = {
    "train": JobTypeConfig(
        name="train",
        script_dir_getter=lambda s: s.datasphere_job_train_dir,
        config_filename="config.yaml",
        required_input_files=["train.dill", "val.dill", "config.json", "inference.dill"],
        optional_input_files=[],
        expected_output_files=["model.onnx", "predictions.csv", "metrics.json"],
        result_processor_name="process_training_results",
        input_preparator_name="training_input_preparator",
        output_file_roles={"model.onnx": "model", "predictions.csv": "predictions", "metrics.json": "metrics"},
        async_processor=True,
        datasets_to_generate=["train", "val", "inference"],
    ),
    "tune": JobTypeConfig(
        name="tune",
        script_dir_getter=lambda s: s.datasphere_job_tune_dir,
        config_filename="config.yaml",
        required_input_files=["train.dill", "val.dill"],
        optional_input_files=["tuning_settings.json", "initial_configs.json"],
        expected_output_files=["best_configs.json", "metrics.json"],
        result_processor_name="process_tuning_results",
        input_preparator_name="tuning_input_preparator",
        output_file_roles={"best_configs.json": "configs", "metrics.json": "metrics"},
        async_processor=False,
        datasets_to_generate=["train", "val"],
    ),
}


def get_job_type_config(job_type: str) -> JobTypeConfig:
    """
    Get job type configuration by name.

    Args:
        job_type: Name of the job type

    Returns:
        JobTypeConfig for the specified job type

    Raises:
        ValueError: If job type is not found
    """
    config = JOB_TYPE_CONFIGS.get(job_type)
    if not config:
        available_types = ", ".join(JOB_TYPE_CONFIGS.keys())
        raise ValueError(
            f"Unknown job type: {job_type}. Available types: {available_types}"
        )
    return copy.deepcopy(config)


def get_available_job_types() -> List[str]:
    """Get list of available job types."""
    return list(JOB_TYPE_CONFIGS.keys()) 