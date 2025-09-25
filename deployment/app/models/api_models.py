import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, model_validator


class YandexCloudToken(BaseModel):
    """Model for Yandex Cloud OAuth token with validation."""
    token: str = Field(..., pattern=r"^y[0-3]_[a-zA-Z0-9_-]+$", description="Yandex Cloud OAuth token")


class JobStatus(str, Enum):
    """Enum for job status values"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    """Enum for job types"""

    DATA_UPLOAD = "data_upload"
    MANUAL_UPLOAD = "manual_upload"
    TRAINING = "training"
    PREDICTION = "prediction"
    REPORT = "report"
    TUNING = "tuning"


class JobResponse(BaseModel):
    """Response model for job creation"""

    job_id: str
    status: JobStatus

    model_config = ConfigDict(from_attributes=True)


class JobDetails(BaseModel):
    """Detailed job information"""

    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)


class JobsList(BaseModel):
    """List of jobs"""

    jobs: list[JobDetails]
    total: int

    model_config = ConfigDict(from_attributes=True)


# Data Upload Models


class DataUploadResponse(JobResponse):
    """Response model for data upload job"""

    pass


# Training Models


class ModelConfig(BaseModel):
    """Neural network model configuration"""

    num_encoder_layers: int = Field(2, description="Number of encoder layers", gt=0)
    num_decoder_layers: int = Field(4, description="Number of decoder layers", gt=0)
    decoder_output_dim: int = Field(
        256, description="Output dimension of decoder", gt=0
    )
    temporal_width_past: int = Field(
        1, description="Temporal width for past sequences", ge=0
    )
    temporal_width_future: int = Field(
        0, description="Temporal width for future sequences", ge=0
    )
    temporal_hidden_size_past: int = Field(
        4, description="Hidden size for past temporal network", gt=0
    )
    temporal_hidden_size_future: int = Field(
        32, description="Hidden size for future temporal network", gt=0
    )
    temporal_decoder_hidden: int = Field(
        32, description="Hidden size for temporal decoder", gt=0
    )
    batch_size: int = Field(256, description="Batch size for training", gt=0)
    dropout: float = Field(0.498, description="Dropout rate", ge=0, lt=1)
    use_reversible_instance_norm: bool = Field(
        False, description="Whether to use reversible instance normalization"
    )
    use_layer_norm: bool = Field(False, description="Whether to use layer normalization")

    model_config = ConfigDict(from_attributes=True)


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""

    lr: float = Field(4.616, description="Learning rate", gt=0)
    weight_decay: float = Field(6.206, description="Weight decay (L2 penalty)", ge=0)

    model_config = ConfigDict(from_attributes=True)


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""

    T_0: int = Field(
        130, description="Number of iterations for the first restart", gt=0
    )
    T_mult: int = Field(
        3, description="Factor by which Ti increases after restart", ge=1
    )

    model_config = ConfigDict(from_attributes=True)


class TrainingDatasetConfig(BaseModel):
    """Training dataset configuration"""

    alpha: float = Field(2.556, description="Alpha config value for training", gt=0)
    span: int = Field(7, description="Span config value for training", gt=0)

    model_config = ConfigDict(from_attributes=True)


class SWAConfig(BaseModel):
    """Stochastic Weight Averaging configuration"""

    swa_lrs: float = Field(0.0001, description="SWA learning rate", gt=0)
    swa_epoch_start: float = Field(65, description="Epoch to start SWA", ge=0)
    annealing_epochs: int = Field(45, description="Number of annealing epochs", gt=0)

    model_config = ConfigDict(from_attributes=True)


class WeightsConfig(BaseModel):
    """Model weights configuration"""

    sigma_left: float = Field(0.754, description="Left sigma config value", gt=0)
    sigma_right: float = Field(1.390, description="Right sigma config value", gt=0)

    model_config = ConfigDict(from_attributes=True)


class TrainingConfig(BaseModel):
    """Configuration for training job"""

    nn_model_config: ModelConfig = Field(
        default_factory=ModelConfig, description="Neural network model configuration"
    )

    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    lr_shed_config: LRSchedulerConfig = Field(
        default_factory=LRSchedulerConfig, description="Learning rate scheduler configuration"
    )
    train_ds_config: TrainingDatasetConfig = Field(
        default_factory=TrainingDatasetConfig, description="Training dataset configuration"
    )
    swa_config: Optional[SWAConfig] = Field(
        default_factory=SWAConfig, description="Stochastic Weight Averaging configuration"
    )
    weights_config: Optional[WeightsConfig] = Field(
        default_factory=WeightsConfig, description="Model weights configuration"
    )
    lags: int = Field(9, description="Number of lag periods to consider", gt=0)
    quantiles: list[float] = Field(
        default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95],
        description="Quantiles for probabilistic forecasting"
    )
    model_id: str = Field(
        "4ff41e5d", 
        description="ID of the model to use for training"
    )
    additional_hparams: Optional[dict[str, Any]] = Field(
        None, description="Additional training hparams"
    )

    @model_validator(mode='before')
    @classmethod
    def _handle_model_config_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'model_config' in data:
            if 'nn_model_config' in data:
                raise ValueError("Cannot provide both 'model_config' and 'nn_model_config'")
            data['nn_model_config'] = data.pop('model_config')
        return data

    model_config = ConfigDict(from_attributes=True)


class TrainingResponse(JobResponse):
    """Response model for training job"""

    model_id: str | None = None
    config_id: str | None = None
    using_active_parameters: bool | None = None


# Prediction Models

class PredictionParams(BaseModel):
    """Config for prediction job"""

    model_id: str = Field(..., description="ID of the model to use for prediction")
    prediction_length: int = Field(
        ..., description="Number of time steps to predict", gt=0
    )
    additional_params: dict[str, Any] | None = Field(
        None, description="Additional prediction parameters"
    )

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(JobResponse):
    """Response model for prediction job"""

    pass


# Report Models


class ReportType(str, Enum):
    """Enum for report types"""

    PREDICTION_REPORT = "prediction_report"


class ReportParams(BaseModel):
    """Parameters for prediction report job"""

    report_type: ReportType = Field(
        ReportType.PREDICTION_REPORT,
        description="Type of report to generate (only prediction_report supported)",
    )
    prediction_month: date | None = Field(
        None,
        description="Month for which to generate predictions (YYYY-MM-DD). If null, latest available month is used.",
    )
    filters: dict[str, Any] | None = Field(
        None, description="Filters to apply to report data"
    )
    model_id: str | None = Field(
        None,
        description="ID of the model to use for predictions in the report (defaults to active model)",
    )

    model_config = ConfigDict(from_attributes=True)


class TrainingParams(BaseModel):
    """Parameters for a training job."""

    dataset_start_date: date | None = Field(
        None, description="Start date for the training dataset (YYYY-MM-DD). Optional, defaults to None."
    )
    dataset_end_date: date | None = Field(
        None, description="End date for the training dataset (YYYY-MM-DD). Optional, defaults to None."
    )

    @model_validator(mode="after")
    def check_date_range(self):
        from deployment.app.utils.validation import validate_date_range_or_none
        validate_date_range_or_none(self.dataset_start_date, self.dataset_end_date)
        return self

    model_config = ConfigDict(from_attributes=True)

class TuningParams(BaseModel):
    """Parameters for a tuning job."""

    dataset_start_date: date | None = Field(
        None,
        description="Start date for the training dataset (YYYY-MM-DD). Optional, defaults to None."
    )
    dataset_end_date: date | None = Field(
        None,
        description="End date for the training dataset (YYYY-MM-DD). Optional, defaults to None."
    )
    mode: str = Field(
        "lite",
        description="Tuning mode: full or lite",
        pattern="^(lite|full)$"
    )
    time_budget_s: int | None = Field(
        None,
        description="Time budget for tuning in seconds",
        gt=0
    )

    @model_validator(mode="after")
    def check_date_range(self):
        from deployment.app.utils.validation import validate_date_range_or_none
        validate_date_range_or_none(self.dataset_start_date, self.dataset_end_date)
        return self

    model_config = ConfigDict(from_attributes=True)

class ReportResponse(BaseModel):
    """Response model for report generation"""

    report_type: str = Field(..., description="Type of report generated")
    prediction_month: str = Field(
        ..., description="Month for which predictions were generated"
    )
    records_count: int = Field(..., description="Number of records in the report")
    csv_data: str = Field(..., description="CSV data as string")
    generated_at: datetime = Field(..., description="When the report was generated")
    filters_applied: dict[str, Any] | None = Field(
        None, description="Filters that were applied"
    )

    model_config = ConfigDict(from_attributes=True)


# --- API Models for Model/Parameter Set Management ---


class ConfigResponse(BaseModel):
    """Response model for config information"""

    config_id: str
    config: dict[str, Any]
    is_active: bool | None = False
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class TrainingResultResponse(BaseModel):
    """Response model for a single training result"""

    result_id: str
    job_id: str
    model_id: str
    config_id: str
    metrics: dict[str, Any]
    duration: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TuningResultResponse(BaseModel):
    """Response model for a single tuning result"""

    result_id: str
    job_id: str
    config_id: str
    metrics: dict[str, Any]
    duration: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelResponse(BaseModel):
    """Response model for model information"""

    model_id: str
    model_path: str
    is_active: bool | None = False
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    job_id: str | None = None

    model_config = ConfigDict(from_attributes=True)


class DeleteIdsRequest(BaseModel):
    """Request model for deleting items by a list of IDs"""

    ids: list[str]


class DeleteResponse(BaseModel):
    """Response model for bulk delete operations"""

    successful: int
    failed: int
    errors: list[str] = []


class ConfigCreateRequest(BaseModel):
    """Request model for creating a config"""

    json_payload: dict[str, Any] = Field(..., description="Config dictionary")
    is_active: bool | None = Field(False, description="Set as active after creation")


class ModelCreateRequest(BaseModel):
    """Request model for creating a model record"""

    model_id: str = Field(..., description="Unique model identifier")
    job_id: str = Field(..., description="Job ID that produced the model")
    model_path: str = Field(..., description="Path to the model file")
    metadata: dict[str, Any] | None = Field(None, description="Model metadata")
    is_active: bool | None = Field(False, description="Set as active after creation")
    created_at: str | None = Field(None, description="Creation timestamp (ISO format)")


class ModelUploadMetadata(BaseModel):
    """Metadata for uploaded models."""
    description: str | None = Field(None, description="A brief description of the model.")
    version: str | None = Field(None, description="Version of the model.")
    tags: list[str] | None = Field(None, description="List of tags for categorization.")
    training_metrics: dict[str, Any] | None = Field(None, description="Metrics from the training run that produced this model.")
    source_job_id: str | None = Field(None, description="ID of the job that created this model.")

    @classmethod
    def as_form(
        cls,
        description: str | None = Form(None, description="A brief description of the model."),
        version: str | None = Form(None, description="Version of the model."),
        tags: str | None = Form(None, description="Comma-separated list of tags for categorization."),
        training_metrics: str | None = Form(None, description="JSON string of metrics from the training run."),
        source_job_id: str | None = Form(None, description="ID of the job that created this model."),
    ):
        parsed_tags = tags.split(',') if tags else None
        parsed_metrics = json.loads(training_metrics) if training_metrics else None
        return cls(
            description=description,
            version=version,
            tags=parsed_tags,
            training_metrics=parsed_metrics,
            source_job_id=source_job_id,
        )


class DataUploadFormParameters(BaseModel):
    """Form parameters for data upload."""
    overwrite: bool | None = Field(False, description="Whether to overwrite existing features.")
    @classmethod
    def as_form(
        cls,
        overwrite: bool | None = Form(False, description="Whether to overwrite existing features."),
    ):
        return cls(overwrite=overwrite)


class ErrorDetailResponse(BaseModel):
    """Standardized error response model."""

    message: str = Field(..., description="A human-readable error message.")
    code: str | None = Field(None, description="An internal code representing the error type.")
    status_code: int | None = Field(None, description="The HTTP status code associated with the error.")
    details: dict[str, Any] | None = Field(None, description="Additional details about the error.")

    model_config = ConfigDict(from_attributes=True)
