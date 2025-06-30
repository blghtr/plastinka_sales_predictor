from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


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
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class JobsList(BaseModel):
    """List of jobs"""
    jobs: List[JobDetails]
    total: int
    
    model_config = ConfigDict(from_attributes=True)


# Data Upload Models

class DataUploadResponse(JobResponse):
    """Response model for data upload job"""
    pass


# Training Models

class ModelConfig(BaseModel):
    """Neural network model configuration"""
    num_encoder_layers: int = Field(..., description="Number of encoder layers", gt=0)
    num_decoder_layers: int = Field(..., description="Number of decoder layers", gt=0)
    decoder_output_dim: int = Field(..., description="Output dimension of decoder", gt=0)
    temporal_width_past: int = Field(..., description="Temporal width for past sequences", ge=0)
    temporal_width_future: int = Field(..., description="Temporal width for future sequences", ge=0)
    temporal_hidden_size_past: int = Field(..., description="Hidden size for past temporal network", gt=0)
    temporal_hidden_size_future: int = Field(..., description="Hidden size for future temporal network", gt=0)
    temporal_decoder_hidden: int = Field(..., description="Hidden size for temporal decoder", gt=0)
    batch_size: int = Field(..., description="Batch size for training", gt=0)
    dropout: float = Field(..., description="Dropout rate", ge=0, lt=1)
    use_reversible_instance_norm: bool = Field(..., description="Whether to use reversible instance normalization")
    use_layer_norm: bool = Field(..., description="Whether to use layer normalization")
    
    model_config = ConfigDict(from_attributes=True)


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""
    lr: float = Field(..., description="Learning rate", gt=0)
    weight_decay: float = Field(..., description="Weight decay (L2 penalty)", ge=0)
    
    model_config = ConfigDict(from_attributes=True)


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""
    T_0: int = Field(..., description="Number of iterations for the first restart", gt=0)
    T_mult: int = Field(..., description="Factor by which Ti increases after restart", ge=1)
    
    model_config = ConfigDict(from_attributes=True)


class TrainingDatasetConfig(BaseModel):
    """Training dataset configuration"""
    alpha: float = Field(..., description="Alpha config value for training", gt=0)
    span: int = Field(..., description="Span config value for training", gt=0)
    
    model_config = ConfigDict(from_attributes=True)


class SWAConfig(BaseModel):
    """Stochastic Weight Averaging configuration"""
    swa_lrs: float = Field(..., description="SWA learning rate", gt=0)
    swa_epoch_start: int = Field(..., description="Epoch to start SWA", gt=0)
    annealing_epochs: int = Field(..., description="Number of annealing epochs", gt=0)
    
    model_config = ConfigDict(from_attributes=True)


class WeightsConfig(BaseModel):
    """Model weights configuration"""
    sigma_left: float = Field(..., description="Left sigma config value", gt=0)
    sigma_right: float = Field(..., description="Right sigma config value", gt=0)
    
    model_config = ConfigDict(from_attributes=True)


class TrainingConfig(BaseModel):
    """Configuration for training job"""
    nn_model_config: ModelConfig = Field(
        ...,
        description="Neural network model configuration"
    )
    optimizer_config: OptimizerConfig = Field(
        ...,
        description="Optimizer configuration"
    )
    lr_shed_config: LRSchedulerConfig = Field(
        ...,
        description="Learning rate scheduler configuration"
    )
    train_ds_config: TrainingDatasetConfig = Field(
        ...,
        description="Training dataset configuration"
    )
    swa_config: Optional[SWAConfig] = Field(
        None,
        description="Stochastic Weight Averaging configuration"
    )
    weights_config: Optional[WeightsConfig] = Field(
        None,
        description="Model weights configuration"
    )
    lags: int = Field(
        ...,
        description="Number of lag periods to consider",
        gt=0
    )
    quantiles: Optional[List[float]] = Field(
        None,
        description="Quantiles for probabilistic forecasting"
    )
    model_id: Optional[str] = Field(
        None,
        description="ID of the model to use for training"
    )
    additional_hparams: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional training hparams"
    )
    
    model_config = ConfigDict(from_attributes=True)


class TrainingResponse(JobResponse):
    """Response model for training job"""
    model_id: Optional[str] = None
    config_id: Optional[str] = None
    using_active_parameters: Optional[bool] = None


# Prediction Models

class PredictionParams(BaseModel):
    """Config for prediction job"""
    model_id: str = Field(
        ...,
        description="ID of the model to use for prediction"
    )
    prediction_length: int = Field(
        ...,
        description="Number of time steps to predict",
        gt=0
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional prediction parameters"
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
        description="Type of report to generate (only prediction_report supported)"
    )
    prediction_month: datetime = Field(
        ...,
        description="Month for which to generate predictions (YYYY-MM format)"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Filters to apply to report data"
    )
    model_id: Optional[str] = Field(
        None,
        description="ID of the model to use for predictions in the report (defaults to active model)"
    )
    
    model_config = ConfigDict(from_attributes=True)


class ReportResponse(BaseModel):
    """Response model for report generation"""
    report_type: str = Field(..., description="Type of report generated")
    prediction_month: str = Field(..., description="Month for which predictions were generated")
    records_count: int = Field(..., description="Number of records in the report")
    csv_data: str = Field(..., description="CSV data as string")
    has_enriched_metrics: bool = Field(..., description="Whether report includes enriched metrics")
    enriched_columns: List[str] = Field(default_factory=list, description="List of enriched column names")
    generated_at: datetime = Field(..., description="When the report was generated")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Filters that were applied")
    
    model_config = ConfigDict(from_attributes=True)


# --- API Models for Model/Parameter Set Management ---

class ConfigResponse(BaseModel):
    """Response model for config information"""
    config_id: str
    configs: Dict[str, Any]
    is_active: Optional[bool] = False
    created_at: Optional[datetime] = None
    default_metric_name: Optional[str] = None
    default_metric_value: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class ModelResponse(BaseModel):
    """Response model for model information"""
    model_id: str
    model_path: str
    is_active: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    job_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DeleteIdsRequest(BaseModel):
    """Request model for deleting items by a list of IDs"""
    ids: List[str]

class DeleteResponse(BaseModel):
    """Response model for bulk delete operations"""
    successful: int
    failed: int
    errors: List[str] = []


class ConfigCreateRequest(BaseModel):
    """Request model for creating a config"""
    json_payload: Dict[str, Any] = Field(..., description="Config dictionary")
    is_active: Optional[bool] = Field(False, description="Set as active after creation")


class ModelCreateRequest(BaseModel):
    """Request model for creating a model record"""
    model_id: str = Field(..., description="Unique model identifier")
    job_id: str = Field(..., description="Job ID that produced the model")
    model_path: str = Field(..., description="Path to the model file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    is_active: Optional[bool] = Field(False, description="Set as active after creation")
    created_at: Optional[str] = Field(None, description="Creation timestamp (ISO format)") 