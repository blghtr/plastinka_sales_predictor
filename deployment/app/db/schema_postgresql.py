"""
PostgreSQL schema for Plastinka ML Service.

This module contains the PostgreSQL DDL schema converted from SQLite.
Key conversions:
- INTEGER PRIMARY KEY AUTOINCREMENT → BIGSERIAL PRIMARY KEY
- TEXT for JSON fields → JSONB
- TIMESTAMP → TIMESTAMP WITH TIME ZONE
- BOOLEAN DEFAULT 0 → BOOLEAN DEFAULT FALSE
- REAL → DOUBLE PRECISION
- DECIMAL(10,2) → NUMERIC(10,2)
"""

import logging
from asyncpg import Pool

logger = logging.getLogger(__name__)

# PostgreSQL schema SQL
SCHEMA_SQL = """
-- Dimension Tables
CREATE TABLE IF NOT EXISTS dim_multiindex_mapping (
    multiindex_id BIGSERIAL PRIMARY KEY,
    barcode TEXT,  -- Штрихкод
    artist TEXT,   -- Исполнитель
    album TEXT,    -- Альбом
    cover_type TEXT, -- Конверт
    price_category TEXT, -- Ценовая категория
    release_type TEXT,  -- Тип
    recording_decade TEXT, -- Год записи
    release_decade TEXT,  -- Год выпуска
    style TEXT,     -- Стиль
    recording_year INTEGER, -- recording_year
    UNIQUE(barcode, artist, album, cover_type, price_category, release_type,
           recording_decade, release_decade, style, recording_year)
);

-- Fact Tables (with standardized column names)

-- fact_stock is obsolete and now removed from the code : stock now not used

-- fact_prices is obsolete and now removed from the code : prices now implicitly stored as part of multiindex_id

CREATE TABLE IF NOT EXISTS fact_sales (
    multiindex_id BIGINT,
    data_date DATE,
    value DOUBLE PRECISION,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_stock_movement (
    multiindex_id BIGINT,
    data_date DATE,
    value DOUBLE PRECISION,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

-- Note: fact_predictions table is created later, after models and prediction_results tables

-- Utility Tables
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id BIGSERIAL PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    status TEXT,
    source_files TEXT
);

-- New table for unique configs (must be created before jobs due to FK constraint)
CREATE TABLE IF NOT EXISTS configs (
    config_id TEXT PRIMARY KEY, -- Could be a hash of the parameters
    config JSONB NOT NULL,       -- JSON string of the parameters
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    source TEXT                 -- 'manual' | 'tuning' | NULL
);

-- Job Tables
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,  -- 'data_upload', 'training', 'prediction', 'report'
    status TEXT NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    config_id TEXT,          -- Reference to configs table
    model_id TEXT,           -- Reference to models table (no FK constraint due to circular dependency)
    parameters JSONB,       -- JSON of job parameters
    result_id TEXT,          -- ID of the result resource (if applicable)
    error_message TEXT,
    progress DOUBLE PRECISION,  -- 0-100 percentage
    requirements_hash TEXT,  -- SHA256 hash of requirements.txt for optimization through cloning
    parent_job_id TEXT,      -- ID of source job when cloning (NULL for new jobs)
    datasphere_job_id TEXT,  -- DataSphere job ID returned by DataSphere API (NULL for non-datasphere jobs)
    FOREIGN KEY (config_id) REFERENCES configs(config_id)
);

-- Job status history table for tracking status changes over time
CREATE TABLE IF NOT EXISTS job_status_history (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    progress DOUBLE PRECISION,  -- 0-100 percentage
    status_message TEXT,     -- Detailed status message
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS data_upload_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    records_processed INTEGER,
    features_generated JSONB, -- JSON list of feature types
    processing_run_id BIGINT,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (processing_run_id) REFERENCES processing_runs(run_id)
);

-- New table for model metadata
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_path TEXT NOT NULL,      -- Path to the saved model file (e.g., .onnx or .pt)
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB,                -- Optional JSON for file size, framework, etc.
    is_active BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS training_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT,
    config_id TEXT,
    metrics JSONB, -- JSON dictionary of metrics
    duration DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (config_id) REFERENCES configs(config_id)
);

CREATE TABLE IF NOT EXISTS prediction_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE,
    prediction_month DATE,     -- New field: month for which predictions were made
    output_path TEXT,       -- Path to prediction output file
    summary_metrics JSONB,   -- JSON of prediction metrics
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

-- New fact table for predictions storage (created after models and prediction_results)
CREATE TABLE IF NOT EXISTS fact_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    multiindex_id BIGINT NOT NULL,
    prediction_month DATE NOT NULL,
    result_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    quantile_05 NUMERIC(10,2) NOT NULL,
    quantile_25 NUMERIC(10,2) NOT NULL,
    quantile_50 NUMERIC(10,2) NOT NULL,
    quantile_75 NUMERIC(10,2) NOT NULL,
    quantile_95 NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (result_id) REFERENCES prediction_results(result_id),
    UNIQUE(multiindex_id, prediction_month, model_id)
);

CREATE TABLE IF NOT EXISTS report_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    report_type TEXT,
    parameters JSONB,        -- JSON of report parameters
    output_path TEXT,       -- Path to generated report
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS tuning_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    metrics JSONB,
    duration DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (config_id) REFERENCES configs(config_id)
);

-- Submission locks for refractory period per job_type + parameter hash
CREATE TABLE IF NOT EXISTS job_submission_locks (
    job_type TEXT NOT NULL,
    param_hash TEXT NOT NULL,
    lock_until TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (job_type, param_hash)
);


CREATE TABLE IF NOT EXISTS report_features (
    data_date DATE NOT NULL,
    multiindex_id BIGINT NOT NULL,
    availability DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    masked_mean_sales_items DOUBLE PRECISION,
    masked_mean_sales_rub DOUBLE PRECISION,
    lost_sales DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (data_date, multiindex_id),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS retry_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    component TEXT,
    operation TEXT,
    attempt INTEGER,
    max_attempts INTEGER,
    successful BOOLEAN,
    duration_ms INTEGER,
    exception_type TEXT,
    exception_message TEXT
);

-- Indexes for optimization
CREATE INDEX IF NOT EXISTS idx_multiindex_barcode ON dim_multiindex_mapping(barcode);
CREATE INDEX IF NOT EXISTS idx_multiindex_artist_album ON dim_multiindex_mapping(artist, album);
CREATE INDEX IF NOT EXISTS idx_multiindex_style ON dim_multiindex_mapping(style);
CREATE INDEX IF NOT EXISTS idx_multiindex_multiindex_id ON dim_multiindex_mapping(multiindex_id);


CREATE INDEX IF NOT EXISTS idx_sales_date ON fact_sales(data_date);
CREATE INDEX IF NOT EXISTS idx_movement_date ON fact_stock_movement(data_date);

-- Indexes for predictions
CREATE INDEX IF NOT EXISTS idx_predictions_multiindex ON fact_predictions(multiindex_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON fact_predictions(prediction_month);
CREATE INDEX IF NOT EXISTS idx_predictions_result ON fact_predictions(result_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON fact_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date_model ON fact_predictions(prediction_month, model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date_multiindex ON fact_predictions(prediction_month, multiindex_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date_multiindex_model ON fact_predictions(prediction_month, multiindex_id, model_id);

-- Index for prediction_results
CREATE INDEX IF NOT EXISTS idx_prediction_results_month ON prediction_results(prediction_month);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type);

-- New indexes for job cloning functionality
CREATE INDEX IF NOT EXISTS idx_jobs_requirements_hash ON jobs(requirements_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_parent_job_id ON jobs(parent_job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_datasphere_job_id ON jobs(datasphere_job_id);

-- Index for job status history
CREATE INDEX IF NOT EXISTS idx_job_history_job_id ON job_status_history(job_id);
CREATE INDEX IF NOT EXISTS idx_job_history_updated_at ON job_status_history(updated_at);
CREATE INDEX IF NOT EXISTS idx_job_history_job_updated ON job_status_history(job_id, updated_at);

-- Indexes for models and training results to improve data retention queries
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at);
CREATE INDEX IF NOT EXISTS idx_models_active_created ON models(is_active, created_at);
CREATE INDEX IF NOT EXISTS idx_training_results_config ON training_results(config_id);
CREATE INDEX IF NOT EXISTS idx_training_results_model ON training_results(model_id);

-- Indexes for tuning_results
CREATE INDEX IF NOT EXISTS idx_tuning_results_config ON tuning_results(config_id);
CREATE INDEX IF NOT EXISTS idx_tuning_results_job ON tuning_results(job_id);
CREATE INDEX IF NOT EXISTS idx_tuning_results_created ON tuning_results(created_at);

CREATE INDEX IF NOT EXISTS idx_retry_events_op ON retry_events(component, operation);
CREATE INDEX IF NOT EXISTS idx_retry_events_time ON retry_events(timestamp);

-- Index to help cleanups of stale locks
CREATE INDEX IF NOT EXISTS idx_job_submission_locks_until ON job_submission_locks(lock_until);

-- GIN indexes for JSONB fields to improve query performance
CREATE INDEX IF NOT EXISTS idx_jobs_parameters_gin ON jobs USING GIN (parameters);
CREATE INDEX IF NOT EXISTS idx_configs_config_gin ON configs USING GIN (config);
CREATE INDEX IF NOT EXISTS idx_training_results_metrics_gin ON training_results USING GIN (metrics);
CREATE INDEX IF NOT EXISTS idx_tuning_results_metrics_gin ON tuning_results USING GIN (metrics);
CREATE INDEX IF NOT EXISTS idx_prediction_results_summary_metrics_gin ON prediction_results USING GIN (summary_metrics);
CREATE INDEX IF NOT EXISTS idx_models_metadata_gin ON models USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_data_upload_results_features_gin ON data_upload_results USING GIN (features_generated);
CREATE INDEX IF NOT EXISTS idx_report_results_parameters_gin ON report_results USING GIN (parameters);
"""

MULTIINDEX_NAMES = [
    "barcode",
    "artist",
    "album",
    "cover_type",
    "price_category",
    "release_type",
    "recording_decade",
    "release_decade",
    "style",
    "recording_year",
]


async def init_postgres_schema(pool: Pool) -> bool:
    """
    Initialize PostgreSQL database with schema.
    
    This function is idempotent - running it multiple times is safe.
    Tables and indexes are created only if they don't already exist.
    
    Args:
        pool: PostgreSQL connection pool
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        async with pool.acquire() as conn:
            # Enable foreign key constraints (PostgreSQL has them enabled by default, but ensure)
            await conn.execute("SET session_replication_role = 'replica';")
            
            # Execute schema SQL
            await conn.execute(SCHEMA_SQL)
            
            logger.info("PostgreSQL schema initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"PostgreSQL schema initialization failed: {e}", exc_info=True)
        return False

