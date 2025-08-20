import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# SQL statements for creating the database schema
SCHEMA_SQL = """
-- Dimension Tables
CREATE TABLE IF NOT EXISTS dim_multiindex_mapping (
    multiindex_id INTEGER PRIMARY KEY,
    barcode TEXT,  -- Штрихкод
    artist TEXT,   -- Исполнитель
    album TEXT,    -- Альбом
    cover_type TEXT, -- Конверт
    price_category TEXT, -- Ценовая категория
    release_type TEXT,  -- Тип
    recording_decade TEXT, -- Год записи
    release_decade TEXT,  -- Год выпуска
    style TEXT,     -- Стиль
    record_year INTEGER, -- precise_record_year
    UNIQUE(barcode, artist, album, cover_type, price_category, release_type,
           recording_decade, release_decade, style, record_year)
);

-- Fact Tables (with standardized column names)

-- fact_stock is obsolete and now removed from the code : stock now not used

-- fact_prices is obsolete and now removed from the code : prices now implicitly stored as part of multiindex_id

CREATE TABLE IF NOT EXISTS fact_sales (
    multiindex_id INTEGER,
    data_date DATE,
    value REAL,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_stock_movement (
    multiindex_id INTEGER,
    data_date DATE,
    value REAL,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

-- New fact table for predictions storage
CREATE TABLE IF NOT EXISTS fact_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    multiindex_id INTEGER NOT NULL,
    prediction_month DATE NOT NULL,
    result_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    quantile_05 DECIMAL(10,2) NOT NULL,
    quantile_25 DECIMAL(10,2) NOT NULL,
    quantile_50 DECIMAL(10,2) NOT NULL,
    quantile_75 DECIMAL(10,2) NOT NULL,
    quantile_95 DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (result_id) REFERENCES prediction_results(result_id),
    UNIQUE(multiindex_id, prediction_month, model_id)
);

-- Utility Tables
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id INTEGER PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT,
    source_files TEXT
);

-- Job Tables
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,  -- 'data_upload', 'training', 'prediction', 'report'
    status TEXT NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    config_id TEXT,          -- Reference to configs table
    model_id TEXT,           -- Reference to models table (no FK constraint due to circular dependency)
    parameters TEXT,         -- JSON of job parameters
    result_id TEXT,          -- ID of the result resource (if applicable)
    error_message TEXT,
    progress REAL,           -- 0-100 percentage
    requirements_hash TEXT,  -- SHA256 hash of requirements.txt for optimization through cloning
    parent_job_id TEXT,      -- ID of source job when cloning (NULL for new jobs)
    datasphere_job_id TEXT,  -- DataSphere job ID returned by DataSphere API (NULL for non-datasphere jobs)
    FOREIGN KEY (config_id) REFERENCES configs(config_id)
);

-- Job status history table for tracking status changes over time
CREATE TABLE IF NOT EXISTS job_status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    progress REAL,           -- 0-100 percentage
    status_message TEXT,     -- Detailed status message
    updated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS data_upload_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    records_processed INTEGER,
    features_generated TEXT, -- JSON list of feature types
    processing_run_id INTEGER,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (processing_run_id) REFERENCES processing_runs(run_id)
);

-- New table for model metadata
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_path TEXT NOT NULL,      -- Path to the saved model file (e.g., .onnx or .pt)
    created_at TIMESTAMP NOT NULL,
    metadata TEXT,                 -- Optional JSON for file size, framework, etc.
    is_active BOOLEAN DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

-- New table for unique configs
CREATE TABLE IF NOT EXISTS configs (
    config_id TEXT PRIMARY KEY, -- Could be a hash of the parameters
    config TEXT NOT NULL,        -- JSON string of the parameters
    created_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT 0,
    source TEXT                 -- 'manual' | 'tuning' | NULL
);

CREATE TABLE IF NOT EXISTS training_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT,
    config_id TEXT,
    metrics TEXT, -- JSON dictionary of metrics
    duration REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (config_id) REFERENCES configs(config_id)
);

CREATE TABLE IF NOT EXISTS prediction_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prediction_date TIMESTAMP,
    prediction_month DATE,     -- New field: month for which predictions were made
    output_path TEXT,       -- Path to prediction output file
    summary_metrics TEXT,   -- JSON of prediction metrics
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS report_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    report_type TEXT,
    parameters TEXT,        -- JSON of report parameters
    output_path TEXT,       -- Path to generated report
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS tuning_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    metrics TEXT,
    duration REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (config_id) REFERENCES configs(config_id)
);


CREATE TABLE IF NOT EXISTS report_features (
    data_date DATE NOT NULL,
    multiindex_id INTEGER NOT NULL,
    availability REAL,
    confidence REAL,
    masked_mean_sales_items REAL,
    masked_mean_sales_rub REAL,
    lost_sales REAL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (data_date, multiindex_id),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS retry_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
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
    "record_year",
]


def init_db(db_path: str = None, connection: sqlite3.Connection = None):
    """
    Initialize the database with schema.

    This function is idempotent - running it multiple times is safe.
    Tables and indexes are created only if they don't already exist.

    Args:
        db_path: Path to SQLite database file (optional, if connection is provided)
        connection: Optional existing database connection. If provided, db_path is ignored.

    Returns:
        bool: True if successful, False otherwise
    """
    conn = None
    conn_created_internally = False
    original_row_factory = None

    try:
        if connection:
            conn = connection
        elif db_path:
            # Убираем SQLite префиксы из пути
            sqlite_prefixes = ["sqlite:///", "sqlite://", "sqlite:"]
            parsed_db_path = db_path
            
            for prefix in sqlite_prefixes:
                if parsed_db_path.startswith(prefix):
                    parsed_db_path = parsed_db_path[len(prefix):]
                    break

            if not parsed_db_path:
                raise ValueError(
                    f"Database path became empty after parsing scheme from: {db_path}"
                )

            actual_file_path = Path(parsed_db_path)
            actual_file_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(db_path)
            conn_created_internally = True

            try:
                os.chmod(str(actual_file_path), 0o600)
                logger.info(f"Set database file permissions for {actual_file_path} to 0o600.")
            except OSError as e:
                logger.warning(f"Could not set database file permissions for {actual_file_path}: {e}")

        if conn:
            original_row_factory = conn.row_factory
            conn.row_factory = None

            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.executescript(SCHEMA_SQL)
            conn.commit()

            return True
        else:
            logger.error("No database connection established")
            return False

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        if conn and conn_created_internally: # Only rollback if we created the connection
            conn.rollback()
        return False
    finally:
        if connection and original_row_factory is not None:
            connection.row_factory = original_row_factory

        if conn and conn_created_internally: # Only close if we created the connection
            try:
                conn.close()
            except Exception as close_e:
                logger.error(f"Error closing database connection: {close_e}")


if __name__ == "__main__":
    init_db()
