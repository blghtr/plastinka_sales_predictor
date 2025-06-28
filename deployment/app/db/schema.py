import sqlite3
from pathlib import Path
from datetime import datetime
import json
import logging

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

CREATE TABLE IF NOT EXISTS dim_price_categories (
    category_id INTEGER PRIMARY KEY,
    range_start DECIMAL(10,2),
    range_end DECIMAL(10,2),
    category_name TEXT,
    UNIQUE(category_name)
);

CREATE TABLE IF NOT EXISTS dim_styles (
    style_id INTEGER PRIMARY KEY,
    style_name TEXT,
    parent_style TEXT,
    UNIQUE(style_name)
);

-- Fact Tables (with standardized column names)
CREATE TABLE IF NOT EXISTS fact_stock (
    multiindex_id INTEGER,
    data_date DATE,
    value REAL,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_prices (
    multiindex_id INTEGER,
    data_date DATE,
    value REAL,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_sales (
    multiindex_id INTEGER,
    data_date DATE,
    value REAL,
    PRIMARY KEY (multiindex_id, data_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_stock_changes (
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
    prediction_date TIMESTAMP NOT NULL,
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
    UNIQUE(multiindex_id, prediction_date, model_id)
);

-- Utility Tables
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id INTEGER PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT,
    cutoff_date DATE,
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
    is_active BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT,
    config_id TEXT,  -- New column added
    metrics TEXT,           -- JSON of training metrics
    duration INTEGER,       -- in seconds
    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id), -- Added FK to new models table
    FOREIGN KEY (config_id) REFERENCES configs(config_id) -- Added FK
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

-- Indexes for optimization
CREATE INDEX IF NOT EXISTS idx_multiindex_barcode ON dim_multiindex_mapping(barcode);
CREATE INDEX IF NOT EXISTS idx_multiindex_artist_album ON dim_multiindex_mapping(artist, album);
CREATE INDEX IF NOT EXISTS idx_multiindex_style ON dim_multiindex_mapping(style);

CREATE INDEX IF NOT EXISTS idx_stock_date ON fact_stock(data_date);
CREATE INDEX IF NOT EXISTS idx_sales_date ON fact_sales(data_date);
CREATE INDEX IF NOT EXISTS idx_changes_date ON fact_stock_changes(data_date);

-- Indexes for predictions
CREATE INDEX IF NOT EXISTS idx_predictions_multiindex ON fact_predictions(multiindex_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON fact_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_result ON fact_predictions(result_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON fact_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date_model ON fact_predictions(prediction_date, model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date_multiindex ON fact_predictions(prediction_date, multiindex_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date_multiindex_model ON fact_predictions(prediction_date, multiindex_id, model_id);

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
"""

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
    conn_created = False
    original_row_factory = None
    
    try:
        if connection:
            conn = connection
        elif db_path:
            # Parse SQLite URI scheme if present
            parsed_db_path = db_path
            if parsed_db_path.startswith("sqlite:///"):
                parsed_db_path = parsed_db_path[len("sqlite:///"):]
            elif parsed_db_path.startswith("sqlite://"):
                parsed_db_path = parsed_db_path[len("sqlite://"):]
            elif parsed_db_path.startswith("sqlite:"):
                parsed_db_path = parsed_db_path[len("sqlite:"):]
            
            if not parsed_db_path:
                raise ValueError(f"Database path became empty after parsing scheme from: {db_path}")

            # Ensure directory exists
            actual_file_path = Path(parsed_db_path)
            actual_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            conn_created = True

        if conn:
            # Store and reset row_factory for schema operations
            original_row_factory = conn.row_factory
            conn.row_factory = None
            
            # Enable foreign keys and execute schema
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.executescript(SCHEMA_SQL)
            
            if conn_created:
                conn.commit()
            
            return True
        else:
            logger.error("No database connection established")
            return False

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        if conn_created and conn:
            conn.rollback()
        return False
    finally:
        # Restore original row_factory
        if connection and original_row_factory is not None:
            connection.row_factory = original_row_factory
            
        # Close connection if we created it
        if conn_created and conn:
            try:
                conn.close()
            except Exception as close_e:
                logger.error(f"Error closing database connection: {close_e}")

def migrate_add_prediction_month(db_path: str = None, connection: sqlite3.Connection = None):
    """
    Migration to add prediction_month column to prediction_results table.
    This function is idempotent - running it multiple times is safe.
    
    Args:
        db_path: Path to SQLite database file (optional, if connection is provided)
        connection: Optional existing database connection. If provided, db_path is ignored.
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = None
    conn_created = False
    original_row_factory = None
    
    try:
        if connection:
            conn = connection
        elif db_path:
            # Parse SQLite URI scheme if present
            parsed_db_path = db_path
            if parsed_db_path.startswith("sqlite:///"):
                parsed_db_path = parsed_db_path[len("sqlite:///"):]
            elif parsed_db_path.startswith("sqlite://"):
                parsed_db_path = parsed_db_path[len("sqlite://"):]
            elif parsed_db_path.startswith("sqlite:"):
                parsed_db_path = parsed_db_path[len("sqlite:"):]
            
            if not parsed_db_path:
                raise ValueError(f"Database path became empty after parsing scheme from: {db_path}")

            # Ensure directory exists
            actual_file_path = Path(parsed_db_path)
            actual_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            conn_created = True

        if conn:
            # Store and reset row_factory for schema operations
            original_row_factory = conn.row_factory
            conn.row_factory = None
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            
            # Check if column already exists
            cursor.execute("PRAGMA table_info(prediction_results)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'prediction_month' not in columns:
                logger.info("Adding prediction_month column to prediction_results table")
                cursor.execute("ALTER TABLE prediction_results ADD COLUMN prediction_month DATE")
                
                # Add the index
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_results_month ON prediction_results(prediction_month)")
                
                if conn_created:
                    conn.commit()
                logger.info("Successfully added prediction_month column and index")
            else:
                logger.info("prediction_month column already exists, skipping migration")
            
            return True
        else:
            logger.error("No database connection established for migration")
            return False

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        if conn_created and conn:
            conn.rollback()
        return False
    finally:
        # Restore original row_factory
        if connection and original_row_factory is not None:
            connection.row_factory = original_row_factory
            
        # Close connection if we created it
        if conn_created and conn:
            try:
                conn.close()
            except Exception as close_e:
                logger.error(f"Error closing database connection: {close_e}")

if __name__ == "__main__":
    init_db() 