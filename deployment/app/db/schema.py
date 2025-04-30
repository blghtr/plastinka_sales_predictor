import sqlite3
from pathlib import Path
from datetime import datetime
import json

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

-- Fact Tables
CREATE TABLE IF NOT EXISTS fact_stock (
    multiindex_id INTEGER,
    snapshot_date DATE,
    quantity REAL,
    PRIMARY KEY (multiindex_id, snapshot_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_prices (
    multiindex_id INTEGER,
    price_date DATE,
    price DECIMAL(10,2),
    PRIMARY KEY (multiindex_id, price_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_sales (
    multiindex_id INTEGER,
    sale_date DATE,
    quantity REAL,
    PRIMARY KEY (multiindex_id, sale_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
);

CREATE TABLE IF NOT EXISTS fact_stock_changes (
    multiindex_id INTEGER,
    change_date DATE,
    quantity_change REAL,
    PRIMARY KEY (multiindex_id, change_date),
    FOREIGN KEY (multiindex_id) REFERENCES dim_multiindex_mapping(multiindex_id)
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
    job_type TEXT NOT NULL,  -- 'data_upload', 'training', 'prediction', 'report', 'cloud_training', 'cloud_prediction'
    status TEXT NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    parameters TEXT,         -- JSON of job parameters
    result_id TEXT,          -- ID of the result resource (if applicable)
    error_message TEXT,
    progress REAL           -- 0-100 percentage
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

CREATE TABLE IF NOT EXISTS training_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT,
    metrics TEXT,           -- JSON of training metrics
    parameters TEXT,        -- JSON of training parameters
    duration INTEGER,       -- in seconds
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS prediction_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prediction_date TIMESTAMP,
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

-- Cloud Function Tables
CREATE TABLE IF NOT EXISTS cloud_functions (
    function_id TEXT PRIMARY KEY,
    function_type TEXT NOT NULL,  -- 'training' or 'prediction'
    function_name TEXT NOT NULL,  -- The actual cloud function name
    created_at TIMESTAMP NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,         -- 'active', 'updating', 'error'
    config TEXT                   -- JSON of function configuration
);

CREATE TABLE IF NOT EXISTS cloud_function_executions (
    execution_id TEXT PRIMARY KEY,
    function_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,         -- 'pending', 'running', 'completed', 'failed'
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    request_payload TEXT,         -- JSON of the request sent to the cloud function
    response_payload TEXT,        -- JSON of the final response from the cloud function
    cloud_request_id TEXT,        -- Cloud provider's request identifier
    error_message TEXT,
    FOREIGN KEY (function_id) REFERENCES cloud_functions(function_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS cloud_function_status_updates (
    update_id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    status TEXT NOT NULL,         -- 'running', 'success', 'error'
    progress_percentage REAL,     -- 0-100 percentage
    current_step TEXT,            -- Description of current step
    steps_completed INTEGER,      -- Number of completed steps
    steps_total INTEGER,          -- Total number of steps
    logs TEXT,                    -- JSON array of recent log entries
    error_code TEXT,              -- Error code if status is 'error'
    error_message TEXT,           -- Error message if status is 'error'
    error_details TEXT,           -- JSON of error details if status is 'error'
    FOREIGN KEY (execution_id) REFERENCES cloud_function_executions(execution_id)
);

CREATE TABLE IF NOT EXISTS cloud_storage_objects (
    object_id TEXT PRIMARY KEY,
    bucket_name TEXT NOT NULL,
    object_path TEXT NOT NULL,
    content_type TEXT,
    size_bytes INTEGER,
    created_at TIMESTAMP NOT NULL,
    md5_hash TEXT,                -- For integrity verification
    related_job_id TEXT,          -- Job that created or uses this object
    object_type TEXT NOT NULL,    -- 'dataset', 'model', 'result'
    expiration_time TIMESTAMP,    -- When the object should be deleted (if temporary)
    FOREIGN KEY (related_job_id) REFERENCES jobs(job_id),
    UNIQUE(bucket_name, object_path)
);

-- Indexes for optimization
CREATE INDEX IF NOT EXISTS idx_multiindex_barcode ON dim_multiindex_mapping(barcode);
CREATE INDEX IF NOT EXISTS idx_multiindex_artist_album ON dim_multiindex_mapping(artist, album);
CREATE INDEX IF NOT EXISTS idx_multiindex_style ON dim_multiindex_mapping(style);

CREATE INDEX IF NOT EXISTS idx_stock_date ON fact_stock(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_sales_date ON fact_sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_changes_date ON fact_stock_changes(change_date);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type);

-- Cloud function indexes
CREATE INDEX IF NOT EXISTS idx_cloud_func_type ON cloud_functions(function_type);
CREATE INDEX IF NOT EXISTS idx_cloud_func_exec_job ON cloud_function_executions(job_id);
CREATE INDEX IF NOT EXISTS idx_cloud_func_exec_status ON cloud_function_executions(status);
CREATE INDEX IF NOT EXISTS idx_cloud_status_exec_id ON cloud_function_status_updates(execution_id);
CREATE INDEX IF NOT EXISTS idx_cloud_storage_job ON cloud_storage_objects(related_job_id);
CREATE INDEX IF NOT EXISTS idx_cloud_storage_type ON cloud_storage_objects(object_type);
"""

def init_db(db_path: str = "deployment/data/plastinka.db"):
    """
    Initialize the database with schema.
    
    This function is idempotent - running it multiple times is safe.
    Tables and indexes are created only if they don't already exist.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if database already exists
        db_exists = Path(db_path).exists()
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute schema creation SQL
        cursor.executescript(SCHEMA_SQL)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        if db_exists:
            print(f"Database schema validated at {db_path}")
        else:
            print(f"Database initialized at {db_path}")
        
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    init_db() 