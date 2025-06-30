import sqlite3

import pytest

from deployment.app.db.database import (  # Added DatabaseError import
    DatabaseError,
    dict_factory,
    execute_query,
)
from deployment.app.db.schema import SCHEMA_SQL, init_db  # Import init_db

# Use a dedicated in-memory database for these tests to avoid interference
TEST_DB_PATH = ":memory:"

@pytest.fixture(scope="function")
def db_conn():
    """Fixture to set up and tear down the in-memory database for each test."""
    # Directly connect to the in-memory database for the test
    conn = sqlite3.connect(TEST_DB_PATH)

    # Enable Foreign Key support
    conn.execute("PRAGMA foreign_keys = ON;")

    # Set dict_factory for this connection
    conn.row_factory = dict_factory # Use dict_factory

    # Initialize schema using init_db
    init_db(connection=conn)

    # Removed: cursor = conn.cursor() # No longer needed here as init_db handles it
    # Removed: cursor.executescript(SCHEMA_SQL) # Redundant call as init_db already does this
    conn.commit() # Commit changes made by init_db for this connection

    # Diagnostic print
    original_row_factory = conn.row_factory
    conn.row_factory = None # Temporarily set to None for PRAGMA queries
    try:
        debug_cursor = conn.cursor()
        debug_cursor.execute("PRAGMA table_info(training_results);")
        columns_info = debug_cursor.fetchall()
        column_names = [info[1] for info in columns_info] # Extract just the names
        print(f"DEBUG: test_foreign_key_constraints.py: db_conn fixture: training_results column names: {column_names}")

        schema_sql_snippet_start = SCHEMA_SQL.find('CREATE TABLE IF NOT EXISTS training_results')
        if schema_sql_snippet_start != -1:
            schema_sql_snippet = SCHEMA_SQL[schema_sql_snippet_start:schema_sql_snippet_start + 300]
            print(f"DEBUG: test_foreign_key_constraints.py: db_conn fixture: SCHEMA_SQL for training_results (snippet): {schema_sql_snippet}")
        else:
            print("DEBUG: test_foreign_key_constraints.py: db_conn fixture: Could not find training_results DDL in SCHEMA_SQL")
    finally:
        conn.row_factory = original_row_factory # Restore original row_factory

    # Verify PRAGMA is ON
    cursor = conn.cursor() # Re-initialize cursor for the assertion below
    cursor.execute("PRAGMA foreign_keys;")
    fk_status = cursor.fetchone()
    assert fk_status['foreign_keys'] == 1, "Foreign keys should be ON for the test connection"

    yield conn

    conn.close()


def test_foreign_key_enforcement_on_insert_jobs_history(db_conn):
    """
    Test that inserting into job_status_history fails if the job_id does not exist in jobs.
    """
    with pytest.raises(DatabaseError) as excinfo:
        execute_query(
            "INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("non_existent_job_id", "running", 50.0, "Processing...", "2023-01-01T12:00:00"),
            connection=db_conn
        )
    assert "foreign key constraint failed" in str(excinfo.value.original_error).lower()

def test_foreign_key_enforcement_on_insert_training_results_model(db_conn):
    """
    Test that inserting into training_results fails if the model_id does not exist in models.
    """
    # First, create a job and a parameter set, as training_results depends on them too
    job_id = "test_job_fk_model"
    execute_query(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "training", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00"),
        connection=db_conn
    )

    config_id = "test_config_fk_model"
    execute_query(
        "INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)",
        (config_id, '{}', "2023-01-01T00:00:00"),
        connection=db_conn
    )

    with pytest.raises(DatabaseError) as excinfo:
        execute_query(
            "INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics, duration) VALUES (?, ?, ?, ?, ?, ?)",
            ("tr_res_1", job_id, "non_existent_model_id", config_id, "{}", 100),
            connection=db_conn
        )
    assert ("foreign key constraint failed" in str(excinfo.value.original_error).lower() or \
           "no such table: models" in str(excinfo.value.original_error).lower())

def test_foreign_key_enforcement_on_insert_training_results_config(db_conn):
    """
    Test that inserting into training_results fails if the config_id does not exist in configs.
    """
    job_id = "test_job_fk_config"
    execute_query(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "training", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00"),
        connection=db_conn
    )

    model_id = "test_model_fk_config"
    execute_query(
        "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
        (model_id, job_id, "/path/to/model", "2023-01-01T00:00:00"),
        connection=db_conn
    )

    with pytest.raises(DatabaseError) as excinfo:
        execute_query(
            "INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics, duration) VALUES (?, ?, ?, ?, ?, ?)",
            ("tr_res_2", job_id, model_id, "non_existent_config_id", "{}", 100),
            connection=db_conn
        )
    assert "foreign key constraint failed" in str(excinfo.value.original_error).lower()


def test_foreign_key_cascade_delete_jobs(db_conn):
    """
    Test that deleting a job cascades to delete related job_status_history entries.
    (Assuming ON DELETE CASCADE is set on the foreign key in schema - if not, this test would fail
     or need to be adapted to test restricted delete if that's the behavior).
    The current schema for job_status_history.job_id does NOT specify ON DELETE CASCADE.
    So, this test should verify that deleting a job with history entries is RESTRICTED.
    """
    job_id = "job_to_delete"

    # Create a job
    execute_query(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00"),
        connection=db_conn
    )

    # Create a history entry for this job
    execute_query(
        "INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "running", 50.0, "Processing...", "2023-01-01T12:00:00"),
        connection=db_conn
    )

    # Attempt to delete the job
    with pytest.raises(DatabaseError) as excinfo:
        execute_query(
            "DELETE FROM jobs WHERE job_id = ?",
            (job_id,),
            connection=db_conn
        )
    assert "foreign key constraint failed" in str(excinfo.value.original_error).lower()

    # Verify job and history entry still exist
    job_entry = execute_query("SELECT * FROM jobs WHERE job_id = ?", (job_id,), connection=db_conn)
    history_entry = execute_query("SELECT * FROM job_status_history WHERE job_id = ?", (job_id,), connection=db_conn, fetchall=True)

    assert job_entry is not None
    assert len(history_entry) == 1

def test_successful_insert_with_valid_foreign_keys(db_conn):
    """
    Test that inserts are successful when foreign keys are valid.
    """
    job_id = "valid_job_id"
    execute_query(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "test_type", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00"),
        connection=db_conn
    )

    # This should succeed
    execute_query(
        "INSERT INTO job_status_history (job_id, status, progress, status_message, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "completed", 100.0, "Done", "2023-01-01T13:00:00"),
        connection=db_conn
    )

    history_entry = execute_query("SELECT * FROM job_status_history WHERE job_id = ?", (job_id,), connection=db_conn, fetchall=True)
    assert len(history_entry) == 1
    assert history_entry[0]['status'] == "completed"
