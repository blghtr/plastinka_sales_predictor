import pytest
import sqlite3
from deployment.app.db.database import execute_query, DatabaseError # Removed get_db_connection
from deployment.app.db.schema import SCHEMA_SQL # Import SCHEMA_SQL directly

# Use a dedicated in-memory database for these tests to avoid interference
TEST_DB_PATH = ":memory:"

@pytest.fixture(scope="function")
def db_conn():
    """Fixture to set up and tear down the in-memory database for each test."""
    # Directly connect to the in-memory database for the test
    conn = sqlite3.connect(TEST_DB_PATH)
    
    # Enable Foreign Key support
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # Define and set dict_factory for this connection
    def dict_factory_for_test(cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
    conn.row_factory = dict_factory_for_test

    # Initialize schema directly on this connection
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)
    conn.commit()
    
    # Verify PRAGMA is ON
    cursor.execute("PRAGMA foreign_keys;")
    fk_status = cursor.fetchone()
    assert fk_status['foreign_keys'] == 1, "Foreign keys should be ON for the test connection"
    
    yield conn
    
    conn.close()
    # No need to manipulate settings.db.path anymore as we are not calling get_db_connection


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
    
    param_set_id = "test_param_set_fk_model"
    execute_query(
        "INSERT INTO parameter_sets (parameter_set_id, parameters, created_at) VALUES (?, ?, ?)",
        (param_set_id, '{}', "2023-01-01T00:00:00"),
        connection=db_conn
    )

    with pytest.raises(DatabaseError) as excinfo:
        execute_query(
            "INSERT INTO training_results (result_id, job_id, model_id, parameter_set_id, metrics, duration) VALUES (?, ?, ?, ?, ?, ?)",
            ("tr_res_1", job_id, "non_existent_model_id", param_set_id, "{}", 100),
            connection=db_conn
        )
    assert "foreign key constraint failed" in str(excinfo.value.original_error).lower()

def test_foreign_key_enforcement_on_insert_training_results_param_set(db_conn):
    """
    Test that inserting into training_results fails if the parameter_set_id does not exist in parameter_sets.
    """
    job_id = "test_job_fk_param"
    execute_query(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "training", "pending", "2023-01-01T00:00:00", "2023-01-01T00:00:00"),
        connection=db_conn
    )
    
    model_id = "test_model_fk_param"
    execute_query(
        "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
        (model_id, job_id, "/path/to/model", "2023-01-01T00:00:00"),
        connection=db_conn
    )

    with pytest.raises(DatabaseError) as excinfo:
        execute_query(
            "INSERT INTO training_results (result_id, job_id, model_id, parameter_set_id, metrics, duration) VALUES (?, ?, ?, ?, ?, ?)",
            ("tr_res_2", job_id, model_id, "non_existent_param_set_id", "{}", 100),
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