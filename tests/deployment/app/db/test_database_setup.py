import sqlite3

import pytest

# @pytest.fixture
# def temp_db():
#     """Create a temporary SQLite database with the correct schema."""
#     conn = None
#     tmp = None
#     try:
#         tmp = tempfile.NamedTemporaryFile(delete=False)
#         # Create the database file
#         conn = sqlite3.connect(tmp.name)
#         # Create the schema
#         conn.executescript("""
#         CREATE TABLE jobs (
#             job_id TEXT PRIMARY KEY,
#             created_at TIMESTAMP,
#             updated_at TIMESTAMP,
#             status TEXT,
#             progress INTEGER,
#             status_message TEXT,
#             error_message TEXT
#         );

#         CREATE TABLE job_status_history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             job_id TEXT,
#             status TEXT,
#             progress INTEGER,
#             status_message TEXT,
#             error_message TEXT,
#             created_at TIMESTAMP,
#             FOREIGN KEY (job_id) REFERENCES jobs(job_id)
#         );
#         """)
#         conn.commit()
#         conn.close()
#         conn = None  # Set to None to indicate it's closed

#         yield tmp.name
#     finally:
#         # Make sure connection is closed
#         if conn:
#             try:
#                 conn.close()
#             except Exception:
#                 pass

#         # Clean up the file
#         if tmp:
#             try:
#                 # Close the file handle first
#                 tmp.close()
#                 # Wait a moment to ensure the file handle is released
#                 time.sleep(0.1)
#                 # Now try to delete the file
#                 Path(tmp.name).unlink(missing_ok=True)
#             except PermissionError:
#                 print(
#                     f"Warning: Could not delete temporary file {tmp.name}, it may still be in use"
#                 )
#             except Exception as e:
#                 print(f"Warning: Error cleaning up temporary file: {e}")


def test_foreign_key_constraint(temp_db):
    """Test that foreign key constraints are enforced."""
    dal = temp_db["dal"]
    conn = dal._connection
    # Connect to the database
    conn.execute("PRAGMA foreign_keys = ON")

    # Insert a job with job_type
    conn.execute(
        "INSERT INTO jobs (job_id, job_type, created_at, updated_at, status) VALUES (?, ?, ?, ?, ?)",
        ("test_job", "training", "2021-01-01", "2021-01-01", "running"),
    )

    # Insert a job status history entry with a valid job_id (только updated_at)
    conn.execute(
        "INSERT INTO job_status_history (job_id, status, progress, updated_at) VALUES (?, ?, ?, ?)",
        ("test_job", "running", 50, "2021-01-01"),
    )

    # Try to insert a job status history entry with an invalid job_id
    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        conn.execute(
            "INSERT INTO job_status_history (job_id, status, progress, updated_at) VALUES (?, ?, ?, ?)",
            ("nonexistent_job", "running", 50, "2021-01-01"),
        )

    # Check that the error message mentions foreign key constraint
    assert "FOREIGN KEY constraint failed" in str(excinfo.value)

    # Clean up
    conn.close()
