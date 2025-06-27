"""
Tests for database functions related to requirements hash and job cloning functionality.

This module tests the database functions that support DataSphere job cloning optimization:
- Requirements hash management
- Compatible job finding
- Parent job references
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from deployment.app.db.schema import init_db
from deployment.app.db.database import (
    create_job,
    update_job_status,
    # Functions to be implemented:
    find_compatible_job,
    update_job_requirements_hash,
    update_job_parent_reference,
    get_job_requirements_hash
)
from deployment.app.models.api_models import JobType, JobStatus


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Initialize the database
        assert init_db(db_path), "Failed to initialize test database"
        yield db_path
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def db_connection(temp_db):
    """Provide a database connection for testing."""
    conn = sqlite3.connect(temp_db)
    # Use the same row factory as in database.py
    def dict_factory(cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
    conn.row_factory = dict_factory
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


class TestRequirementsHashFunctions:
    """Test requirements hash related database functions."""
    
    def test_update_job_requirements_hash_success(self, db_connection):
        """Test successful update of job requirements hash."""
        # Create a test job
        job_id = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        test_hash = "abc123def456"
        
        # Update requirements hash
        update_job_requirements_hash(job_id, test_hash, connection=db_connection)
        
        # Verify the hash was stored
        retrieved_hash = get_job_requirements_hash(job_id, connection=db_connection)
        assert retrieved_hash == test_hash
    
    def test_update_job_requirements_hash_nonexistent_job(self, db_connection):
        """Test updating requirements hash for non-existent job."""
        non_existent_job_id = "job_does_not_exist"
        test_hash = "abc123def456"
        
        # Should not raise exception but should handle gracefully
        update_job_requirements_hash(non_existent_job_id, test_hash, connection=db_connection)
        
        # Verify no hash is returned for non-existent job
        retrieved_hash = get_job_requirements_hash(non_existent_job_id, connection=db_connection)
        assert retrieved_hash is None
    
    def test_get_job_requirements_hash_existing_job(self, db_connection):
        """Test retrieving requirements hash for existing job."""
        # Create a test job and set hash
        job_id = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        test_hash = "def789ghi012"
        update_job_requirements_hash(job_id, test_hash, connection=db_connection)
        
        # Retrieve and verify
        retrieved_hash = get_job_requirements_hash(job_id, connection=db_connection)
        assert retrieved_hash == test_hash
    
    def test_get_job_requirements_hash_nonexistent_job(self, db_connection):
        """Test retrieving requirements hash for non-existent job."""
        non_existent_job_id = "job_does_not_exist"
        
        retrieved_hash = get_job_requirements_hash(non_existent_job_id, connection=db_connection)
        assert retrieved_hash is None


class TestParentJobReferences:
    """Test parent job reference functionality."""
    
    def test_update_job_parent_reference_success(self, db_connection):
        """Test successful update of job parent reference."""
        # Create parent and child jobs
        parent_job_id = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        child_job_id = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        test_hash = "parent_hash_123"
        
        # Update parent reference
        update_job_parent_reference(child_job_id, parent_job_id, test_hash, connection=db_connection)
        
        # Verify both hash and parent reference were set
        cursor = db_connection.cursor()
        cursor.execute(
            "SELECT requirements_hash, parent_job_id FROM jobs WHERE job_id = ?",
            (child_job_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row['requirements_hash'] == test_hash
        assert row['parent_job_id'] == parent_job_id
    
    def test_update_job_parent_reference_nonexistent_child(self, db_connection):
        """Test updating parent reference for non-existent child job."""
        parent_job_id = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        non_existent_child_id = "child_does_not_exist"
        test_hash = "parent_hash_123"
        
        # Should handle gracefully without raising exception
        update_job_parent_reference(non_existent_child_id, parent_job_id, test_hash, connection=db_connection)


class TestCompatibleJobFinding:
    """Test finding compatible jobs for cloning."""
    
    def setup_test_jobs(self, db_connection):
        """Set up test jobs with various states for compatibility testing."""
        base_time = datetime.now()
        test_hash = "compatible_hash_456"
        different_hash = "different_hash_789"
        
        jobs = {}
        
        # Job 1: Recent, completed, same hash - SHOULD be compatible
        jobs['compatible_recent'] = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        update_job_status(jobs['compatible_recent'], JobStatus.COMPLETED.value, connection=db_connection)
        update_job_requirements_hash(jobs['compatible_recent'], test_hash, connection=db_connection)
        
        # Job 2: Old, completed, same hash - should NOT be compatible (too old)
        jobs['compatible_old'] = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        update_job_status(jobs['compatible_old'], JobStatus.COMPLETED.value, connection=db_connection)
        update_job_requirements_hash(jobs['compatible_old'], test_hash, connection=db_connection)
        # Simulate old creation time by directly updating the database
        cursor = db_connection.cursor()
        old_time = (base_time - timedelta(days=35)).isoformat()
        cursor.execute(
            "UPDATE jobs SET created_at = ? WHERE job_id = ?",
            (old_time, jobs['compatible_old'])
        )
        db_connection.commit()
        
        # Job 3: Recent, failed, same hash - should NOT be compatible (failed status)
        jobs['failed_recent'] = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        update_job_status(jobs['failed_recent'], JobStatus.FAILED.value, connection=db_connection)
        update_job_requirements_hash(jobs['failed_recent'], test_hash, connection=db_connection)
        
        # Job 4: Recent, completed, different hash - should NOT be compatible (different hash)
        jobs['different_hash'] = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        update_job_status(jobs['different_hash'], JobStatus.COMPLETED.value, connection=db_connection)
        update_job_requirements_hash(jobs['different_hash'], different_hash, connection=db_connection)
        
        # Job 5: Recent, running, same hash - should NOT be compatible (not completed)
        jobs['running_recent'] = create_job(JobType.TRAINING, parameters={}, connection=db_connection)
        update_job_status(jobs['running_recent'], JobStatus.RUNNING.value, connection=db_connection)
        update_job_requirements_hash(jobs['running_recent'], test_hash, connection=db_connection)
        
        return jobs, test_hash
    
    def test_find_compatible_job_success(self, db_connection):
        """Test finding a compatible job successfully."""
        jobs, test_hash = self.setup_test_jobs(db_connection)
        
        # Should find the compatible recent job
        compatible_job_id = find_compatible_job(
            test_hash, 
            JobType.TRAINING.value,
            max_age_days=30,
            connection=db_connection
        )
        
        assert compatible_job_id == jobs['compatible_recent']
    
    def test_find_compatible_job_no_match(self, db_connection):
        """Test finding compatible job when none exists."""
        jobs, _ = self.setup_test_jobs(db_connection)
        non_existent_hash = "hash_that_does_not_exist"
        
        # Should return None
        compatible_job_id = find_compatible_job(
            non_existent_hash,
            JobType.TRAINING.value,
            max_age_days=30,
            connection=db_connection
        )
        
        assert compatible_job_id is None
    
    def test_find_compatible_job_different_job_type(self, db_connection):
        """Test finding compatible job with different job type."""
        jobs, test_hash = self.setup_test_jobs(db_connection)
        
        # Should return None because job type doesn't match
        compatible_job_id = find_compatible_job(
            test_hash,
            JobType.PREDICTION.value,  # Different job type
            max_age_days=30,
            connection=db_connection
        )
        
        assert compatible_job_id is None
    
    def test_find_compatible_job_strict_age_limit(self, db_connection):
        """Test finding compatible job with strict age limit."""
        jobs, test_hash = self.setup_test_jobs(db_connection)
        
        # With very strict age limit, should not find old jobs
        compatible_job_id = find_compatible_job(
            test_hash,
            JobType.TRAINING.value,
            max_age_days=1,  # Very strict
            connection=db_connection
        )
        
        # Should still find the recent compatible job
        assert compatible_job_id == jobs['compatible_recent']


class TestDatabaseSchemaUpdates:
    """Test that the database schema includes the new columns."""
    
    def test_jobs_table_has_new_columns(self, db_connection):
        """Test that jobs table includes requirements_hash and parent_job_id columns."""
        cursor = db_connection.cursor()
        cursor.execute("PRAGMA table_info(jobs)")
        columns = cursor.fetchall()
        
        # For dict_factory, use column names as keys
        column_names = [col['name'] for col in columns]
        assert 'requirements_hash' in column_names, "requirements_hash column missing from jobs table"
        assert 'parent_job_id' in column_names, "parent_job_id column missing from jobs table"
    
    def test_jobs_table_indexes_exist(self, db_connection):
        """Test that the new indexes exist for the jobs table."""
        cursor = db_connection.cursor()
        cursor.execute("PRAGMA index_list(jobs)")
        indexes = cursor.fetchall()
        
        # For dict_factory, use index names as keys
        index_names = [idx['name'] for idx in indexes]
        assert 'idx_jobs_requirements_hash' in index_names, "requirements_hash index missing"
        assert 'idx_jobs_parent_job_id' in index_names, "parent_job_id index missing" 