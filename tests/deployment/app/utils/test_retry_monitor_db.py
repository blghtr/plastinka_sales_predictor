import os
import tempfile

import sqlite3

import pytest

from deployment.app.db.schema import init_db
from deployment.app.utils.retry_monitor import retry_monitor, get_retry_statistics
from deployment.app.db.database import fetch_recent_retry_events


def test_retry_monitor_persists_to_db(tmp_path):
    """Record a retry event and verify it is stored in SQLite and aggregated."""
    # Prepare isolated DB file
    db_file = tmp_path / "retry_test.db"

    # Initialise schema in fresh DB
    assert init_db(db_path=str(db_file)) is True

    # Point global monitor to this DB
    retry_monitor._db_path = str(db_file)

    # Record a retry event
    retry_monitor.record_retry(
        operation="sample_op",
        exception_type="ValueError",
        exception_message="test",
        attempt=1,
        max_attempts=3,
        successful=False,
        component="test_component",
        duration_ms=5,
    )

    # Fetch directly from DB
    events = fetch_recent_retry_events(limit=10, connection=sqlite3.connect(str(db_file)))
    assert len(events) == 1
    ev = events[0]
    assert ev["operation"] == "sample_op"
    assert ev["component"] == "test_component"

    # Aggregated stats should include event
    stats = get_retry_statistics()
    assert stats["total_retries"] >= 1
    assert stats["operation_stats"]["test_component.sample_op"]["count"] >= 1 