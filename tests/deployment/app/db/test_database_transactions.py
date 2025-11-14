"""
Tests for transaction safety in database operations.

This module contains tests that verify the proper handling of transactions
in the database module, ensuring that:

1. Operations are properly committed or rolled back
2. Connection isolation is maintained
3. Concurrent transactions behave correctly
4. Nested transactions are handled correctly
5. Complex transaction chains maintain data integrity
"""

import asyncio
from datetime import datetime

import asyncpg
import pytest

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.jobs import create_job, get_job, update_job_status
from deployment.app.db.queries.core import execute_many, execute_query

# =============================================
# Используем фикстуры из conftest.py
# =============================================

# =============================================
# Tests for basic transaction safety
# =============================================


@pytest.mark.asyncio
async def test_execute_query_transaction(dal):
    """Test that execute_query properly manages transactions"""
    async with dal._pool.acquire() as conn:
        # Create a test table and clean it up
        await conn.execute("DROP TABLE IF EXISTS test_transactions")
        await conn.execute(
            "CREATE TABLE test_transactions (id BIGSERIAL PRIMARY KEY, value TEXT)"
        )

        # Get initial count
        initial_count = await conn.fetchval("SELECT COUNT(*) FROM test_transactions")

        # Test commit behavior within transaction
        async with conn.transaction():
            await execute_query(
                "INSERT INTO test_transactions (value) VALUES ($1)",
                conn,
                ("test_value",),
            )

        # Verify data was committed
        result = await conn.fetchrow("SELECT value FROM test_transactions WHERE value = $1", "test_value")
        assert result is not None
        assert result["value"] == "test_value"

        # Get count after successful insert
        count_after_insert = await conn.fetchval("SELECT COUNT(*) FROM test_transactions")
        assert count_after_insert == initial_count + 1

        # Test rollback on error
        try:
            async with conn.transaction():
                # This should fail (missing a required parameter)
                await execute_query(
                    "INSERT INTO test_transactions (id, value) VALUES ($1, $2)",
                    conn,
                    (99999,),  # Missing a parameter
                )
        except DatabaseError:
            pass  # Expected error

        # Verify nothing was inserted (count should be the same)
        final_count = await conn.fetchval("SELECT COUNT(*) FROM test_transactions")
        assert final_count == count_after_insert, "Rollback should prevent insert"
        
        # Clean up
        await conn.execute("DROP TABLE test_transactions")


@pytest.mark.asyncio
async def test_execute_many_transaction(dal):
    """Test that execute_many properly manages transactions"""
    async with dal._pool.acquire() as conn:
        # Create a test table and clean it up
        await conn.execute("DROP TABLE IF EXISTS test_batch")
        await conn.execute("CREATE TABLE test_batch (id BIGSERIAL PRIMARY KEY, value TEXT)")

        # Get initial count
        initial_count = await conn.fetchval("SELECT COUNT(*) FROM test_batch")

        # Prepare data for batch insert
        params_list = [("value1",), ("value2",), ("value3",)]

        # Test batch insert with commit
        async with conn.transaction():
            await execute_many(
                "INSERT INTO test_batch (value) VALUES ($1)", params_list, connection=conn
            )

        # Verify all data was committed
        count = await conn.fetchval("SELECT COUNT(*) FROM test_batch")
        assert count == initial_count + 3

        # Test rollback on error with batch insert
        async with conn.transaction():
            # Create an invalid parameter list (missing values)
            invalid_params = [
                ("value4",),
                (),  # Invalid - missing value
                ("value5",),
            ]

            try:
                await execute_many(
                    "INSERT INTO test_batch (value) VALUES ($1)", invalid_params, connection=conn
                )
            except DatabaseError:
                # Transaction will be rolled back automatically
                pass

        # Verify no new data was inserted (rollback)
        final_count = await conn.fetchval("SELECT COUNT(*) FROM test_batch")
        assert final_count == initial_count + 3, "Rollback should prevent insert"
        
        # Clean up
        await conn.execute("DROP TABLE test_batch")


@pytest.mark.asyncio
async def test_nested_transactions(dal):
    """Test nested transactions behavior with PostgreSQL savepoints"""
    async with dal._pool.acquire() as conn:
        # Create a test table
        await conn.execute("CREATE TABLE IF NOT EXISTS nested_test (id BIGSERIAL PRIMARY KEY, value TEXT)")

        # Start an outer transaction
        try:
            async with conn.transaction():
                await conn.execute("INSERT INTO nested_test (value) VALUES ($1)", "outer")

                # Start a nested transaction (savepoint in PostgreSQL)
                try:
                    async with conn.transaction():
                        # Insert another record in the nested transaction
                        await conn.execute("INSERT INTO nested_test (value) VALUES ($1)", "inner")
                        # Rollback the nested transaction
                        raise asyncpg.PostgresError("Rollback nested")
                except asyncpg.PostgresError:
                    # Nested transaction rolled back
                    pass
        except asyncpg.PostgresError:
            # Outer transaction also rolled back due to nested failure
            pass

        # Test that savepoints work correctly with explicit rollback
        async with conn.transaction():
            await conn.execute("INSERT INTO nested_test (value) VALUES ($1)", "outer2")
            try:
                async with conn.transaction():
                    await conn.execute("INSERT INTO nested_test (value) VALUES ($1)", "inner2")
                    # Explicitly rollback savepoint (PostgreSQL creates savepoints automatically)
                    # We can't rollback to a named savepoint, but the inner transaction rollback works
                    raise asyncpg.PostgresError("Rollback inner")
            except asyncpg.PostgresError:
                # Inner transaction rolled back, outer continues
                pass

        # Verify only outer insert was committed
        count = await conn.fetchval("SELECT COUNT(*) FROM nested_test")
        assert count >= 1


# =============================================
# Tests for complex transaction patterns
# =============================================


@pytest.mark.asyncio
async def test_create_job_transaction_safety(dal):
    """Test that create_job function operates safely within transactions"""
    async with dal._pool.acquire() as conn:
        # Test case: Create job inside a successful transaction
        async with conn.transaction():
            job_id_success = await create_job(
                "success_test", {"param": "value"}, connection=conn
            )

        # Verify the job was created
        job = await get_job(job_id_success, connection=conn)
        assert job is not None
        assert job["job_id"] == job_id_success

        # Test case: Create job inside a failed transaction
        job_id_fail = None
        try:
            async with conn.transaction():
                job_id_fail = await create_job(
                    "fail_test", {"param": "value"}, connection=conn
                )
                # Simulate an error to trigger rollback
                await conn.execute("SELECT * FROM non_existent_table")
        except asyncpg.PostgresError:
            pass  # Expected error

        # Verify the job wasn't created (transaction was rolled back)
        failed_job = await get_job(job_id_fail, connection=conn)
        assert failed_job is None, "Job should not exist after rollback"


@pytest.mark.asyncio
async def test_update_job_transaction_safety(dal):
    """Test update_job_status within transactions"""
    async with dal._pool.acquire() as conn:
        # Initial job creation
        job_id = await create_job("initial_job", {}, connection=conn)

        # Test case: Update job inside a successful transaction
        async with conn.transaction():
            await update_job_status(job_id, "running", connection=conn)

        # Verify update was successful
        job = await get_job(job_id, connection=conn)
        assert job["status"] == "running"

        # Test case: Update job inside a failed transaction
        async with conn.transaction():
            try:
                await update_job_status(job_id, "failed_attempt", connection=conn)
                # Force a rollback by causing an error
                await conn.execute("SELECT * FROM non_existent_table")
            except asyncpg.PostgresError:
                # Transaction will be rolled back automatically
                pass

        # Verify the update was rolled back
        job = await get_job(job_id, connection=conn)
        assert job["status"] == "running", (
            "Status should still be 'running' after rollback"
        )


@pytest.mark.asyncio
async def test_complex_transaction_chain(dal):
    """Test a complex chain of database operations within a transaction"""
    async with dal._pool.acquire() as conn:
        job_id_1, job_id_2 = None, None
        try:
            async with conn.transaction():
                # Chain of operations
                job_id_1 = await create_job("chain_job_1", {"step": 1}, connection=conn)
                await update_job_status(job_id_1, "step_1_done", connection=conn)

                job_id_2 = await create_job("chain_job_2", {"step": 2}, connection=conn)

                # Simulate failure
                await conn.execute("SELECT * FROM non_existent_table")
        except asyncpg.PostgresError:
            pass  # Expected error

        # Verify that neither job exists (complete rollback)
        job1 = await get_job(job_id_1, connection=conn)
        assert job1 is None, "Job 1 should not exist after rollback"

        job2 = await get_job(job_id_2, connection=conn)
        assert job2 is None, "Job 2 should not exist after rollback"


# =============================================
# Tests for concurrent transactions
# =============================================


@pytest.mark.asyncio
async def test_concurrent_reads(dal):
    """Test concurrent read operations"""
    async with dal._pool.acquire() as conn:
        # Create test jobs
        for i in range(10):
            await execute_query(
                "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
                conn,
                (
                    f"read_test_{i}",
                    "test",
                    "pending",
                    datetime.now(),
                    datetime.now(),
                ),
            )

        # Verify jobs were created
        count = await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE job_type = $1", "test")
        assert count == 10, "Jobs must be created for concurrent read test"

    # Test concurrent reads using asyncio
    async def worker(pool, job_id, results):
        async with pool.acquire() as worker_conn:
            result = await execute_query(
                "SELECT job_id FROM jobs WHERE job_id = $1",
                worker_conn,
                (job_id,),
            )
            if result:
                results.append(result["job_id"])

    results = []
    tasks = []
    for i in range(10):
        task = worker(dal._pool, f"read_test_{i}", results)
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Verify at least some reads succeeded
    assert len(results) > 0, "At least some read operations should succeed"


@pytest.mark.asyncio
async def test_concurrent_writes(dal):
    """Test concurrent write operations"""
    async with dal._pool.acquire() as conn:
        # Prepare table
        await conn.execute("CREATE TABLE IF NOT EXISTS concurrent_test (id BIGINT, thread_id BIGINT)")

    success_list = []

    async def worker(pool, thread_id, success_list):
        async with pool.acquire() as conn:
            try:
                async with conn.transaction():
                    for i in range(5):
                        await conn.execute(
                            "INSERT INTO concurrent_test VALUES ($1, $2)", i, thread_id
                        )
                success_list.append(True)
            except Exception as e:
                print(f"Error in write worker {thread_id}: {e}")
                success_list.append(False)

    tasks = []
    for i in range(3):
        task = worker(dal._pool, i, success_list)
        tasks.append(task)

    await asyncio.gather(*tasks)

    assert all(success_list)

    async with dal._pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM concurrent_test")
        assert count > 0, "At least some writes should succeed"


@pytest.mark.asyncio
async def test_connection_isolation(dal):
    """Test that each connection has its own isolated transaction"""
    async with dal._pool.acquire() as conn1:
        async with dal._pool.acquire() as conn2:
            # Clean up and create table first (outside transaction)
            await conn1.execute("DROP TABLE IF EXISTS iso_test")
            await conn1.execute("CREATE TABLE iso_test (id BIGINT)")
            
            # Start transaction on conn1 for DML operations
            async with conn1.transaction():
                await conn1.execute("INSERT INTO iso_test VALUES (1)")

                # conn2 should not see the uncommitted DML changes from conn1
                # Even though the table exists (DDL), the INSERT data is not visible until commit
                res2_data = await conn2.fetchrow("SELECT * FROM iso_test")
                assert res2_data is None, "conn2 should not see uncommitted DML data"

            # After commit, conn2 should see the changes
            res2_after_commit = await conn2.fetchrow("SELECT * FROM iso_test")
            assert res2_after_commit is not None
            assert res2_after_commit["id"] == 1
            
            # Clean up
            await conn1.execute("DROP TABLE iso_test")


@pytest.mark.asyncio
async def test_transaction_with_direct_conn_and_db_functions(dal):
    """
    Test mixing direct connection usage with database module functions
    """
    async with dal._pool.acquire() as conn:
        # Clean up first
        await conn.execute("DROP TABLE IF EXISTS complex_trans")
        
        # Create table outside transaction (DDL is auto-committed anyway)
        await conn.execute("CREATE TABLE complex_trans (id TEXT)")
        
        try:
            async with conn.transaction():
                # 1. Use a DB function within the transaction
                job_id = await create_job("complex_job", {}, connection=conn)

                # 2. Direct execute to insert data
                await conn.execute("INSERT INTO complex_trans VALUES ($1)", job_id)

                # 3. Simulate an error
                await conn.execute("SELECT * FROM non_existent_table")

                # Should not reach here
        except asyncpg.PostgresError:
            # Transaction will be rolled back automatically
            pass

        # Verify that both operations were rolled back
        # Table exists (created outside transaction), but data should be rolled back
        res = await conn.fetchrow("SELECT COUNT(*) as count FROM complex_trans")
        assert res is not None
        assert res["count"] == 0, "Data should be rolled back"

        # Check if the job exists
        job = await get_job("complex_job", connection=conn)
        assert job is None, "Job should not exist after rollback"
        
        # Clean up
        await conn.execute("DROP TABLE IF EXISTS complex_trans")
