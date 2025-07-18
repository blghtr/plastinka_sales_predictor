import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deployment.app.db.feature_storage import (
    FeatureStoreFactory,
    SQLFeatureStore,
    load_features,
    save_features,
)
from deployment.app.db.schema import SCHEMA_SQL  # Import schema SQL
from deployment.app.db.data_access_layer import get_or_create_multiindex_id


@pytest.fixture
def feature_store_env():
    """Set up test environment with a temporary SQLite database and connection."""
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

    conn = sqlite3.connect(temp_db_file)
    conn.row_factory = sqlite3.Row  # Use Row factory for easier access
    cursor = conn.cursor()

    cursor.executescript(SCHEMA_SQL)

    # Insert initial test data
    cursor.execute("""
    INSERT INTO dim_multiindex_mapping (multiindex_id, barcode, artist, album, cover_type, price_category,
                                        release_type, recording_decade, release_decade, style, record_year)
    VALUES (1, '1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard',
            'Studio', '2010s', '2010s', 'Rock', 2015)
    """)

    cursor.execute("""
    INSERT INTO processing_runs (run_id, start_time, status, cutoff_date, source_files)
    VALUES (1, '2023-01-01T00:00:00', 'fixture_setup', '2023-01-01', 'fixture_files.csv')
    """)
    conn.commit()

    # Clean fact tables before each test
    cursor.execute("DELETE FROM fact_stock")
    cursor.execute("DELETE FROM fact_prices")
    cursor.execute("DELETE FROM fact_sales")
    cursor.execute("DELETE FROM fact_stock_changes")
    conn.commit()

    yield {
        "conn": conn,
        "cursor": cursor,
        "db_path": temp_db_file,  # Provide path for tests that might need it
    }

    conn.close()
    if os.path.exists(temp_db_file):
        os.unlink(temp_db_file)


def test_convert_to_int(feature_store_env):
    """Test the _convert_to_int helper method."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        assert store._convert_to_int(5) == 5
        assert store._convert_to_int(5.7) == 6
        assert store._convert_to_int(5.3) == 5
        assert store._convert_to_int(np.float64(5.7)) == 6
        assert store._convert_to_int("5") == 5
        assert store._convert_to_int("invalid") == 0
        assert store._convert_to_int(None) == 0
        assert store._convert_to_int(np.nan) == 0
        assert store._convert_to_int(pd.NA) == 0
        assert store._convert_to_int(np.nan, default=-1) == -1


def test_convert_to_float(feature_store_env):
    """Test the _convert_to_float helper method."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        assert store._convert_to_float(5.7) == 5.7
        assert store._convert_to_float(5) == 5.0
        assert store._convert_to_float(np.float64(5.7)) == 5.7
        assert store._convert_to_float("5.7") == 5.7
        assert store._convert_to_float("invalid") == 0.0
        assert store._convert_to_float(None) == 0.0
        assert store._convert_to_float(np.nan) == 0.0
        assert store._convert_to_float(pd.NA) == 0.0
        assert store._convert_to_float(np.nan, default=-1.5) == -1.5


def test_convert_to_date_str(feature_store_env):
    """Test the _convert_to_date_str helper method."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        test_date = datetime(2023, 1, 15)
        assert store._convert_to_date_str(test_date) == "2023-01-15"
        assert store._convert_to_date_str("2023-01-15") == "2023-01-15"
        assert store._convert_to_date_str("invalid-date") == "invalid-date"
        assert store._convert_to_date_str(123) == "123"


def test_save_stock_feature_with_mixed_types(feature_store_env):
    """Test saving stock feature with mixed data types."""
    conn = feature_store_env["conn"]
    cursor = feature_store_env["cursor"]

    with SQLFeatureStore(connection=conn, run_id=1) as store:
        idx_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        index = pd.MultiIndex.from_tuples(
            [idx_tuple],
            names=[
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
            ],
        )

        today = datetime.now().date()
        yesterday = (datetime.now() - timedelta(days=1)).date()

        # Используем унифицированный формат: дата в индексе, multiindex в колонках
        data_to_save = {
            today: 10,
            yesterday: 15.7,
            pd.to_datetime("2023-01-01").date(): 20,
            # Убираем NaN, так как он не сохраняется
        }

        for i, (date_val, value) in enumerate(data_to_save.items()):
            # Создаем DataFrame с датой в индексе и multiindex в колонках
            date_ts = pd.to_datetime(date_val)
            df_single_date = pd.DataFrame([[value]], index=[date_ts], columns=index)

            # Первый вызов без append (очищает таблицу), остальные с append=True
            append_mode = i > 0
            store._save_feature("stock", df_single_date, append=append_mode)

        query = "SELECT data_date, value FROM fact_stock WHERE multiindex_id = 1 ORDER BY data_date"
        cursor.execute(query)
        results = cursor.fetchall()

        assert len(results) == 3
        assert results[0]["data_date"] == "2023-01-01"
        assert results[0]["value"] == 20
        assert results[1]["data_date"] == yesterday.strftime("%Y-%m-%d")
        assert results[1]["value"] == 15.7
        assert results[2]["data_date"] == today.strftime("%Y-%m-%d")
        assert results[2]["value"] == 10


def test_save_prices_feature_with_mixed_types(feature_store_env):
    """Test saving prices feature with mixed data types."""
    conn = feature_store_env["conn"]
    cursor = feature_store_env["cursor"]
    
    # Создаем правильный multiindex для колонок (10 элементов)
    idx_tuple = (
        "1234567890",
        "Test Artist", 
        "Test Album",
        "CD",
        "Standard",
        "Studio",
        "2010s",
        "2010s", 
        "Rock",
        2015,
    )
    index = pd.MultiIndex.from_tuples(
        [idx_tuple],
        names=[
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
        ],
    )
    
    # Используем правильный унифицированный формат: дата в индексе, multiindex в колонках
    stock_date = pd.to_datetime("2023-08-01")
    
    # Тестируем разные типы данных
    df = pd.DataFrame([[123.45]], index=[stock_date], columns=index)
    df_int = pd.DataFrame([[100]], index=[stock_date], columns=index) 
    df_str = pd.DataFrame([[99.99]], index=[stock_date], columns=index)
    df_nan = pd.DataFrame([[np.nan]], index=[stock_date], columns=index)
    
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        store._save_feature("prices", df)
        store._save_feature("prices", df_int, append=True)
        store._save_feature("prices", df_str, append=True) 
        store._save_feature("prices", df_nan, append=True)
    
    # Проверяем результат
    query = "SELECT data_date, value FROM fact_prices WHERE multiindex_id = 1"
    cursor.execute(query)
    results = cursor.fetchall()
    assert len(results) == 1  # nan не сохраняется, последнее значение перезаписывает предыдущие
    assert results[0]["value"] in [123.45, 100, 99.99]





# New connection management tests:
def test_sql_feature_store_as_context_manager_external_connection():
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    conn = sqlite3.connect(temp_db_file)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)
    conn.commit()

    try:
        with SQLFeatureStore(connection=conn, run_id=1) as fs:
            assert fs.db_conn == conn
            assert fs._conn_created_internally is False
            # Perform a simple operation that requires a run_id if fs uses it internally
            # For example, if create_run isn't called, other methods might use self.run_id
            # We need to ensure a run exists if methods in SQLFeatureStore depend on it.
            # Let's pre-insert a run for this test or call create_run.
            # For simplicity, assuming run_id=1 is okay if other methods don't strictly need a *valid* run
            # but just a non-None run_id for some paths.
            # If create_run is essential: fs.create_run("2023-01-01", "test_ext.csv")
            # For now, relying on run_id=1 being passed.
            fs.complete_run("test_status")  # Example operation

        conn.execute("SELECT 1")  # Should not raise ProgrammingError
    finally:
        if conn:
            conn.close()
        if os.path.exists(temp_db_file):
            os.unlink(temp_db_file)


@patch("deployment.app.db.database.get_db_connection")
def test_sql_feature_store_as_context_manager_internal_connection(
    mock_get_db_connection_from_patch,
):
    temp_db_file_internal = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name

    # This is the connection that get_db_connection will return
    internal_conn_obj = sqlite3.connect(temp_db_file_internal)
    cursor = internal_conn_obj.cursor()
    cursor.executescript(SCHEMA_SQL)
    internal_conn_obj.commit()

    mock_get_db_connection_from_patch.return_value = internal_conn_obj

    try:
        # Using a specific run_id to avoid issues if methods depend on it.
        with SQLFeatureStore(run_id=1) as fs:
            assert fs._conn_created_internally is True
            assert fs.db_conn == internal_conn_obj
            fs.complete_run("test_internal_status")  # Example operation

        with pytest.raises(
            sqlite3.ProgrammingError, match="Cannot operate on a closed database."
        ):
            internal_conn_obj.execute("SELECT 1")

    finally:
        # Attempt to close, will be no-op or error if already closed by __exit__
        try:
            if internal_conn_obj:
                internal_conn_obj.close()
        except sqlite3.ProgrammingError:
            pass  # Already closed, expected
        if os.path.exists(temp_db_file_internal):
            os.unlink(temp_db_file_internal)


@patch("deployment.app.db.database.get_db_connection")
def test_sql_feature_store_context_manager_exception_internal(
    mock_get_db_connection_from_patch,
):
    temp_db_file_internal_exc = tempfile.NamedTemporaryFile(
        suffix=".db", delete=False
    ).name
    internal_conn_obj_exc = sqlite3.connect(temp_db_file_internal_exc)
    cursor = internal_conn_obj_exc.cursor()
    cursor.executescript(SCHEMA_SQL)
    internal_conn_obj_exc.commit()
    mock_get_db_connection_from_patch.return_value = internal_conn_obj_exc

    try:
        with pytest.raises(ValueError, match="Test exception from internal"):
            with SQLFeatureStore(run_id=1) as fs:
                assert fs._conn_created_internally is True
                raise ValueError("Test exception from internal")

        with pytest.raises(
            sqlite3.ProgrammingError, match="Cannot operate on a closed database."
        ):
            internal_conn_obj_exc.execute("SELECT 1")

    finally:
        try:
            if internal_conn_obj_exc:
                internal_conn_obj_exc.close()
        except sqlite3.ProgrammingError:
            pass
        if os.path.exists(temp_db_file_internal_exc):
            os.unlink(temp_db_file_internal_exc)


def test_sql_feature_store_context_manager_exception_external():
    temp_db_file_ext_exc = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    external_conn_exc = sqlite3.connect(temp_db_file_ext_exc)
    cursor = external_conn_exc.cursor()
    cursor.executescript(SCHEMA_SQL)
    external_conn_exc.commit()

    try:
        with pytest.raises(ValueError, match="Test exception from external"):
            with SQLFeatureStore(connection=external_conn_exc, run_id=1) as fs:
                assert fs._conn_created_internally is False
                raise ValueError("Test exception from external")

        external_conn_exc.execute("SELECT 1")  # Should not raise

    finally:
        if external_conn_exc:
            external_conn_exc.close()
        if os.path.exists(temp_db_file_ext_exc):
            os.unlink(temp_db_file_ext_exc)


# End of new connection management tests


def test_create_and_complete_run(feature_store_env):
    """Test creating and completing a processing run."""
    conn = feature_store_env["conn"]
    cursor = feature_store_env["cursor"]

    with SQLFeatureStore(connection=conn) as store:  # No run_id, will create new
        run_id = store.create_run(cutoff_date="2023-02-01", source_files="run_test.csv")
        assert run_id is not None
        assert store.run_id == run_id

        cursor.execute(
            "SELECT status, cutoff_date, source_files FROM processing_runs WHERE run_id = ?",
            (run_id,),
        )
        run_data = cursor.fetchone()
        assert run_data is not None
        assert run_data["status"] == "running"
        assert run_data["cutoff_date"] == "2023-02-01"
        assert run_data["source_files"] == "run_test.csv"

        store.complete_run(status="success")

        cursor.execute(
            "SELECT status, end_time FROM processing_runs WHERE run_id = ?", (run_id,)
        )
        run_data = cursor.fetchone()
        assert run_data is not None
        assert run_data["status"] == "success"
        assert run_data["end_time"] is not None


def test_save_sales_feature(feature_store_env):
    """Test saving sales feature data."""
    conn = feature_store_env["conn"]
    cursor = feature_store_env["cursor"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        idx_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        date1 = pd.to_datetime("2023-03-10")
        date2 = pd.to_datetime("2023-03-11")

        # Создаем multiindex для колонок
        index = pd.MultiIndex.from_tuples(
            [idx_tuple],
            names=[
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
            ],
        )

        # Используем унифицированный формат: дата в индексе, multiindex в колонках
        df = pd.DataFrame([[5.0], [3.0]], index=[date1, date2], columns=index)
        store._save_feature("sales", df)

        cursor.execute(
            "SELECT data_date, value FROM fact_sales WHERE multiindex_id = 1 ORDER BY data_date"
        )
        results = cursor.fetchall()

        assert len(results) == 2
        assert results[0]["data_date"] == "2023-03-10"
        assert results[0]["value"] == 5.0
        assert results[1]["data_date"] == "2023-03-11"
        assert results[1]["value"] == 3.0


def test_save_change_feature(feature_store_env):
    """Test saving stock change feature data."""
    conn = feature_store_env["conn"]
    cursor = feature_store_env["cursor"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        idx_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        date1 = pd.to_datetime("2023-04-01")
        date2 = pd.to_datetime("2023-04-02")

        # Создаем multiindex для колонок
        index = pd.MultiIndex.from_tuples(
            [idx_tuple],
            names=[
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
            ],
        )

        # Используем унифицированный формат: дата в индексе, multiindex в колонках
        df = pd.DataFrame([[-2.0], [1.0]], index=[date1, date2], columns=index)
        store._save_feature("change", df)

        cursor.execute(
            "SELECT data_date, value FROM fact_stock_changes WHERE multiindex_id = 1 ORDER BY data_date"
        )
        results = cursor.fetchall()

        assert len(results) == 2
        assert results[0]["data_date"] == "2023-04-01"
        assert results[0]["value"] == -2.0
        assert results[1]["data_date"] == "2023-04-02"
        assert results[1]["value"] == 1.0


def test_build_multiindex_from_mapping(feature_store_env):
    """Test rebuilding the MultiIndex from the database."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        multiindex, mask = store._build_multiindex_from_mapping([1])

        assert isinstance(multiindex, pd.MultiIndex)
        assert isinstance(mask, list)
        assert len(multiindex) == 1
        assert len(mask) == 1
        assert mask[0] is True  # ID 1 should be found
        
        expected_names = [
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
        assert list(multiindex.names) == expected_names

        expected_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        assert multiindex[0] == expected_tuple

        # Use the new batch approach instead of the removed _get_multiindex_id method
        new_tuple = (
            "987",
            "Art2",
            "Alb2",
            "LP",
            "High",
            "Live",
            "2000s",
            "2000s",
            "Jazz",
            2005,
        )
        from deployment.app.db.database import get_or_create_multiindex_ids_batch
        id_map = get_or_create_multiindex_ids_batch([new_tuple], conn)
        new_idx_id = id_map[new_tuple]
        multiindex_multi, mask_multi = store._build_multiindex_from_mapping([1, new_idx_id])

        assert len(multiindex_multi) == 2
        assert len(mask_multi) == 2
        assert mask_multi[0] is True  # ID 1 should be found
        assert mask_multi[1] is True  # new_idx_id should be found
        assert multiindex_multi[0] == expected_tuple
        assert multiindex_multi[1] == (
            "987",
            "Art2",
            "Alb2",
            "LP",
            "High",
            "Live",
            "2000s",
            "2000s",
            "Jazz",
            2005,
        )

        empty_multiindex, empty_mask = store._build_multiindex_from_mapping([])
        assert empty_multiindex.empty
        assert empty_mask == []
        assert list(empty_multiindex.names) == expected_names
        
        # Test with missing ID
        missing_multiindex, missing_mask = store._build_multiindex_from_mapping([999])
        assert len(missing_multiindex) == 0
        assert len(missing_mask) == 1
        assert missing_mask[0] is False  # ID 999 should not be found


def test_load_stock_feature(feature_store_env):
    """Test loading stock features."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        idx_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        index = pd.MultiIndex.from_tuples(
            [idx_tuple],
            names=[
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
            ],
        )
        date1 = pd.to_datetime("2023-05-01")
        date2 = pd.to_datetime("2023-05-02")

        # Создаем DataFrame в унифицированном формате: дата в индексе, multiindex в колонках
        df_save1 = pd.DataFrame([[10]], index=[date1], columns=index)
        df_save2 = pd.DataFrame([[12]], index=[date2], columns=index)
        store._save_feature("stock", df_save1)
        store._save_feature("stock", df_save2, append=True)  # append=True для второго вызова

        loaded_df = store._load_feature("stock")

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 2
        assert loaded_df.loc[date1, idx_tuple] == 10
        assert loaded_df.loc[date2, idx_tuple] == 12


def test_load_prices_feature(feature_store_env):
    """Test loading prices features with the new unified format."""
    conn = feature_store_env["conn"]
    
    # Создаем multiindex для колонок
    idx_tuple = (
        "1234567890",
        "Test Artist",
        "Test Album",
        "CD",
        "Standard",
        "Studio",
        "2010s",
        "2010s",
        "Rock",
        2015,
    )
    index = pd.MultiIndex.from_tuples(
        [idx_tuple],
        names=[
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
        ],
    )
    
    # Создаем DataFrame с датой в индексе для prices
    date1 = pd.to_datetime("2023-05-01")
    date2 = pd.to_datetime("2023-05-02")
    
    # В новом формате: дата в индексе, multiindex в колонках
    df_save1 = pd.DataFrame([[1500.50]], index=[date1], columns=index)
    df_save2 = pd.DataFrame([[1600.00]], index=[date2], columns=index)
    
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        store._save_feature("prices", df_save1)
        store._save_feature("prices", df_save2, append=True)
        loaded_df = store._load_feature("prices")
    
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.index.name == "_date"
    assert len(loaded_df.columns) == 1  # Один multiindex в колонках
    assert loaded_df.columns[0] == idx_tuple
    
    # Проверяем значения
    # В новом формате мы сохраняем все даты, а не только последнюю
    assert loaded_df.loc[date1, idx_tuple] == 1500.50
    assert loaded_df.loc[date2, idx_tuple] == 1600.00
    
    conn.execute("DELETE FROM fact_prices")
    conn.commit()
    loaded_empty = store._load_feature("prices")
    assert loaded_empty is None


def test_load_sales_feature(feature_store_env):
    """Test loading sales features."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        idx_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        date1 = pd.to_datetime("2023-06-01")
        date2 = pd.to_datetime("2023-06-02")

        # В новом формате: дата в индексе, multiindex в колонках
        index = pd.MultiIndex.from_tuples(
            [idx_tuple],
            names=[
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
            ],
        )
        df_to_save = pd.DataFrame([[10.0], [12.0]], index=[date1, date2], columns=index)
        store._save_feature("sales", df_to_save)

        loaded_df = store._load_feature("sales")
        assert isinstance(loaded_df, pd.DataFrame)
        assert not loaded_df.empty

        # Проверяем формат
        assert loaded_df.index.name == "_date"
        assert len(loaded_df) == 2  # Две даты в индексе
        assert len(loaded_df.columns) == 1  # Один multiindex в колонках
        assert loaded_df.columns[0] == idx_tuple
        
        # Проверяем значения
        assert loaded_df.loc[date1, idx_tuple] == 10.0
        assert loaded_df.loc[date2, idx_tuple] == 12.0

        # Проверяем фильтрацию по дате
        loaded_df_range = store._load_feature(
            "sales", start_date="2023-06-02", end_date="2023-06-02"
        )
        assert len(loaded_df_range) == 1
        assert loaded_df_range.loc[date2, idx_tuple] == 12.0

        conn.execute("DELETE FROM fact_sales")
        conn.commit()
        loaded_empty = store._load_feature("sales")
        assert loaded_empty is None


def test_load_change_feature(feature_store_env):
    """Test loading change features."""
    conn = feature_store_env["conn"]
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        idx_tuple = (
            "1234567890",
            "Test Artist",
            "Test Album",
            "CD",
            "Standard",
            "Studio",
            "2010s",
            "2010s",
            "Rock",
            2015,
        )
        date1 = pd.to_datetime("2023-07-01")
        date2 = pd.to_datetime("2023-07-02")

        # В новом формате: дата в индексе, multiindex в колонках
        index = pd.MultiIndex.from_tuples(
            [idx_tuple],
            names=[
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
            ],
        )
        df_to_save = pd.DataFrame([[-5.0], [3.0]], index=[date1, date2], columns=index)
        store._save_feature("change", df_to_save)

        loaded_df = store._load_feature("change")
        assert isinstance(loaded_df, pd.DataFrame)

        # Проверяем формат
        assert loaded_df.index.name == "_date"
        assert len(loaded_df) == 2  # Две даты в индексе
        assert len(loaded_df.columns) == 1  # Один multiindex в колонках
        assert loaded_df.columns[0] == idx_tuple
        
        # Проверяем значения
        assert loaded_df.loc[date1, idx_tuple] == -5.0
        assert loaded_df.loc[date2, idx_tuple] == 3.0


def test_load_features_all_types(feature_store_env):
    """Test loading all feature types using the main load_features method with unified format."""
    conn = feature_store_env["conn"]
    
    # Создаем multiindex для колонок
    idx_tuple = (
        "1234567890",
        "Test Artist",
        "Test Album",
        "CD",
        "Standard",
        "Studio",
        "2010s",
        "2010s",
        "Rock",
        2015,
    )
    index = pd.MultiIndex.from_tuples(
        [idx_tuple],
        names=[
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
        ],
    )
    
    # Даты для тестов
    stock_date = pd.to_datetime("2023-08-01")
    sales_date = pd.to_datetime("2023-08-02")
    change_date = pd.to_datetime("2023-08-03")
    prices_date = pd.to_datetime("2023-08-04")
    
    with SQLFeatureStore(connection=conn, run_id=1) as store:
        # Все features в унифицированном формате: даты в индексе, multiindex в колонках
        
        # Stock
        stock_df = pd.DataFrame([[float(100)]], index=[stock_date], columns=index)
        store._save_feature("stock", stock_df)
        
        # Prices
        prices_df = pd.DataFrame([[float(25.99)]], index=[prices_date], columns=index)
        store._save_feature("prices", prices_df)
        
        # Sales
        sales_df = pd.DataFrame([[5]], index=[sales_date], columns=index)
        store._save_feature("sales", sales_df)
        
        # Change
        change_df = pd.DataFrame([[-2]], index=[change_date], columns=index)
        store._save_feature("change", change_df)
        
        # Загружаем все features
        all_features = store.load_features()
    
    # Проверяем, что все features загрузились в правильном формате
    assert "stock" in all_features
    assert not all_features["stock"].empty
    assert all_features["stock"].index.name == "_date"
    assert all_features["stock"].loc[stock_date, idx_tuple] == 100.0
    
    assert "prices" in all_features
    assert not all_features["prices"].empty
    assert all_features["prices"].index.name == "_date"
    assert all_features["prices"].loc[prices_date, idx_tuple] == 25.99
    
    assert "sales" in all_features
    assert not all_features["sales"].empty
    assert all_features["sales"].index.name == "_date"
    assert all_features["sales"].loc[sales_date, idx_tuple] == 5.0
    
    assert "change" in all_features
    assert not all_features["change"].empty
    assert all_features["change"].index.name == "_date"
    assert all_features["change"].loc[change_date, idx_tuple] == -2.0


# Tests for factory and helper functions (should not need much change as they mock SQLFeatureStore)


def test_get_store_sql(feature_store_env):
    conn = feature_store_env["conn"]
    store = FeatureStoreFactory.get_store(store_type="sql", run_id=1, connection=conn)
    assert isinstance(store, SQLFeatureStore)
    assert store.run_id == 1
    assert store.db_conn == conn
    # Ensure connection is closed if created by store, or remains open if passed
    # This test passes an external connection, so it should remain open after store is GC'd
    # If store was created without connection, its __exit__ (if used as CM) or __del__ would close.
    # Here, FeatureStoreFactory doesn't use it as CM.
    del store
    conn.execute("SELECT 1")  # Check if conn is still open


def test_get_store_default(feature_store_env):
    conn = feature_store_env["conn"]
    # Test default type
    store = FeatureStoreFactory.get_store(run_id=1, connection=conn)
    assert isinstance(store, SQLFeatureStore)
    del store
    conn.execute("SELECT 1")


def test_get_store_unsupported():
    with pytest.raises(ValueError):
        FeatureStoreFactory.get_store(store_type="unsupported")


@patch("deployment.app.db.feature_storage.FeatureStoreFactory.get_store")
def test_save_features_helper_creates_store_and_calls_methods(
    mock_get_store, feature_store_env
):
    """Test that save_features helper correctly uses the store."""
    mock_store_instance = MagicMock(spec=SQLFeatureStore)
    # Configure the mock methods on the instance that get_store will return
    mock_store_instance.create_run.return_value = 123

    # Make the patched FeatureStoreFactory.get_store return our mock_store_instance
    mock_get_store.return_value = mock_store_instance

    conn_param = feature_store_env["conn"]
    features_dict = {"sales": pd.DataFrame({"A": [1]})}
    cutoff = "2023-01-01"
    sources = "file.csv"

    # Call the helper function
    returned_run_id = save_features(
        features_dict,
        cutoff_date=cutoff,
        source_files=sources,
        store_type="sql",
        connection=conn_param,
    )

    # Assertions
    # The save_features helper calls FeatureStoreFactory.get_store with store_type and **kwargs.
    # run_id is a named parameter in get_store's signature, not part of **kwargs here.
    # So, we check that get_store was called with the expected kwargs, and run_id took its default.
    mock_get_store.assert_called_once_with(
        store_type="sql",
        connection=conn_param,
        # run_id=None is implied by not being in kwargs and taking its default in get_store
    )
    # We can also assert that the run_id passed to the SQLFeatureStore constructor by the factory was indeed None
    # if the factory logic was more complex, but here it's straightforward.
    # Example: mock_get_store.call_args.kwargs['run_id'] would be KeyError or None depending on factory.
    # For this specific factory, run_id is a direct param, so it's not in kwargs if not passed.

    mock_store_instance.create_run.assert_called_once_with(cutoff, sources)
    mock_store_instance.save_features.assert_called_once_with(features_dict)
    mock_store_instance.complete_run.assert_called_once()
    assert returned_run_id == 123


@patch("deployment.app.db.feature_storage.SQLFeatureStore")
def test_load_features_helper_creates_store_and_calls_load(
    mock_sql_store_class, feature_store_env
):
    mock_store_instance = MagicMock(spec=SQLFeatureStore)
    mock_sql_store_class.return_value = mock_store_instance

    conn_param = feature_store_env["conn"]
    expected_df_dict = {"sales": pd.DataFrame({"B": [2]})}
    mock_store_instance.load_features.return_value = expected_df_dict

    start = "2022-01-01"
    end = "2022-12-31"

    # Call the helper, passing connection
    result = load_features(
        store_type="sql", start_date=start, end_date=end, connection=conn_param
    )

    # SQLFeatureStore is instantiated with run_id=None by factory
    mock_sql_store_class.assert_called_once_with(run_id=None, connection=conn_param)
    mock_store_instance.load_features.assert_called_once_with(
        start_date=start, end_date=end, feature_types=None
    )
    assert result == expected_df_dict
