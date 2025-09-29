import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.feature_storage import (
    EXPECTED_REPORT_FEATURES,
    SQLFeatureStore,
)
from deployment.app.db.schema import MULTIINDEX_NAMES


@pytest.fixture
def comprehensive_feature_store_env():
    """
    Set up a test environment with a temporary SQLite database, DAL,
    and multiple products in the multi-index mapping.
    """
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    dal = DataAccessLayer(db_path=temp_db_file)
    conn = dal._connection
    cursor = conn.cursor()

    # Insert multiple products for comprehensive testing
    products_data = [
        (1, '111', 'Artist A', 'Album A', 'CD', 'Std', 'Studio', '2010s', '2020s', 'Rock', '2015'),
        (2, '222', 'Artist B', 'Album B', 'Vinyl', 'Ltd', 'Live', '2000s', '2020s', 'Pop', '2008'),
        (3, '333', 'Artist A', 'Album C', 'CD', 'Std', 'Studio', '2010s', '2020s', 'Rock', '2018'),
    ]
    # Correctly build the INSERT statement with named columns
    columns_str = ', '.join(MULTIINDEX_NAMES)
    placeholders = ', '.join('?' * (len(MULTIINDEX_NAMES) + 1))
    cursor.executemany(f"""
    INSERT INTO dim_multiindex_mapping (multiindex_id, {columns_str})
    VALUES ({placeholders})
    """, products_data)
    conn.commit()

    yield dal

    dal.close()
    if os.path.exists(temp_db_file):
        os.unlink(temp_db_file)


def test_save_and_load_all_features_end_to_end(comprehensive_feature_store_env):
    """
    A comprehensive end-to-end test for SQLFeatureStore.
    It saves multiple feature types (pivoted and flat) using the public API
    and then loads them back, performing detailed validation.
    This test is designed to fail if the save/load logic is incorrect,
    especially in how it handles different data formats and missing values.
    """
    dal = comprehensive_feature_store_env

    # 1. ARRANGE: Create complex, wide-format DataFrames

    # Product Multi-indexes
    idx1 = ('111', 'Artist A', 'Album A', 'CD', 'Std', 'Studio', '2010s', '2020s', 'Rock', '2015')
    idx2 = ('222', 'Artist B', 'Album B', 'Vinyl', 'Ltd', 'Live', '2000s', '2020s', 'Pop', '2008')
    multi_index = pd.MultiIndex.from_tuples([idx1, idx2], names=MULTIINDEX_NAMES)

    # Date Index
    date1 = pd.to_datetime("2023-01-01")
    date2 = pd.to_datetime("2023-01-02")
    date_index = [date1, date2]

    # --- Create DataFrames for saving ---

    # Sales (pivoted format: products in index, dates in columns)
    sales_data = [[10.0, 12.0], [0.0, 22.0]]
    sales_df_wide = pd.DataFrame(sales_data, index=multi_index, columns=date_index)

    # Movement (pivoted, with a NaN to test handling)
    movement_data = [[-5.0, np.nan], [3.0, -8.0]]
    movement_df_wide = pd.DataFrame(movement_data, index=multi_index, columns=date_index)

    # Report Features (created in a format that will be processed into the expected wide format)
    # The save logic for report_features expects a wide df with features as columns
    # and a multi-level index of (_date, product_attributes...)
    final_tuples = []
    for date in date_index:
        for prod_tuple in multi_index:
            final_tuples.append((date,) + prod_tuple)

    report_index = pd.MultiIndex.from_tuples(final_tuples, names=['_date'] + MULTIINDEX_NAMES)
    report_features_df_wide = pd.DataFrame(index=report_index)
    report_features_df_wide['availability'] = [0.9, 0.8, 0.95, 0.85]
    report_features_df_wide['confidence'] = [0.7, 0.6, 0.75, 0.65]
    report_features_df_wide['masked_mean_sales_items'] = [5.0, 7.0, 6.0, 8.0]
    report_features_df_wide['masked_mean_sales_rub'] = [500.0, 700.0, 600.0, 800.0]
    report_features_df_wide['lost_sales'] = [1.0, 2.0, 0.0, 3.0]


    features_to_save = {
        "sales": sales_df_wide,
        "movement": movement_df_wide,
        "report_features": report_features_df_wide
    }

    # 2. Save them using the PUBLIC API
    with SQLFeatureStore(dal=dal, run_id=1) as store:
        store.save_features(features_to_save)

    # 3. ACT: Load them back using the PUBLIC API
    with SQLFeatureStore(dal=dal) as store:
        loaded_features = store.load_features(start_date="2023-01-01", end_date="2023-01-02")

    # 4. ASSERT
    assert "sales" in loaded_features
    assert "movement" in loaded_features
    assert "report_features" in loaded_features

    # --- Assertions for 'sales' (pivoted) ---
    sales_loaded = loaded_features["sales"]
    assert sales_loaded.shape == (2, 2)  # 2 dates, 2 products
    assert isinstance(sales_loaded.columns, pd.MultiIndex)
    assert sales_loaded.index.name == "_date"
    # Zeros are dropped on save, then filled on load. The loaded data is transposed.
    expected_sales = sales_df_wide.T.fillna(0)
    expected_sales.index.name = "_date"  # Align index name for comparison
    pd.testing.assert_frame_equal(sales_loaded, expected_sales)

    # --- Assertions for 'movement' (pivoted) ---
    movement_loaded = loaded_features["movement"]
    # The NaN value was dropped during save, so only one product remains in the loaded data.
    # The expected shape is (2, 1) because the row with NaN was dropped.
    assert movement_loaded.shape == (2, 1)  # 2 dates, 1 product (the one without NaN)
    assert isinstance(movement_loaded.columns, pd.MultiIndex)
    # Create expected movement data with only the non-NaN product
    expected_movement = movement_df_wide.loc[['222']].T  # Only the second product (no NaN)
    expected_movement.index.name = "_date"  # Align index name for comparison
    pd.testing.assert_frame_equal(movement_loaded, expected_movement)

    # --- Assertions for 'report_features' (flat) ---
    report_loaded = loaded_features["report_features"]
    assert isinstance(report_loaded, pd.DataFrame)
    assert not isinstance(report_loaded.columns, pd.MultiIndex)
    assert "artist" in report_loaded.columns  # Check enrichment
    assert "multiindex_id" in report_loaded.columns  # Check that multiindex_id is preserved

    # 4 rows = 2 products * 2 dates
    assert report_loaded.shape[0] == 4
    # Check number of columns: product attributes + date + expected features + multiindex_id
    assert report_loaded.shape[1] == len(MULTIINDEX_NAMES) + 1 + len(EXPECTED_REPORT_FEATURES) + 1

    # Check a specific value to ensure correct join and data integrity
    report_loaded['data_date'] = pd.to_datetime(report_loaded['data_date'])
    artist_a_data = report_loaded[report_loaded['artist'] == 'Artist A'].sort_values('data_date').reset_index()
    assert artist_a_data.loc[0, 'availability'] == 0.9
    assert artist_a_data.loc[1, 'confidence'] == 0.75
