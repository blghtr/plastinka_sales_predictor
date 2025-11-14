from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import pytest

TEST_X_API_KEY = "test_x_api_key_conftest"
TEST_BEARER_TOKEN = "test_admin_token"


class TestPredictionsApi:
    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_success(self, async_api_client, dal, auth_header_name, auth_token):
        """Test successful retrieval of predictions history with all data."""
        # Arrange - Create test data
        today = date.today()
        prediction_month = today.replace(day=1)
        
        # Create multiindex
        multiindex_ids = await dal.get_or_create_multiindex_ids_batch([
            (
                "1234567890", "Test Artist", "Test Album", "CD", "Standard",
                "Studio", "2010s", "2020s", "Rock", 2015
            ),
        ])
        multiindex_id = multiindex_ids[0]
        
        # Create model
        job_id = await dal.create_job(job_type="training", parameters={})
        model_id = "test_model_123"
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=False,
        )
        
        # Create prediction result
        result_id = await dal.create_prediction_result(
            job_id=job_id,
            model_id=model_id,
            prediction_month=prediction_month,
            output_path="/fake/path/predictions.csv",
            summary_metrics={},
        )
        
        # Insert prediction
        import pandas as pd
        predictions_df = pd.DataFrame({
            "barcode": ["1234567890"],
            "artist": ["Test Artist"],
            "album": ["Test Album"],
            "cover_type": ["CD"],
            "price_category": ["Standard"],
            "release_type": ["Studio"],
            "recording_decade": ["2010s"],
            "release_decade": ["2020s"],
            "style": ["Rock"],
            "recording_year": [2015],
            "0.05": [0.0],
            "0.25": [0.0],
            "0.5": [0.35],
            "0.75": [1.92],
            "0.95": [5.06],
        })
        await dal.insert_predictions(
            result_id=result_id,
            model_id=model_id,
            prediction_month=prediction_month,
            df=predictions_df,
        )
        
        # Insert actual sales
        await dal.execute_raw_query(
            "INSERT INTO fact_sales (multiindex_id, data_date, value) VALUES ($1, $2, $3)",
            (multiindex_id, prediction_month, 0.0),
        )
        
        # Insert features
        await dal.insert_report_features([
            (
                prediction_month,
                multiindex_id,
                1.0,  # availability
                0.2,  # confidence
                0.0,  # masked_mean_sales_items
                0.0,  # masked_mean_sales_rub
                0.0,  # lost_sales
                datetime.now(),  # created_at
            )
        ])
        
        # Act
        response = await async_api_client.get(
            f"/api/v1/predictions/history?prediction_month_from={prediction_month.isoformat()}",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "metadata" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["multiindex_id"] == multiindex_id
        assert len(data["items"][0]["predictions"]) == 1
        assert len(data["items"][0]["actuals"]) == 1
        assert len(data["items"][0]["features"]) == 1
        assert data["metadata"]["total_items"] == 1

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_with_filters(self, async_api_client, dal, auth_header_name, auth_token):
        """Test predictions history with multiindex_id and date range filters."""
        # Arrange
        today = date.today()
        prediction_month = today.replace(day=1)
        
        multiindex_ids = await dal.get_or_create_multiindex_ids_batch([
            (
                "1111111111", "Artist A", "Album A", "CD", "Standard",
                "Studio", "2010s", "2020s", "Rock", 2015
            ),
        ])
        multiindex_id = multiindex_ids[0]
        
        job_id = await dal.create_job(job_type="training", parameters={})
        model_id = "test_model_filter"
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=False,
        )
        
        result_id = await dal.create_prediction_result(
            job_id=job_id,
            model_id=model_id,
            prediction_month=prediction_month,
            output_path="/fake/path/predictions.csv",
            summary_metrics={},
        )
        
        import pandas as pd
        predictions_df = pd.DataFrame({
            "barcode": ["1111111111"],
            "artist": ["Artist A"],
            "album": ["Album A"],
            "cover_type": ["CD"],
            "price_category": ["Standard"],
            "release_type": ["Studio"],
            "recording_decade": ["2010s"],
            "release_decade": ["2020s"],
            "style": ["Rock"],
            "recording_year": [2015],
            "0.05": [1.0],
            "0.25": [2.0],
            "0.5": [3.0],
            "0.75": [4.0],
            "0.95": [5.0],
        })
        await dal.insert_predictions(
            result_id=result_id,
            model_id=model_id,
            prediction_month=prediction_month,
            df=predictions_df,
        )
        
        # Act
        response = await async_api_client.get(
            f"/api/v1/predictions/history?multiindex_id={multiindex_id}&prediction_month_from={prediction_month.isoformat()}&prediction_month_to={prediction_month.isoformat()}",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["multiindex_id"] == multiindex_id

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_without_actuals(self, async_api_client, dal, auth_header_name, auth_token):
        """Test predictions history with include_actuals=False."""
        # Arrange
        today = date.today()
        prediction_month = today.replace(day=1)
        
        multiindex_ids = await dal.get_or_create_multiindex_ids_batch([
            (
                "2222222222", "Artist B", "Album B", "CD", "Standard",
                "Studio", "2010s", "2020s", "Rock", 2015
            ),
        ])
        multiindex_id = multiindex_ids[0]
        
        job_id = await dal.create_job(job_type="training", parameters={})
        model_id = "test_model_no_actuals"
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=False,
        )
        
        result_id = await dal.create_prediction_result(
            job_id=job_id,
            model_id=model_id,
            prediction_month=prediction_month,
            output_path="/fake/path/predictions.csv",
            summary_metrics={},
        )
        
        import pandas as pd
        predictions_df = pd.DataFrame({
            "barcode": ["2222222222"],
            "artist": ["Artist B"],
            "album": ["Album B"],
            "cover_type": ["CD"],
            "price_category": ["Standard"],
            "release_type": ["Studio"],
            "recording_decade": ["2010s"],
            "release_decade": ["2020s"],
            "style": ["Rock"],
            "recording_year": [2015],
            "0.05": [1.0],
            "0.25": [2.0],
            "0.5": [3.0],
            "0.75": [4.0],
            "0.95": [5.0],
        })
        await dal.insert_predictions(
            result_id=result_id,
            model_id=model_id,
            prediction_month=prediction_month,
            df=predictions_df,
        )
        
        # Act
        response = await async_api_client.get(
            f"/api/v1/predictions/history?prediction_month_from={prediction_month.isoformat()}&include_actuals=false",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert len(data["items"][0]["actuals"]) == 0

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_without_features(self, async_api_client, dal, auth_header_name, auth_token):
        """Test predictions history with include_features=False."""
        # Arrange
        today = date.today()
        prediction_month = today.replace(day=1)
        
        multiindex_ids = await dal.get_or_create_multiindex_ids_batch([
            (
                "3333333333", "Artist C", "Album C", "CD", "Standard",
                "Studio", "2010s", "2020s", "Rock", 2015
            ),
        ])
        multiindex_id = multiindex_ids[0]
        
        job_id = await dal.create_job(job_type="training", parameters={})
        model_id = "test_model_no_features"
        await dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=False,
        )
        
        result_id = await dal.create_prediction_result(
            job_id=job_id,
            model_id=model_id,
            prediction_month=prediction_month,
            output_path="/fake/path/predictions.csv",
            summary_metrics={},
        )
        
        import pandas as pd
        predictions_df = pd.DataFrame({
            "barcode": ["3333333333"],
            "artist": ["Artist C"],
            "album": ["Album C"],
            "cover_type": ["CD"],
            "price_category": ["Standard"],
            "release_type": ["Studio"],
            "recording_decade": ["2010s"],
            "release_decade": ["2020s"],
            "style": ["Rock"],
            "recording_year": [2015],
            "0.05": [1.0],
            "0.25": [2.0],
            "0.5": [3.0],
            "0.75": [4.0],
            "0.95": [5.0],
        })
        await dal.insert_predictions(
            result_id=result_id,
            model_id=model_id,
            prediction_month=prediction_month,
            df=predictions_df,
        )
        
        # Act
        response = await async_api_client.get(
            f"/api/v1/predictions/history?prediction_month_from={prediction_month.isoformat()}&include_features=false",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert len(data["items"][0]["features"]) == 0

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_invalid_date_format(self, async_api_client, dal, auth_header_name, auth_token):
        """Test 400 error for invalid date format."""
        # Act
        response = await async_api_client.get(
            "/api/v1/predictions/history?prediction_month_from=2025-09-15",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "validation_error"
        # Check error details
        assert "details" in data["error"]
        assert data["error"]["details"]["parameter"] == "prediction_month_from"

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_invalid_date_range(self, async_api_client, dal, auth_header_name, auth_token):
        """Test 400 error for invalid date range (from > to)."""
        today = date.today()
        month_from = today.replace(day=1)
        month_to = (month_from - relativedelta(months=1)).replace(day=1)
        
        # Act
        response = await async_api_client.get(
            f"/api/v1/predictions/history?prediction_month_from={month_from.isoformat()}&prediction_month_to={month_to.isoformat()}",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "validation_error"
        # Check error details
        assert "details" in data["error"]
        assert "prediction_month_from" in data["error"]["details"]
        assert "prediction_month_to" in data["error"]["details"]

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @pytest.mark.asyncio
    async def test_get_predictions_history_not_found(self, async_api_client, dal, auth_header_name, auth_token):
        """Test 404 error when no predictions found."""
        # Act
        response = await async_api_client.get(
            "/api/v1/predictions/history?multiindex_id=99999&prediction_month_from=2025-09-01&prediction_month_to=2025-09-01",
            headers={auth_header_name: auth_token}
        )
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "http_404"
        # Check nested error details
        assert "original_detail" in data["error"]["details"]
        assert data["error"]["details"]["original_detail"]["code"] == "no_predictions_found"

    @pytest.mark.asyncio
    async def test_get_predictions_history_auth_required(self, async_api_client):
        """Test that authentication is required."""
        # Act
        response = await async_api_client.get("/api/v1/predictions/history")
        
        # Assert
        assert response.status_code == 401

