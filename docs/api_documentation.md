# Plastinka Sales Predictor API Documentation

This document provides detailed information about the Plastinka Sales Predictor API, including endpoints, request/response formats, error handling, and authentication.

## Base URL

```
http://localhost:8000
```

For production, the base URL will be your deployed server URL.

## Authentication

API authentication is performed using API keys via the `Authorization` header.

```
Authorization: Bearer YOUR_API_KEY
```

## API Endpoints

### Health Check

#### GET /api/v1/health

Check the health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2023-10-30T12:34:56.789Z",
  "components": {
    "database": "healthy",
    "cloud_storage": "healthy",
    "cloud_functions": "healthy"
  }
}
```

#### GET /api/v1/health/stats

Get detailed statistics about API operations, including retry metrics.

**Response:**
```json
{
  "status": "healthy",
  "retry_stats": {
    "cloud_storage": {
      "total_attempts": 120,
      "successful_retries": 5,
      "failed_retries": 1,
      "average_retry_time_ms": 156
    },
    "cloud_function": {
      "total_attempts": 87,
      "successful_retries": 3,
      "failed_retries": 0,
      "average_retry_time_ms": 212
    }
  },
  "uptime_seconds": 7200
}
```

### Jobs API

#### POST /api/v1/jobs/data-upload

Upload data files for processing.

**Request:**
- `stock_file`: Excel file with stock data (Form file)
- `sales_files`: List of Excel files with sales data (Form files)
- `cutoff_date`: Cutoff date for data processing (Form field, format: DD.MM.YYYY)

**Response:**
```json
{
  "job_id": "abc123def456",
  "status": "pending"
}
```

#### POST /api/v1/jobs/training

Start a model training job.

**Request:**
```json
{
  "model_type": "prophet",
  "features": ["sales", "stock", "seasonality"],
  "start_date": "01.01.2022",
  "end_date": "30.09.2022",
  "hyperparameters": {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0
  }
}
```

**Response:**
```json
{
  "job_id": "abc123def456",
  "status": "pending"
}
```

#### POST /api/v1/jobs/prediction

Generate predictions using a trained model.

**Request:**
```json
{
  "model_id": "model123",
  "start_date": "01.10.2022",
  "end_date": "31.12.2022",
  "prediction_interval": 0.95
}
```

**Response:**
```json
{
  "job_id": "abc123def456",
  "status": "pending"
}
```

#### POST /api/v1/jobs/reports

Generate a report based on prediction results.

**Request:**
```json
{
  "prediction_id": "pred123",
  "report_type": "summary",
  "format": "pdf"
}
```

**Response:**
```json
{
  "job_id": "abc123def456",
  "status": "pending"
}
```

#### GET /api/v1/jobs/{job_id}

Get details about a job.

**Response:**
```json
{
  "job_id": "abc123def456",
  "type": "training",
  "status": "completed",
  "parameters": {
    "model_type": "prophet",
    "features": ["sales", "stock", "seasonality"],
    "start_date": "01.01.2022",
    "end_date": "30.09.2022"
  },
  "created_at": "2023-10-01T12:00:00Z",
  "updated_at": "2023-10-01T12:10:00Z",
  "result": {
    "model_id": "model123",
    "metrics": {
      "mape": 5.2,
      "rmse": 10.5
    }
  }
}
```

#### GET /api/v1/jobs

List all jobs with optional filtering.

**Parameters:**
- `job_type`: Filter by job type (Optional)
- `status`: Filter by job status (Optional)
- `limit`: Maximum number of jobs to return (Default: 100)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "abc123def456",
      "type": "training",
      "status": "completed",
      "created_at": "2023-10-01T12:00:00Z",
      "updated_at": "2023-10-01T12:10:00Z"
    },
    {
      "job_id": "ghi789jkl012",
      "type": "prediction",
      "status": "in_progress",
      "created_at": "2023-10-01T12:30:00Z",
      "updated_at": "2023-10-01T12:30:00Z"
    }
  ],
  "total": 2
}
```

## Error Handling

The API uses a standardized error response format for all endpoints.

### Error Response Format

```json
{
  "error": {
    "message": "Human-readable error message",
    "code": "error_code",
    "request_id": "unique-request-id",
    "details": {
      "field-specific": "error details"
    },
    "retry_info": {
      "retry_after": 30,
      "max_retries": 3
    }
  }
}
```

- `message`: Human-readable error message
- `code`: Machine-readable error code
- `request_id`: Unique identifier for the request (use this when reporting issues)
- `details`: Additional error details, varies based on error type
- `retry_info`: Optional retry information for retryable errors

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `validation_error` | Invalid input parameters |
| 401 | `unauthorized` | Missing or invalid API key |
| 403 | `forbidden` | Insufficient permissions |
| 404 | `not_found` | Resource not found |
| 422 | `validation_error` | Request validation failed |
| 429 | `rate_limit_exceeded` | Too many requests |
| 500 | `internal_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

### Validation Errors

Validation errors include details about which fields failed validation:

```json
{
  "error": {
    "message": "Validation error",
    "code": "validation_error",
    "request_id": "abc123def456",
    "details": {
      "errors": [
        {
          "loc": ["body", "start_date"],
          "msg": "Invalid date format. Expected format: DD.MM.YYYY",
          "type": "value_error"
        }
      ]
    }
  }
}
```

### Retryable Errors

Some errors are marked as retryable, meaning the client should retry the request after a specified delay:

```json
{
  "error": {
    "message": "Cloud function temporarily unavailable",
    "code": "cloud_function_unavailable",
    "request_id": "abc123def456",
    "details": {
      "function_name": "predict-sales"
    },
    "retry_info": {
      "retry_after": 30,
      "max_retries": 3
    }
  }
}
```

For retryable errors:
1. Wait for the duration specified in `retry_after` (in seconds)
2. Retry the request with the same parameters
3. Do not exceed the number of retries specified in `max_retries`

### Error Handling Best Practices

1. **Always check the status code** before processing the response
2. **Keep track of the request_id** for error tracking and support
3. **Implement exponential backoff** for retryable errors
4. **Log complete error responses** for debugging
5. **Handle validation errors** by fixing the input data before retrying

## Date Range Validation

The API validates date ranges for historical data and forecasts with specific constraints:

### Historical Date Ranges
- Minimum date: January 1, 2000
- Maximum date: Current date (can't be in the future)
- Maximum range: 5 years (1,825 days)

### Forecast Date Ranges
- Minimum date: Current date
- Maximum date: 2 years in the future
- Maximum range: 1 year (365 days)

Date formats must be DD.MM.YYYY unless otherwise specified.

## Configuration

For API configuration options, see [Environment Variables Documentation](environment_variables.md).

## Rate Limiting

The API implements rate limiting to protect against abuse. Current limits:

- 100 requests per minute per IP address
- 10 concurrent jobs per API key

When rate limits are exceeded, the API returns a 429 status code with a `retry_after` header indicating when to retry.

## SDK and Client Libraries

Official client libraries for the API are available in:

- Python: `pip install plastinka-client`
- JavaScript: `npm install plastinka-client`

## Support

For API support, please include:
- API version
- Request ID from error responses
- Full request details (endpoint, parameters)
- Error message and code 