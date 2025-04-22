# Cloud Function Interfaces

This document specifies the interfaces for the Yandex Cloud Functions that handle training and prediction operations for the Plastinka Sales Predictor.

## Overview

The system includes two primary cloud functions:
1. **Training Function** - Handles model training and hyperparameter tuning
2. **Prediction Function** - Performs forecasting using trained models

Both functions interact with the FastAPI application via well-defined interfaces and use cloud storage for data exchange.

## Common Interface Elements

### Environment Variables

All cloud functions require the following environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `API_ENDPOINT` | URL of the FastAPI application for callbacks | `https://api.example.com` |
| `API_KEY` | Authentication key for API callbacks | `sk_live_1234567890abcdef` |
| `STORAGE_BUCKET` | Cloud storage bucket name | `plastinka-ml-data` |
| `STORAGE_ACCESS_KEY` | Storage access key | `AKIAIOSFODNN7EXAMPLE` |
| `STORAGE_SECRET_KEY` | Storage secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Input Format

Both functions accept input in JSON format with the following base structure:

```json
{
  "job_id": "string",         // Unique job identifier
  "callback_url": "string",   // URL for status updates
  "storage_paths": {          // Cloud storage paths
    "input": "string",        // Input data location
    "output": "string",       // Output data location
    "models": "string"        // Model storage location (optional)
  }
}
```

### Output Format

Both functions return a result in JSON format with the following base structure:

```json
{
  "job_id": "string",         // Matching job identifier
  "status": "string",         // "success" or "error"
  "result": {                 // Result data (if successful)
    "storage_path": "string", // Path to result in cloud storage
    "metrics": {}             // Function-specific metrics
  },
  "error": {                  // Error information (if failed)
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

### Status Callback Format

Both functions send status updates to the API callback URL with the following format:

```json
{
  "job_id": "string",         // Matching job identifier
  "status": "string",         // "running", "success", or "error"
  "progress": {               // Progress information
    "percentage": number,     // 0-100 completion percentage
    "current_step": "string", // Description of current step
    "steps_completed": number,// Number of completed steps
    "steps_total": number     // Total number of steps
  },
  "logs": [                   // Recent log entries
    {
      "timestamp": "string",
      "level": "string",
      "message": "string"
    }
  ],
  "error": {                  // Error information (if status is "error")
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

## Training Function Interface

### Specific Input Format

The training function requires additional fields in the input JSON:

```json
{
  "job_id": "string",
  "callback_url": "string",
  "storage_paths": {
    "input": "string",
    "output": "string",
    "models": "string"
  },
  "training_params": {
    "model_type": "string",       // Type of model to train
    "hyperparameters": {},        // Model-specific hyperparameters
    "training_config": {          // Training configuration
      "epochs": number,
      "batch_size": number,
      "validation_split": number,
      "early_stopping": boolean,
      "patience": number
    },
    "dataset_config": {           // Dataset configuration
      "target_column": "string",
      "features": ["string"],
      "date_column": "string",
      "train_start": "string",    // ISO format date
      "train_end": "string",      // ISO format date
      "val_start": "string",      // ISO format date (optional)
      "val_end": "string"         // ISO format date (optional)
    }
  }
}
```

### Specific Output Format

The training function returns additional fields in the output JSON:

```json
{
  "job_id": "string",
  "status": "string",
  "result": {
    "storage_path": "string",
    "model_info": {
      "model_id": "string",
      "model_type": "string",
      "created_at": "string",
      "input_features": ["string"],
      "hyperparameters": {},
      "metrics": {
        "training_loss": number,
        "validation_loss": number,
        "training_metrics": {},
        "validation_metrics": {}
      }
    },
    "training_details": {
      "epochs_completed": number,
      "training_time_seconds": number,
      "early_stopping_triggered": boolean,
      "best_epoch": number
    }
  },
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

## Prediction Function Interface

### Specific Input Format

The prediction function requires additional fields in the input JSON:

```json
{
  "job_id": "string",
  "callback_url": "string",
  "storage_paths": {
    "input": "string",
    "output": "string",
    "models": "string"
  },
  "prediction_params": {
    "model_id": "string",         // ID of model to use
    "forecast_horizon": number,    // Number of periods to forecast
    "prediction_config": {         // Prediction configuration
      "include_history": boolean,  // Include historical data in output
      "return_intervals": boolean, // Include prediction intervals
      "interval_width": number     // Confidence interval width (0-1)
    },
    "dataset_config": {            // Dataset configuration
      "target_column": "string",
      "features": ["string"],
      "date_column": "string",
      "history_start": "string",   // ISO format date
      "history_end": "string"      // ISO format date
    }
  }
}
```

### Specific Output Format

The prediction function returns additional fields in the output JSON:

```json
{
  "job_id": "string",
  "status": "string",
  "result": {
    "storage_path": "string",
    "prediction_info": {
      "model_id": "string",
      "created_at": "string",
      "forecast_horizon": number,
      "prediction_start": "string",
      "prediction_end": "string",
      "metrics": {
        "mape": number,        // Mean Absolute Percentage Error (if applicable)
        "rmse": number,        // Root Mean Square Error (if applicable)
        "mae": number,         // Mean Absolute Error (if applicable)
        "other_metrics": {}
      }
    },
    "prediction_details": {
      "prediction_time_seconds": number,
      "data_points": number,
      "feature_importance": {}  // Feature importance if available
    }
  },
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

## Error Codes

Both functions use a common set of error codes:

| Code | Description |
|------|-------------|
| `INVALID_INPUT` | Invalid or missing input parameters |
| `STORAGE_ERROR` | Error accessing cloud storage |
| `DATA_ERROR` | Error processing input data |
| `MODEL_ERROR` | Error loading or using model |
| `TRAINING_ERROR` | Error during model training |
| `PREDICTION_ERROR` | Error during prediction generation |
| `RESOURCE_ERROR` | Insufficient resources (memory, CPU) |
| `TIMEOUT_ERROR` | Function execution timeout |
| `INTERNAL_ERROR` | Unexpected internal error |

## Implementation Considerations

1. **Chunking for Large Datasets**:
   - Functions should support processing data in chunks
   - Progress updates should track chunk processing

2. **Memory Management**:
   - Functions should implement careful memory management
   - Large datasets should be processed in streaming fashion when possible

3. **Exception Handling**:
   - All exceptions should be caught and reported via the callback mechanism
   - Stack traces should be included in error details

4. **Timeouts**:
   - Functions should implement internal timeout tracking
   - Long-running operations should provide regular progress updates

5. **Recovery Mechanisms**:
   - Functions should support resuming from intermediate checkpoints
   - Partial results should be saved when possible 