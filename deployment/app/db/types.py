"""Type definitions and constants for database operations."""

# Define allowed metric names for dynamic queries
ALLOWED_METRICS = [
    # Standardized metric names from training phases (with prefixes)
    "val_MIC",
    "val_MIWS",
    "val_MIWS_MIC_Ratio",
    "val_loss",
    "train_loss",
    "train_MIC",
    "train_MIWS",
    "train_MIWS_MIC_Ratio",
    "training_duration_seconds",
    # FIXED: Add actual metric names with double prefixes from database
    "val_val_loss",
    "val_val_MIWS",
    "val_val_MIC",
    "val_val_MIWS_MIC_Ratio",
    "train_train_loss",
    "train_train_MIWS",
    "train_train_MIC",
    "train_train_MIWS_MIC_Ratio",
]

EXPECTED_REPORT_FEATURES = [
    "availability",
    "confidence",
    "masked_mean_sales_items",
    "masked_mean_sales_rub",
    "lost_sales",
]

EXPECTED_REPORT_FEATURES_SET = set(EXPECTED_REPORT_FEATURES)

