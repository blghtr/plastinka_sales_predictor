import shutil
from pathlib import Path

# Import our custom modules
from deployment.app.config import get_settings
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.feature_storage import save_features
from deployment.app.models.api_models import JobStatus

# Import the necessary functions from the original codebase
from plastinka_sales_predictor.data_preparation import process_data


async def process_data_files(
    job_id: str,
    stock_file_path: str,
    sales_files_paths: list[str],
    cutoff_date: str,
    temp_dir_path: str,
    dal: DataAccessLayer,
) -> None:
    """
    Process uploaded files to extract features.

    Args:
        job_id: ID of the job
        stock_file_path: Path to the saved stock file
        sales_files_paths: List of paths to the saved sales files
        cutoff_date: Cutoff date for processing (DD.MM.YYYY)
        temp_dir_path: Path to the temporary directory used for this job
        dal: DataAccessLayer instance
    """
    temp_dir = Path(temp_dir_path)
    stock_path = Path(stock_file_path)
    settings = get_settings()

    try:
        # Update job status to running
        dal.update_job_status(job_id, JobStatus.RUNNING.value, progress=0)

        # Check if files exist
        if not stock_path.exists():
            raise FileNotFoundError(f"Stock file not found at {stock_path}")
        for p in sales_files_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Sales file not found at {p}")

        dal.update_job_status(job_id, JobStatus.RUNNING.value, progress=20)

        # Process the data using the existing pipeline
        sales_dir_path = temp_dir / "sales"
        features = process_data(
            stock_path=str(stock_path),
            sales_path=str(sales_dir_path),
            cutoff_date=cutoff_date,
            bins=settings.price_category_interval_index,
        )

        dal.update_job_status(job_id, JobStatus.RUNNING.value, progress=80)

        # Save features using our SQL feature storage
        stock_filename = stock_path.name
        sales_filenames = [Path(p).name for p in sales_files_paths]
        source_files = ", ".join([stock_filename] + sales_filenames)
        run_id = save_features(features, cutoff_date, source_files, store_type="sql", dal=dal)

        # Create result record
        result_id = dal.create_data_upload_result(
            job_id=job_id,
            records_processed=sum(
                df.shape[0] for df in features.values() if hasattr(df, "shape")
            ),
            features_generated=list(features.keys()),
            processing_run_id=run_id,
        )

        # Update job as completed
        dal.update_job_status(
            job_id, JobStatus.COMPLETED.value, progress=100, result_id=result_id
        )

    except Exception as e:
        # Update job as failed with error message
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=str(e))
        # Re-raise for logging
        raise
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

