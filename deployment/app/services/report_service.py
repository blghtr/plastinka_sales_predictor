from datetime import datetime
import time
import os
import json
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from deployment.app.db.database import update_job_status, create_report_result
from deployment.app.models.api_models import JobStatus, ReportParams, ReportType


async def generate_report(job_id: str, params: ReportParams) -> None:
    """
    Generate a report using the specified parameters.
    
    Args:
        job_id: ID of the job
        params: Report parameters
    """
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Prepare report parameters
        report_type = params.report_type
        start_date = params.start_date
        end_date = params.end_date
        filters = params.filters or {}
        
        # Log the parameters
        update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=10,
            error_message=f"Generating {report_type} report"
        )
        
        # Create output directory
        output_dir = Path(f"deployment/data/reports/{report_type}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report based on type
        output_path = await _generate_report_by_type(
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
        
        # Create result
        result_id = create_report_result(
            job_id=job_id,
            report_type=report_type.value,
            parameters={
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "filters": filters
            },
            output_path=str(output_path)
        )
        
        # Update job as completed
        update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            progress=100,
            result_id=result_id
        )
        
    except Exception as e:
        # Update job as failed with error message
        update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message=str(e)
        )
        # Re-raise for logging
        raise


async def _generate_report_by_type(
    report_type: ReportType,
    start_date: datetime,
    end_date: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """
    Generate a report based on its type.
    
    Args:
        report_type: Type of report to generate
        start_date: Start date for the report
        end_date: End date for the report
        filters: Additional filters for the report
        output_dir: Directory to save the report
        job_id: ID of the job
        
    Returns:
        Path to the generated report
    """
    