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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{report_type.value}_{timestamp}.html"
    
    # Update progress
    update_job_status(job_id, JobStatus.RUNNING.value, progress=20, 
                    error_message="Loading data")
    time.sleep(1)  # Simulating work
    
    # Simulate report generation based on type
    if report_type == ReportType.SALES_SUMMARY:
        await _generate_sales_summary_report(output_path, start_date, end_date, filters, job_id)
    elif report_type == ReportType.MODEL_PERFORMANCE:
        await _generate_model_performance_report(output_path, start_date, end_date, filters, job_id)
    elif report_type == ReportType.PREDICTION_ACCURACY:
        await _generate_prediction_accuracy_report(output_path, start_date, end_date, filters, job_id)
    elif report_type == ReportType.INVENTORY_ANALYSIS:
        await _generate_inventory_analysis_report(output_path, start_date, end_date, filters, job_id)
    else:
        raise ValueError(f"Unknown report type: {report_type}")
    
    return output_path


async def _generate_sales_summary_report(
    output_path: Path, start_date: datetime, end_date: datetime, filters: dict, job_id: str
) -> None:
    """Generate a sales summary report"""
    # Update progress
    update_job_status(job_id, JobStatus.RUNNING.value, progress=40, 
                    error_message="Analyzing sales data")
    time.sleep(1)  # Simulating work
    
    # Create a mock report
    df = pd.DataFrame({
        'date': pd.date_range(start=start_date or '2022-01-01', 
                              end=end_date or '2022-12-31', 
                              freq='D'),
        'sales': [i % 100 for i in range(365)]
    })
    
    # Generate a simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['sales'])
    plt.title('Daily Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.savefig(output_path.with_suffix('.png'))
    
    # Create an HTML report
    html_content = f"""
    <html>
    <head>
        <title>Sales Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Sales Summary Report</h1>
        <div class="summary">
            <p><strong>Period:</strong> {start_date or 'All time'} to {end_date or 'Present'}</p>
            <p><strong>Total Sales:</strong> {df['sales'].sum()}</p>
            <p><strong>Average Daily Sales:</strong> {df['sales'].mean():.2f}</p>
            <p><strong>Highest Daily Sales:</strong> {df['sales'].max()} (on {df.loc[df['sales'].idxmax(), 'date'].strftime('%Y-%m-%d')})</p>
        </div>
        <h2>Sales Trend</h2>
        <img src="{output_path.with_suffix('.png').name}" alt="Sales Trend">
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    update_job_status(job_id, JobStatus.RUNNING.value, progress=80, 
                    error_message="Finalizing report")
    time.sleep(1)  # Simulating work


async def _generate_model_performance_report(
    output_path: Path, start_date: datetime, end_date: datetime, filters: dict, job_id: str
) -> None:
    """Generate a model performance report"""
    # Update progress
    update_job_status(job_id, JobStatus.RUNNING.value, progress=40, 
                    error_message="Analyzing model metrics")
    time.sleep(1)  # Simulating work
    
    # Create a mock report
    models = ['Model A', 'Model B', 'Model C']
    metrics = {
        'MAPE': [15.2, 12.8, 14.5],
        'RMSE': [0.23, 0.19, 0.21],
        'MAE': [0.18, 0.15, 0.17]
    }
    
    df = pd.DataFrame(metrics, index=models)
    
    # Generate a simple bar chart
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Error Metrics')
    plt.xlabel('Model')
    plt.savefig(output_path.with_suffix('.png'))
    
    # Create an HTML report
    html_content = f"""
    <html>
    <head>
        <title>Model Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Model Performance Report</h1>
        <div class="summary">
            <p><strong>Period:</strong> {start_date or 'All time'} to {end_date or 'Present'}</p>
            <p><strong>Best Model:</strong> Model B (Lowest error metrics)</p>
        </div>
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>MAPE</th>
                <th>RMSE</th>
                <th>MAE</th>
            </tr>
            {''.join(f"<tr><td>{model}</td><td>{metrics['MAPE'][i]:.2f}</td><td>{metrics['RMSE'][i]:.2f}</td><td>{metrics['MAE'][i]:.2f}</td></tr>" for i, model in enumerate(models))}
        </table>
        <h2>Performance Comparison</h2>
        <img src="{output_path.with_suffix('.png').name}" alt="Model Performance">
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    update_job_status(job_id, JobStatus.RUNNING.value, progress=80, 
                    error_message="Finalizing report")
    time.sleep(1)  # Simulating work


async def _generate_prediction_accuracy_report(
    output_path: Path, start_date: datetime, end_date: datetime, filters: dict, job_id: str
) -> None:
    """Generate a prediction accuracy report"""
    # Create a stub report
    html_content = """
    <html>
    <head>
        <title>Prediction Accuracy Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
        </style>
    </head>
    <body>
        <h1>Prediction Accuracy Report</h1>
        <p>This is a placeholder for the prediction accuracy report.</p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    update_job_status(job_id, JobStatus.RUNNING.value, progress=80, 
                    error_message="Finalizing report")
    time.sleep(1)  # Simulating work


async def _generate_inventory_analysis_report(
    output_path: Path, start_date: datetime, end_date: datetime, filters: dict, job_id: str
) -> None:
    """Generate an inventory analysis report"""
    # Create a stub report
    html_content = """
    <html>
    <head>
        <title>Inventory Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
        </style>
    </head>
    <body>
        <h1>Inventory Analysis Report</h1>
        <p>This is a placeholder for the inventory analysis report.</p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    update_job_status(job_id, JobStatus.RUNNING.value, progress=80, 
                    error_message="Finalizing report")
    time.sleep(1)  # Simulating work 