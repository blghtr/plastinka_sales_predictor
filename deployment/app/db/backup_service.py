"""PostgreSQL database backup service using pg_dump."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from deployment.app.config import get_settings

logger = logging.getLogger(__name__)


async def create_database_backup() -> Path | None:
    """
    Creates a timestamped backup of the PostgreSQL database using pg_dump.

    Returns:
        Path: The path to the created backup file, or None if backup fails.
    """
    settings = get_settings()
    backup_dir = Path(settings.database_backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"plastinka_db_backup_{timestamp}.sql"
    backup_path = backup_dir / backup_filename

    # Build pg_dump command
    pg_dump_args = [
        "pg_dump",
        "-h", settings.db.postgres_host,
        "-p", str(settings.db.postgres_port),
        "-U", settings.db.postgres_user,
        "-d", settings.db.postgres_database,
        "-F", "c",  # Custom format (compressed)
        "-f", str(backup_path),
    ]

    # Add SSL mode if specified
    if settings.db.postgres_ssl_mode != "disable":
        pg_dump_args.extend(["--no-password"])  # Use .pgpass or environment variable

    # Set PGPASSWORD environment variable
    env = {"PGPASSWORD": settings.db.postgres_password}

    try:
        # Run pg_dump asynchronously
        process = await asyncio.create_subprocess_exec(
            *pg_dump_args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(
                f"pg_dump failed with return code {process.returncode}: {error_msg}"
            )
            # Clean up partial backup file if it exists
            if backup_path.exists():
                backup_path.unlink()
            return None

        if backup_path.exists() and backup_path.stat().st_size > 0:
            logger.info(f"Database backup created successfully: {backup_path}")
            return backup_path
        else:
            logger.error("Backup file was not created or is empty")
            return None

    except FileNotFoundError:
        logger.error(
            "pg_dump command not found. Please ensure PostgreSQL client tools are installed."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}", exc_info=True)
        # Clean up partial backup file if it exists
        if backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass
        return None


def clean_old_database_backups(days_to_keep: int | None = None) -> list[Path]:
    """
    Removes old database backup files beyond the specified retention period.

    Args:
        days_to_keep: Number of days to keep backup files for.
                      If None, uses the value from settings.

    Returns:
        list[Path]: A list of paths to the deleted backup files.
    """
    settings = get_settings()
    backup_dir = Path(settings.database_backup_dir)

    if not backup_dir.is_dir():
        logger.info(
            f"Backup directory not found at {backup_dir}. No backups to clean."
        )
        return []

    if days_to_keep is None:
        days_to_keep = settings.data_retention.backup_retention_days

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_files = []

    # Match both .sql and .dump files (pg_dump custom format uses .dump extension)
    for pattern in ["plastinka_db_backup_*.sql", "plastinka_db_backup_*.dump"]:
        for backup_file in backup_dir.glob(pattern):
            try:
                # Extract timestamp from filename
                # Format: plastinka_db_backup_YYYYMMDD_HHMMSS.sql or .dump
                timestamp_str = backup_file.stem.replace("plastinka_db_backup_", "")
                backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if backup_date < cutoff_date:
                    backup_file.unlink()  # Delete the file
                    deleted_files.append(backup_file)
                    logger.info(f"Deleted old database backup: {backup_file}")
            except ValueError:
                logger.warning(
                    f"Could not parse date from backup filename: {backup_file}. Skipping."
                )
            except OSError as e:
                logger.error(f"Error deleting backup file {backup_file}: {e}")

    if not deleted_files:
        logger.info(
            f"No database backups found older than {days_to_keep} days."
        )

    return deleted_files


async def run_database_backup_job() -> None:
    """
    Runs the complete database backup and cleanup job.
    This function is designed to be scheduled and is async for PostgreSQL operations.
    """
    settings = get_settings()
    if not settings.data_retention.cleanup_enabled:
        logger.info("Database backup and cleanup job is disabled in settings.")
        return

    logger.info("Starting database backup job.")

    # Create backup
    backup_path = await create_database_backup()

    # Clean old backups
    if backup_path:  # Only clean if backup was successfully created
        deleted_count = len(clean_old_database_backups())
        logger.info(f"Cleaned up {deleted_count} old database backup files.")
    else:
        logger.warning(
            "Skipping cleanup of old backups because primary backup failed."
        )

    logger.info("Database backup job completed.")
