import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from deployment.app.config import get_settings

logger = logging.getLogger(__name__)

def create_database_backup() -> Path | None:
    """
    Creates a timestamped backup of the SQLite database file.

    Returns:
        Path: The path to the created backup file, or None if backup fails.
    """
    settings = get_settings()
    db_path = Path(settings.database_path)
    backup_dir = Path(settings.database_backup_dir)

    if not db_path.is_file():
        logger.error(f"Database file not found at {db_path}. Cannot create backup.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"plastinka_db_backup_{timestamp}.db"
    backup_path = backup_dir / backup_filename

    try:
        # Copy the database file
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backup created successfully: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}", exc_info=True)
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
        logger.info(f"Backup directory not found at {backup_dir}. No backups to clean.")
        return []

    if days_to_keep is None:
        # Assuming a default retention period for backups, if not explicitly defined in settings,
        # you might want to add a new setting for this in data_retention or AppSettings.
        # For now, let's use a reasonable default or suggest adding it.
        # As per security_plan.md, DataRetentionSettings has `prediction_retention_days` etc.
        # Let's add a `backup_retention_days` to DataRetentionSettings.
        # For now, I will use a placeholder and then update config.py.
        # Placeholder: 30 days
        # backup_retention_days = getattr(settings.data_retention, "backup_retention_days", 30)
        days_to_keep = settings.data_retention.backup_retention_days

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_files = []

    for backup_file in backup_dir.glob("plastinka_db_backup_*.db"):
        try:
            # Extract timestamp from filename (e.g., plastinka_db_backup_YYYYMMDD_HHMMSS.db)
            timestamp_str = backup_file.stem.replace("plastinka_db_backup_", "")
            backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            if backup_date < cutoff_date:
                backup_file.unlink()  # Delete the file
                deleted_files.append(backup_file)
                logger.info(f"Deleted old database backup: {backup_file}")
        except ValueError:
            logger.warning(f"Could not parse date from backup filename: {backup_file}. Skipping.")
        except OSError as e:
            logger.error(f"Error deleting backup file {backup_file}: {e}")

    if not deleted_files:
        logger.info(f"No database backups found older than {days_to_keep} days.")

    return deleted_files

def run_database_backup_job() -> None:
    """
    Runs the complete database backup and cleanup job.
    This function is designed to be scheduled.
    """
    settings = get_settings()
    if not settings.data_retention.cleanup_enabled: # Reuse cleanup_enabled from data_retention for consistency
        logger.info("Database backup and cleanup job is disabled in settings.")
        return

    logger.info("Starting database backup job.")
    
    # Create backup
    backup_path = create_database_backup()
    
    # Clean old backups
    if backup_path: # Only clean if backup was successfully created
        deleted_count = len(clean_old_database_backups())
        logger.info(f"Cleaned up {deleted_count} old database backup files.")
    else:
        logger.warning("Skipping cleanup of old backups because primary backup failed.")

    logger.info("Database backup job completed.") 