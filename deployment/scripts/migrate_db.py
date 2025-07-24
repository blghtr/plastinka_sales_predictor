import sqlite3
import json
import logging
import hashlib
from datetime import date, datetime
from pathlib import Path

from deployment.app.config import get_settings
from deployment.app.db.database import get_db_connection, db_transaction, execute_query, get_configs, create_or_get_config, json_default_serializer

logger = logging.getLogger("db_migration_script")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def migrate_configs_key(connection: sqlite3.Connection):
    """
    Migrates configs by renaming 'nn_model_config' to 'model_config'.
    """
    logger.info("Starting configuration key migration...")
    configs_migrated = 0
    configs_skipped_no_change = 0
    
    all_configs = get_configs(limit=1000000, connection=connection) # Fetch all configs

    for cfg in all_configs:
        original_config_id = cfg['config_id']
        config_data = cfg['config']
        is_active = bool(cfg['is_active'])
        source = cfg.get('source')

        if 'nn_model_config' in config_data:
            logger.info(f"Processing config {original_config_id} for key rename.")
            
            # Create a deep copy to avoid modifying the original dict during iteration if needed later
            # This is also good practice as we might iterate and then re-create
            new_config_data = dict(config_data) 
            new_config_data['model_config'] = new_config_data.pop('nn_model_config')
            
            # Recalculate hash for the new config data
            new_config_json = json.dumps(new_config_data, sort_keys=True, default=json_default_serializer)
            new_config_id = hashlib.md5(new_config_json.encode()).hexdigest()

            if new_config_id == original_config_id:
                logger.info(f"Config {original_config_id} already has the correct key or no effective change. Skipping.")
                configs_skipped_no_change += 1
                continue

            try:
                # Use execute_query for direct update if we are just changing the JSON string
                # This bypasses the TrainingConfig validation in create_or_get_config
                # But to maintain the active status and re-hash correctly, create_or_get_config is better.

                # If original was active, the new one should become active and deactivate others
                # If original was not active, new one should not be active
                create_or_get_config(new_config_data, is_active=is_active, source=source, connection=connection)
                
                # After successfully creating the new config, delete the old one
                # Need to be careful not to delete active config, but create_or_get_config handles deactivation logic
                # for the old one when the new one is set active.
                # If the original was active, it's now inactive because the new one is active.
                # If the original was not active, it remains non-active.
                # We can safely delete the old one, as it's no longer 'active'.
                delete_query = "DELETE FROM configs WHERE config_id = ?"
                execute_query(delete_query, (original_config_id,), connection=connection)
                
                configs_migrated += 1
                logger.info(f"Config {original_config_id} migrated to {new_config_id}.")
            except Exception as e:
                logger.error(f"Failed to migrate config {original_config_id}: {e}", exc_info=True)
                # Re-raise to trigger transaction rollback
                raise
        else:
            configs_skipped_no_change += 1

    logger.info(f"Configuration key migration finished. Migrated: {configs_migrated}, Skipped (no change/active): {configs_skipped_no_change}.")


def migrate_prediction_month(connection: sqlite3.Connection):
    """
    Updates prediction_month in prediction_results to March 2025.
    """
    logger.info("Starting prediction_month migration...")
    target_month = date(2025, 3, 1)
    
    update_query = """
        UPDATE prediction_results
        SET prediction_month = ?
        WHERE prediction_month IS NULL;
    """
    
    # We should only update if the current month is not already March 2025
    # To make it idempotent, we can check. However, the request is to fix *all* to March 2025.
    # So a direct update is fine.

    try:
        # Check if there are any records to update
        check_query = "SELECT COUNT(*) AS count FROM prediction_results WHERE prediction_month IS NULL;"
        result = execute_query(check_query, connection=connection)
        
        if result and result['count'] > 0:
            execute_query(update_query, (target_month.isoformat(),), connection=connection)
            logger.info(f"Successfully updated prediction_month for {result['count']} records to {target_month.isoformat()}.")
        else:
            logger.info("No prediction_results records found to update prediction_month or all are NULL.")
            
    except Exception as e:
        logger.error(f"Failed to migrate prediction_month: {e}", exc_info=True)
        raise # Re-raise to trigger transaction rollback

    logger.info("Prediction_month migration finished.")


def main():
    settings = get_settings()
    db_path = settings.database_path
    
    logger.info(f"Starting database migration script for database: {db_path}")

    try:
        # Wrap all migrations in a single transaction
        with db_transaction(db_path_or_conn=db_path) as conn:
            migrate_configs_key(conn)
            migrate_prediction_month(conn)
        logger.info("All migrations completed successfully and committed.")
    except Exception as e:
        logger.error(f"An error occurred during migration: {e}", exc_info=True)
        logger.error("Migration failed and changes were rolled back.")
        
if __name__ == "__main__":
    main() 