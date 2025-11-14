"""Database queries for multiindex mapping management."""

import logging
from typing import Any

import asyncpg

from deployment.app.db.queries.core import execute_query, execute_many
from deployment.app.db.utils import split_ids_for_batching

logger = logging.getLogger("plastinka.database")


async def get_or_create_multiindex_ids_batch(
    tuples_to_process: list[tuple],
    connection: asyncpg.Connection
) -> tuple[int, ...]:
    """
    Efficiently gets or creates multiple multi-index IDs in a single batch.

    Args:
        tuples_to_process: A list of unique tuples, each representing a multi-index.
        connection: Required database connection.

    Returns:
        A tuple of multiindex_ids.
    """
    if not tuples_to_process:
        return ()

    # Normalize tuples: convert all to strings except recording_year (last element) which must be int
    normalized_tuples = []
    for tuple_values in tuples_to_process:
        # Convert all except last element to strings, keep last as int
        normalized = tuple(str(value) for value in tuple_values[:-1]) + (int(tuple_values[-1]),)
        normalized_tuples.append(normalized)

    # Create temporary table
    await execute_query(
        """
        CREATE TEMP TABLE _tuples_to_find (
            ord INTEGER PRIMARY KEY, 
            barcode TEXT, 
            artist TEXT, 
            album TEXT, 
            cover_type TEXT, 
            price_category TEXT, 
            release_type TEXT, 
            recording_decade TEXT, 
            release_decade TEXT, 
            style TEXT, 
            recording_year INTEGER
        )
        """,
        connection=connection
    )
    
    try:
        rows_with_ord = [(i,) + t for i, t in enumerate(normalized_tuples)]
        await execute_many(
            "INSERT INTO _tuples_to_find VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
            rows_with_ord,
            connection
        )

        # Insert new multiindex entries if they don't exist
        insert_query = """
            INSERT INTO dim_multiindex_mapping (
                barcode, artist, album, cover_type, price_category,
                release_type, recording_decade, release_decade, style, recording_year
            )
            SELECT barcode, artist, album, cover_type, price_category,
                   release_type, recording_decade, release_decade, style, recording_year
            FROM _tuples_to_find
            ON CONFLICT (barcode, artist, album, cover_type, price_category, release_type, recording_decade, release_decade, style, recording_year) DO NOTHING
        """
        await execute_query(insert_query, connection)

        # Get the IDs
        select_query = """
            SELECT t.ord, m.multiindex_id
            FROM _tuples_to_find t
            JOIN dim_multiindex_mapping m ON
                m.barcode = t.barcode AND
                m.artist = t.artist AND
                m.album = t.album AND
                m.cover_type = t.cover_type AND
                m.price_category = t.price_category AND
                m.release_type = t.release_type AND
                m.recording_decade = t.recording_decade AND
                m.release_decade = t.release_decade AND
                m.style = t.style AND
                m.recording_year = t.recording_year
            ORDER BY t.ord
        """
        rows = await execute_query(select_query, connection, fetchall=True) or []
        ids = tuple(int(r["multiindex_id"]) for r in rows)

    finally:
        await execute_query("DROP TABLE _tuples_to_find", connection)
    
    return ids


async def get_multiindex_mapping_by_ids(
    multiindex_ids: list[int],
    connection: asyncpg.Connection
) -> list[dict]:
    """
    Get multiindex mapping data by IDs.

    Args:
        multiindex_ids: List of multiindex IDs to get mapping for
        connection: Required database connection

    Returns:
        List of dictionaries with mapping data
    """
    if not multiindex_ids:
        return []

    # Use batching for large lists
    all_results = []
    
    for batch in split_ids_for_batching(multiindex_ids, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        query = f"""
        SELECT multiindex_id, barcode, artist, album, cover_type, price_category,
               release_type, recording_decade, release_decade, style, recording_year
        FROM dim_multiindex_mapping
        WHERE multiindex_id IN ({placeholders})
        """
        
        rows = await execute_query(query, connection=connection, params=tuple(batch), fetchall=True) or []
        all_results.extend(rows)
    
    if not all_results:
        return []

    # Create a mapping by ID
    by_id = {
        int(r["multiindex_id"]): {
            "multiindex_id": int(r["multiindex_id"]),
            "barcode": str(r["barcode"]),
            "artist": str(r["artist"]),
            "album": str(r["album"]),
            "cover_type": str(r["cover_type"]),
            "price_category": str(r["price_category"]),
            "release_type": str(r["release_type"]),
            "recording_decade": str(r["recording_decade"]),
            "release_decade": str(r["release_decade"]),
            "style": str(r["style"]),
            "recording_year": str(r["recording_year"]),
        }
        for r in all_results
    }

    return [by_id[i] for i in multiindex_ids if i in by_id]


async def get_multiindex_mapping_batch(
    tuples_to_process: list[tuple],
    connection: asyncpg.Connection
) -> dict[tuple, int]:
    """
    Get multiindex mapping for a batch of tuples.
    
    Args:
        tuples_to_process: List of tuples to get IDs for
        connection: Required database connection
        
    Returns:
        Dictionary mapping tuples to multiindex_ids
    """
    multiindex_ids = await get_or_create_multiindex_ids_batch(tuples_to_process, connection)
    return {tuple_to_process: mid for tuple_to_process, mid in zip(tuples_to_process, multiindex_ids)}

