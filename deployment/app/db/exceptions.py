"""Database exceptions for PostgreSQL operations."""

from typing import Any


class DatabaseError(Exception):
    """Exception raised for database errors."""

    def __init__(
        self,
        message: str,
        query: str = None,
        params: tuple = None,
        original_error: Exception = None,
    ):
        self.message = message
        self.query = query
        self.params = params
        self.original_error = original_error
        super().__init__(self.message)

