import io
from datetime import datetime

import pandas as pd

from deployment.app.utils.validation import (
    validate_date_format,
    validate_excel_file,
    validate_pagination_params,
)


def test_validate_excel_file_valid():
    """Test validation of a valid Excel file."""
    # Create a test Excel file in memory
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]})
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    is_valid, error = validate_excel_file(buffer.read())
    assert is_valid
    assert error == ""


def test_validate_excel_file_invalid():
    """Test validation of an invalid Excel file."""
    # Create an invalid file (just some random bytes)
    invalid_data = b"This is not an Excel file"

    is_valid, error = validate_excel_file(invalid_data)
    assert not is_valid
    assert "Invalid Excel file" in error


def test_validate_excel_file_with_columns():
    """Test validation of an Excel file with specific columns."""
    # Create a test Excel file with specific columns
    df = pd.DataFrame({"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]})
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    # Test with correct expected columns
    is_valid, error = validate_excel_file(
        buffer.read(), expected_columns=["Column1", "Column2"]
    )
    assert is_valid
    assert error == ""

    # Test with incorrect expected columns
    buffer.seek(0)
    is_valid, error = validate_excel_file(
        buffer.read(), expected_columns=["Column1", "Column3"]
    )
    assert not is_valid
    assert "Missing required columns" in error


def test_validate_date_format():
    """Test date format validation."""
    # Valid date
    is_valid, parsed_date = validate_date_format("01.01.2022")
    assert is_valid
    assert isinstance(parsed_date, datetime)
    assert parsed_date.day == 1
    assert parsed_date.month == 1
    assert parsed_date.year == 2022

    # Invalid date
    is_valid, parsed_date = validate_date_format("not-a-date")
    assert not is_valid
    assert parsed_date is None

    # Invalid format
    is_valid, parsed_date = validate_date_format("2022-01-01")
    assert not is_valid
    assert parsed_date is None


def test_validate_pagination_params():
    """Test pagination parameter validation."""
    # Normal case
    offset, limit = validate_pagination_params(10, 50)
    assert offset == 10
    assert limit == 50

    # Negative offset
    offset, limit = validate_pagination_params(-10, 50)
    assert offset == 0
    assert limit == 50

    # Limit too large
    offset, limit = validate_pagination_params(10, 2000)
    assert offset == 10
    assert limit == 1000

    # Limit too small
    offset, limit = validate_pagination_params(10, 0)
    assert offset == 10
    assert limit == 1
