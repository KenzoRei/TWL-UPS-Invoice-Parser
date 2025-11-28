"""Helper utility functions for UPS Invoice Parser."""

import pandas as pd
import numpy as np
from typing import Optional


def is_blank(val) -> bool:
    """
    Check if value is considered empty.
    
    Returns True if value is:
    - NaN / None
    - Empty string ""
    - String containing only whitespace
    - String "nan" (case-insensitive)
    
    Args:
        val: Value to check (any type)
        
    Returns:
        True if value is considered empty, False otherwise
    """
    if pd.isna(val):
        return True
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == "" or stripped.lower() == "nan":
            return True
    return False


def extract_dims(series: pd.Series, col_prefix: str) -> pd.DataFrame:
    """
    Extract dimensions (LxWxH) from a pandas Series.
    
    Parses dimension strings like "10x8x6" or "12.5 x 10.2 x 8.0"
    and extracts length, width, and height as separate numeric columns.
    
    Args:
        series: Series containing dimension strings
        col_prefix: Prefix for output columns (e.g., "Billed" or "Entered")
        
    Returns:
        DataFrame with columns: {prefix} Length, {prefix} Width, {prefix} Height
        
    Examples:
        >>> s = pd.Series(["10x8x6", "12.5x10.2x8.0", "invalid"])
        >>> extract_dims(s, "Package")
        # Returns DataFrame with columns: Package Length, Package Width, Package Height
    """
    # Normalize: keep only digits, dots, and 'x'
    s = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    
    # Extract exactly LxWxH pattern (numbers with optional decimals)
    # Unmatched rows will result in NaN
    dims = s.str.extract(
        r'(?i)^\s*(?P<L>\d+(?:\.\d+)?)x(?P<W>\d+(?:\.\d+)?)x(?P<H>\d+(?:\.\d+)?)\s*$'
    )
    
    # Convert to numeric
    for c in ["L", "W", "H"]:
        dims[c] = pd.to_numeric(dims[c], errors="coerce")
    
    # Rename columns with prefix
    dims = dims.rename(columns={
        "L": f"{col_prefix} Length",
        "W": f"{col_prefix} Width",
        "H": f"{col_prefix} Height",
    })
    
    return dims


def fmt_inch(v) -> str:
    """
    Format inch value for display.
    
    - Returns empty string if value is 0 or NaN
    - Returns integer string if value is a whole number
    - Returns decimal string with 1 decimal place otherwise
    
    Args:
        v: Numeric value in inches
        
    Returns:
        Formatted string representation
        
    Examples:
        >>> fmt_inch(10.0)
        '10'
        >>> fmt_inch(10.5)
        '10.5'
        >>> fmt_inch(0)
        ''
        >>> fmt_inch(None)
        ''
    """
    try:
        f = float(v)
        if np.isnan(f) or f == 0:
            return ""
        return str(int(f)) if f.is_integer() else f"{f:.1f}"
    except (ValueError, TypeError):
        return ""


def fmt_inch_triplet(L, W, H) -> str:
    """
    Format dimension triplet as "LxWxH" string.
    
    Args:
        L: Length in inches
        W: Width in inches
        H: Height in inches
        
    Returns:
        Formatted string like "10x 8x 6" or "" if any dimension is missing
        
    Examples:
        >>> fmt_inch_triplet(10, 8, 6)
        '10x 8x 6'
        >>> fmt_inch_triplet(10.5, 8.2, 6.0)
        '10.5x 8.2x 6'
        >>> fmt_inch_triplet(10, None, 6)
        ''
    """
    a = fmt_inch(L)
    b = fmt_inch(W)
    c = fmt_inch(H)
    return f"{a}x {b}x {c}" if (a and b and c) else ""


def to_cm(inches) -> Optional[float]:
    """
    Convert inches to centimeters.
    
    Args:
        inches: Value in inches
        
    Returns:
        Value in centimeters (rounded to 2 decimals) or None if invalid
        
    Examples:
        >>> to_cm(10)
        25.4
        >>> to_cm(None)
        None
    """
    try:
        return round(float(inches) * 2.54, 2)
    except (ValueError, TypeError):
        return None


def parse_date_safe(val):
    """
    Safely parse date value.
    
    Args:
        val: Date value (string, datetime, or None)
        
    Returns:
        date object or None if parsing fails or value is NaN
        
    Examples:
        >>> parse_date_safe("2025-01-15")
        datetime.date(2025, 1, 15)
        >>> parse_date_safe(None)
        None
    """
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None
