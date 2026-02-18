"""Fiscal quarter utilities for earnings transcript fetching."""

from __future__ import annotations

from datetime import date

# Mapping from calendar quarter (1-indexed) to the most recently completed fiscal quarter.
# Calendar Q1 (Jan-Mar) → completed Q4 of previous year
# Calendar Q2 (Apr-Jun) → completed Q1 of same year
# Calendar Q3 (Jul-Sep) → completed Q2 of same year
# Calendar Q4 (Oct-Dec) → completed Q3 of same year
_COMPLETED_QUARTER: dict[int, int] = {1: 4, 2: 1, 3: 2, 4: 3}


def current_quarter() -> tuple[int, int]:
    """Return (quarter, year) for the most recently completed fiscal quarter.

    Used to determine which earnings transcript to fetch. Maps the current
    month to the most recently completed fiscal quarter.

    Returns:
        Tuple of (quarter_number, year) where quarter_number is 1-4.
    """
    today = date.today()
    calendar_quarter = (today.month - 1) // 3 + 1
    completed = _COMPLETED_QUARTER[calendar_quarter]
    year = today.year - 1 if calendar_quarter == 1 else today.year
    return (completed, year)
