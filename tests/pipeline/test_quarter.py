"""Tests for fiscal quarter utilities."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from stock_radar.pipeline.quarter import current_quarter


class TestCurrentQuarter:
    """Tests for current_quarter() which returns the most recently completed fiscal quarter."""

    def _mock_today(self, year: int, month: int, day: int) -> date:
        """Create a date to use as mocked today."""
        return date(year, month, day)

    # --- Quarter boundary tests: one representative date per calendar quarter ---

    def test_january_returns_q4_previous_year(self) -> None:
        """Jan-Mar maps to Q4 of the previous year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 1, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (4, 2025)

    def test_april_returns_q1_same_year(self) -> None:
        """Apr-Jun maps to Q1 of the same year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 5, 10)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (1, 2026)

    def test_july_returns_q2_same_year(self) -> None:
        """Jul-Sep maps to Q2 of the same year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 8, 20)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (2, 2026)

    def test_october_returns_q3_same_year(self) -> None:
        """Oct-Dec maps to Q3 of the same year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 11, 5)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (3, 2026)

    # --- Edge case tests: exact boundary dates ---

    def test_jan_1_returns_q4_previous_year(self) -> None:
        """Jan 1 is still in Q1 calendar, so completed quarter is Q4 of previous year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 1, 1)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (4, 2025)

    def test_dec_31_returns_q3_same_year(self) -> None:
        """Dec 31 is still in Q4 calendar, so completed quarter is Q3 of same year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 12, 31)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (3, 2026)

    def test_march_31_returns_q4_previous_year(self) -> None:
        """March 31 is the last day of Q1 calendar, completed quarter is still Q4 prev year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 3, 31)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (4, 2025)

    def test_april_1_returns_q1_same_year(self) -> None:
        """April 1 is the first day of Q2 calendar, completed quarter is Q1 same year."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 4, 1)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            assert current_quarter() == (1, 2026)

    # --- Return type test ---

    def test_return_type_is_tuple_of_ints(self) -> None:
        """current_quarter() must return a tuple of two integers."""
        with patch("stock_radar.pipeline.quarter.date") as mock_date:
            mock_date.today.return_value = self._mock_today(2026, 6, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            result = current_quarter()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], int)
            assert isinstance(result[1], int)
