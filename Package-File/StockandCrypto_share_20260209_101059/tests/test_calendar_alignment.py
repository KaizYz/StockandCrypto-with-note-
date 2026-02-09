from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.monitoring.calendar_alignment import _check_backtest_equity, _check_raw_branch


def _pick_check(rows: list[dict], check_name: str) -> dict:
    for row in rows:
        if str(row.get("check_name")) == check_name:
            return row
    raise AssertionError(f"missing check row: {check_name}")


class CalendarAlignmentTests(unittest.TestCase):
    def test_raw_branch_continuous_coverage_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "btc_hourly.csv"
            ts = pd.date_range("2026-01-01 00:00:00+00:00", periods=24, freq="h")
            pd.DataFrame({"timestamp_utc": ts.strftime("%Y-%m-%d %H:%M:%S%z")}).to_csv(raw, index=False)
            rows: list[dict] = []
            _check_raw_branch(raw_path=raw, interval="1h", is_crypto_continuous=True, rows=rows)
            coverage = _pick_check(rows, "continuous_coverage")
            self.assertTrue(bool(coverage["passed"]))

    def test_raw_branch_continuous_coverage_fail_on_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "btc_hourly_gap.csv"
            ts = pd.date_range("2026-01-01 00:00:00+00:00", periods=24, freq="h").delete([8, 9])
            pd.DataFrame({"timestamp_utc": ts.strftime("%Y-%m-%d %H:%M:%S%z")}).to_csv(raw, index=False)
            rows: list[dict] = []
            _check_raw_branch(raw_path=raw, interval="1h", is_crypto_continuous=True, rows=rows)
            coverage = _pick_check(rows, "continuous_coverage")
            self.assertFalse(bool(coverage["passed"]))

    def test_backtest_equity_detects_cn_weekend_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            (processed / "backtest").mkdir(parents=True, exist_ok=True)
            eq = pd.DataFrame(
                {
                    "timestamp_utc": ["2026-01-03 01:00:00+00:00", "2026-01-06 01:00:00+00:00"],
                    "market": ["cn_equity", "cn_equity"],
                }
            )
            eq.to_csv(processed / "backtest" / "equity.csv", index=False)
            rows: list[dict] = []
            _check_backtest_equity(processed, rows)
            weekend = _pick_check(rows, "cn_weekend_bar")
            self.assertFalse(bool(weekend["passed"]))


if __name__ == "__main__":
    unittest.main()
