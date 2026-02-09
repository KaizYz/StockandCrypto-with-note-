from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.news_features import _calc_features_at_ts, merge_news_features_asof


class NewsFeaturesTests(unittest.TestCase):
    def test_calc_features_event_risk_triggers_gate_block(self) -> None:
        query_ts = pd.Timestamp("2026-02-07 12:00:00+00:00")
        event_ts = np.array(
            [
                int((query_ts - pd.Timedelta(minutes=5)).value),
                int((query_ts - pd.Timedelta(minutes=10)).value),
                int((query_ts - pd.Timedelta(minutes=15)).value),
                int((query_ts - pd.Timedelta(minutes=20)).value),
            ],
            dtype=np.int64,
        )
        sentiment = np.array([0.9, 0.8, 0.95, 0.85], dtype=np.float64)
        base_weight = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        out = _calc_features_at_ts(
            event_ts_ns=event_ts,
            sentiment=sentiment,
            base_weight=base_weight,
            query_ts=query_ts,
            windows=[30, 120, 1440],
            tau_minutes=360.0,
        )
        self.assertTrue(bool(out["news_event_risk"]))
        self.assertFalse(bool(out["news_gate_pass"]))
        self.assertIn("POS_EVENT", str(out["news_reason_codes"]))

    def test_merge_news_features_asof_no_future_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news_dir = root / "news"
            news_dir.mkdir(parents=True, exist_ok=True)
            feat = pd.DataFrame(
                [
                    {
                        "market": "crypto",
                        "symbol": "BTC",
                        "timestamp_utc": "2026-02-07 00:00:00+00:00",
                        "news_score_120m": -0.20,
                    },
                    {
                        "market": "crypto",
                        "symbol": "BTC",
                        "timestamp_utc": "2026-02-07 02:00:00+00:00",
                        "news_score_120m": 0.80,
                    },
                ]
            )
            feat.to_csv(news_dir / "news_features_hourly.csv", index=False)

            df = pd.DataFrame(
                [
                    {
                        "market": "crypto",
                        "symbol": "BTCUSDT",
                        "timestamp_utc": "2026-02-07 01:00:00+00:00",
                    },
                    {
                        "market": "crypto",
                        "symbol": "BTCUSDT",
                        "timestamp_utc": "2026-02-07 03:00:00+00:00",
                    },
                ]
            )
            merged = merge_news_features_asof(
                df,
                ts_col="timestamp_utc",
                market_col="market",
                symbol_col="symbol",
                processed_dir=str(root),
            )
            self.assertEqual(len(merged), 2)
            self.assertAlmostEqual(float(merged.loc[0, "news_score_120m"]), -0.20, places=6)
            self.assertAlmostEqual(float(merged.loc[1, "news_score_120m"]), 0.80, places=6)


if __name__ == "__main__":
    unittest.main()

