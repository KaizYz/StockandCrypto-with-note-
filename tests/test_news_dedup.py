from __future__ import annotations

import unittest

import pandas as pd

from src.ingestion.dedup import assign_dedup_groups


class NewsDedupTests(unittest.TestCase):
    def test_assign_dedup_groups_same_url_same_group(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "market": "crypto",
                    "symbol": "BTC",
                    "title": "Bitcoin jumps on ETF optimism",
                    "summary": "Short summary A",
                    "url": "https://example.com/a",
                    "available_at_utc": "2026-02-07 01:00:00+00:00",
                },
                {
                    "market": "crypto",
                    "symbol": "BTC",
                    "title": "Bitcoin jumps on ETF optimism (repost)",
                    "summary": "Short summary B",
                    "url": "https://example.com/a",
                    "available_at_utc": "2026-02-07 02:00:00+00:00",
                },
            ]
        )
        out = assign_dedup_groups(df, simhash_distance=3)
        self.assertEqual(len(out), 2)
        self.assertEqual(str(out.loc[0, "dedup_group_id"]), str(out.loc[1, "dedup_group_id"]))

    def test_assign_dedup_groups_distinct_title_different_group(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "market": "us_equity",
                    "symbol": "AAPL",
                    "title": "Apple beats earnings and raises guidance",
                    "summary": "Revenue growth is strong and margin expands",
                    "url": "https://news.example.com/1",
                    "available_at_utc": "2026-02-07 01:00:00+00:00",
                },
                {
                    "market": "us_equity",
                    "symbol": "AAPL",
                    "title": "Apple faces lawsuit and weak demand concerns",
                    "summary": "Guidance cut and demand uncertainty increases",
                    "url": "https://news.example.com/2",
                    "available_at_utc": "2026-02-07 01:05:00+00:00",
                },
            ]
        )
        out = assign_dedup_groups(df, simhash_distance=3)
        self.assertEqual(len(out), 2)
        self.assertNotEqual(str(out.loc[0, "dedup_group_id"]), str(out.loc[1, "dedup_group_id"]))


if __name__ == "__main__":
    unittest.main()
