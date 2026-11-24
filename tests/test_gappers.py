from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gappers  # noqa: E402


def base(n=260, price=50.0, volume=1_000_000):
    """Flat, quiet stock: close = price, 1-point daily range."""
    days = pd.bdate_range("2025-06-02", periods=n)
    return [{"datetime": int(d.timestamp()), "open": price, "high": price + 0.5,
             "low": price - 0.5, "close": price, "volume": volume} for d in days]


def gap_day(rows, index, open_, high, low, close, volume):
    rows[index].update(open=open_, high=high, low=low, close=close, volume=volume)
    return rows


class GapperScanTests(unittest.TestCase):
    def test_breakaway_gap_with_strong_close(self):
        rows = gap_day(base(), -1, 54, 56, 53.8, 55.8, 5_000_000)
        hit = gappers.scan_ticker(rows)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["Gap_Pct"], 8.0)
        self.assertEqual(hit["RVOL"], 5.0)
        self.assertEqual(hit["Sessions_Ago"], 0)
        self.assertEqual(hit["Gap_Type"], "Breakaway")
        self.assertEqual(hit["Close_Quality"], "Strong close")
        self.assertTrue(hit["Held_Gap_Intraday"])
        self.assertFalse(hit["Gap_Filled"])
        self.assertEqual(hit["Gap_Fill_Level"], 50.0)
        self.assertEqual(hit["AvgDollarVol50_M"], round((49 * 50e6 + 55.8 * 5e6) / 50 / 1e6, 1))

    def test_filters_reject_weak_gaps(self):
        cases = {
            "gap under 3%": (51, 52, 50.8, 51.8, 5_000_000),
            "rvol under 3": (54, 56, 53.8, 55.8, 2_000_000),
        }
        for name, bar in cases.items():
            with self.subTest(name):
                self.assertIsNone(gappers.scan_ticker(gap_day(base(), -1, *bar)))
        cheap = gap_day(base(price=10), -1, 11, 11.5, 10.9, 11.4, 5_000_000)
        self.assertIsNone(gappers.scan_ticker(cheap))
        thin = gap_day(base(volume=100_000), -1, 54, 56, 53.8, 55.8, 400_000)
        self.assertIsNone(gappers.scan_ticker(thin))

    def test_gap_down_ignored(self):
        self.assertIsNone(gappers.scan_ticker(gap_day(base(), -1, 45, 46, 44, 45.5, 5_000_000)))

    def test_faded_gap_that_closes_below_prior_close_is_filled(self):
        rows = gap_day(base(), -1, 54, 54.5, 49, 49.5, 5_000_000)
        hit = gappers.scan_ticker(rows)
        self.assertEqual(hit["Close_Quality"], "Faded")
        self.assertTrue(hit["Gap_Filled"])
        self.assertFalse(hit["Held_Gap_Intraday"])

    def test_later_close_below_gap_level_marks_filled(self):
        rows = gap_day(base(), -4, 54, 56, 53.8, 55.8, 5_000_000)
        for row in rows[-3:]:
            row.update(open=55, high=55.5, low=54.5, close=55)
        rows[-1].update(low=49, close=49.8)
        hit = gappers.scan_ticker(rows)
        self.assertEqual(hit["Sessions_Ago"], 3)
        self.assertTrue(hit["Gap_Filled"])

    def test_extended_stock_flags_exhaustion_risk(self):
        rows = base()
        for k, row in enumerate(rows[-30:-1]):  # steady run-up far above the 50-day
            p = 50 + 1.5 * (k + 1)
            row.update(open=p - 0.4, high=p + 0.5, low=p - 0.5, close=p)
        last = rows[-2]["close"]
        gap_day(rows, -1, last * 1.06, last * 1.09, last * 1.05, last * 1.08, 5_000_000)
        hit = gappers.scan_ticker(rows)
        self.assertEqual(hit["Gap_Type"], "Exhaustion risk")
        self.assertGreater(hit["Pre_Gap_ATR_Ext"], 4)

    def test_outside_lookback_and_short_history(self):
        rows = gap_day(base(), -15, 54, 56, 53.8, 55.8, 5_000_000)
        self.assertIsNone(gappers.scan_ticker(rows))
        self.assertIsNone(gappers.scan_ticker(base(n=40)))

    def test_view_merges_metadata_and_sorts_latest_first(self):
        older = gap_day(base(), -3, 54, 56, 53.8, 55.8, 9_000_000)
        newer = gap_day(base(), -1, 54, 56, 53.8, 55.8, 4_000_000)
        price_data = {"OLD": {"candles": older}, "NEW": {"candles": newer},
                      "FLAT": {"candles": base()}}
        stocks = pd.DataFrame({"Ticker": ["OLD", "NEW", "FLAT", "MISSING"],
                               "Sector": ["Tech"] * 4, "Industry": ["Software"] * 4,
                               "Percentile": [95, 80, 50, 10], "MarketCap": [1e9] * 4})
        frame = gappers.gapper_view(price_data, stocks)
        self.assertEqual(list(frame["Ticker"]), ["NEW", "OLD"])
        self.assertEqual(list(frame.columns), gappers.GAPPER_COLUMNS)
        self.assertEqual(frame.loc[0, "RS_Rating"], 80)

    def test_string_dates_are_accepted(self):
        rows = gap_day(base(), -1, 54, 56, 53.8, 55.8, 5_000_000)
        for row in rows:
            row["datetime"] = pd.Timestamp(row["datetime"], unit="s").strftime("%Y-%m-%d")
        self.assertIsNotNone(gappers.scan_ticker(rows))


if __name__ == "__main__":
    unittest.main()
