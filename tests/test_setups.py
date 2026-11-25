import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('setups_under_test', ROOT / 'setups.py')
setups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setups)

# The production blue-dot code, extracted without importing rs_ranking's
# side effects (same approach as the other test modules).
tree = ast.parse((ROOT / 'rs_ranking.py').read_text(encoding='utf-8'))
functions = {'rs_daily_frame', 'rs_frame_signals', 'rs_history_summary'}
constants = {'RS_SIGNAL_COLUMNS', 'RS_HISTORY_CORE_COLUMNS'}
parts = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                           for t in n.targets)]
ns = {'pd': pd, 'np': np}
exec(compile(ast.Module(body=parts, type_ignores=[]), '<rs helpers>', 'exec'), ns)

SETTINGS = dict(setups.DEFAULTS)


def make_candles(closes, start='2025-01-01', volume=None):
    dates = pd.bdate_range(start, periods=len(closes))
    volume = volume if volume is not None else [1_000_000] * len(closes)
    return [{'datetime': d.strftime('%Y-%m-%d'), 'open': c, 'high': c * 1.01,
             'low': c * 0.99, 'close': c, 'volume': v}
            for d, c, v in zip(dates, closes, volume)]


def flat_benchmark(n):
    return setups.daily_frame(make_candles([100.0] * n))


def leader_then_pullback(n=320, pullback=8, drop=0.006):
    """Steady leader; the last `pullback` sessions drift lower."""
    closes = [50 * 1.004 ** i for i in range(n - pullback)]
    for _ in range(pullback):
        closes.append(closes[-1] * (1 - drop))
    return closes


class BlueDotParityTests(unittest.TestCase):
    def check(self, closes, bench_closes):
        stock_c, bench_c = make_candles(closes), make_candles(bench_closes)
        mine_dots, _ = setups.blue_dots(setups.daily_frame(stock_c), setups.daily_frame(bench_c))
        prod = ns['rs_frame_signals'](ns['rs_daily_frame'](stock_c),
                                      ns['rs_daily_frame'](bench_c), '^GSPC')
        found = np.flatnonzero(mine_dots.to_numpy())
        self.assertEqual(int(len(mine_dots) - 1 - found[-1]) if len(found) else None,
                         prod['Days_Since_Dot'])
        self.assertEqual(int(mine_dots.tail(20).sum()), prod['Dots_Last_20_Days'])
        self.assertEqual(bool(mine_dots.iloc[-1]), prod['RS_Blue_Dot'])

    def test_same_dots_as_rs_ranking(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            stock = list(100 * np.cumprod(1 + rng.normal(0.001, 0.02, 400)))
            bench = list(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 400)))
            self.check(stock, bench)

    def test_same_dots_on_pullback(self):
        self.check(leader_then_pullback(), [100.0] * 320)


class SetupScanTests(unittest.TestCase):
    def test_pullback_after_dots_is_flagged(self):
        closes = leader_then_pullback()
        row = setups.scan_ticker(make_candles(closes), flat_benchmark(len(closes)), SETTINGS)
        self.assertEqual(row['Status'], 'OK')
        self.assertEqual(row['Days_Since_Dot'], 8)
        self.assertGreaterEqual(row['Dots_In_Window'], 3)
        self.assertTrue(row['EMA8W_Rising'])
        self.assertIn('21 EMA', row['Near_Support'])
        self.assertEqual(row['Setup'], 'Pullback')

    def test_fresh_dot_is_not_a_pullback(self):
        closes = [50 * 1.004 ** i for i in range(320)]
        row = setups.scan_ticker(make_candles(closes), flat_benchmark(320), SETTINGS)
        self.assertEqual(row['Days_Since_Dot'], 0)
        self.assertEqual(row['Setup'], '')

    def test_deep_pullback_away_from_support_is_not_flagged(self):
        closes = leader_then_pullback(pullback=10, drop=0.03)
        row = setups.scan_ticker(make_candles(closes), flat_benchmark(len(closes)), SETTINGS)
        self.assertEqual(row['Near_Support'], '')
        self.assertEqual(row['Setup'], '')

    def test_too_old_dot_is_not_flagged(self):
        closes = leader_then_pullback(pullback=30, drop=0.001)
        row = setups.scan_ticker(make_candles(closes), flat_benchmark(len(closes)), SETTINGS)
        self.assertEqual(row['Days_Since_Dot'], 30)
        self.assertEqual(row['Setup'], '')

    def test_mismatched_dates_give_status_not_signal(self):
        closes = leader_then_pullback()
        row = setups.scan_ticker(make_candles(closes), flat_benchmark(len(closes) - 1), SETTINGS)
        self.assertEqual(row, {'Status': 'LATEST_DATES_DIFFER'})

    def test_inside_day_and_hv1(self):
        closes = leader_then_pullback()
        volume = [1_000_000] * len(closes)
        volume[-30] = 9_000_000
        candles = make_candles(closes, volume=volume)
        candles[-1].update(high=candles[-2]['high'] * 0.999, low=candles[-2]['low'] * 1.001)
        row = setups.scan_ticker(candles, flat_benchmark(len(closes)), SETTINGS)
        self.assertTrue(row['Inside_Day'])
        self.assertEqual(row['HV1_Sessions_Ago'], 29)
        self.assertEqual(row['HV1_High'], round(candles[-30]['high'], 2))

    def test_weekly_ema_uses_weekly_closes(self):
        closes = pd.Series(range(1, 51), index=pd.bdate_range('2025-01-06', periods=50),
                           dtype=float)
        weekly = setups.weekly_ema(closes)
        self.assertEqual(len(weekly), 10)
        expected = closes.resample('W-FRI').last().ewm(span=8, adjust=False).mean()
        self.assertTrue(np.allclose(weekly.to_numpy(), expected.to_numpy()))


class SetupViewTests(unittest.TestCase):
    def test_view_filters_rs_and_sorts_setups_first(self):
        n = 320
        price_data = {
            '^GSPC': {'candles': make_candles([100.0] * n)},
            'PULL': {'candles': make_candles(leader_then_pullback(n))},
            'RUN': {'candles': make_candles([50 * 1.004 ** i for i in range(n)])},
            'WEAK': {'candles': make_candles([50.0] * n)},
        }
        stocks = pd.DataFrame([
            {'Ticker': 'RUN', 'Percentile': 99, 'Sector': 'S', 'Industry': 'I',
             'MarketCap': 1, 'RS_Before_Price_Weekly': ''},
            {'Ticker': 'PULL', 'Percentile': 90, 'Sector': 'S', 'Industry': 'I',
             'MarketCap': 1, 'RS_Before_Price_Weekly': '🔵'},
            {'Ticker': 'WEAK', 'Percentile': 10, 'Sector': 'S', 'Industry': 'I',
             'MarketCap': 1, 'RS_Before_Price_Weekly': ''},
        ])
        frame = setups.setups_view(price_data, stocks, '^GSPC', SETTINGS)
        self.assertEqual(list(frame.columns), setups.SETUP_COLUMNS)
        self.assertEqual(frame['Ticker'].tolist(), ['PULL', 'RUN'])
        self.assertEqual(frame.loc[0, 'Setup'], 'Pullback')
        self.assertEqual(frame.loc[0, 'Weekly_RS_Before_Price'], '🔵')


if __name__ == '__main__':
    unittest.main()
