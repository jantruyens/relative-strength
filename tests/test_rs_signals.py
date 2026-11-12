import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPDATED = ROOT / 'rs_ranking.py'
tree = ast.parse(UPDATED.read_text(encoding='utf-8'))
parts = [node for node in tree.body if (
    isinstance(node, ast.FunctionDef) and node.name in (
        'rs_daily_frame', 'rs_line_signals', 'rs_frame_signals', 'rs_weekly_frame', 'rs_signals_view')) or (
    isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
        'RS_SIGNAL_COLUMNS', 'RS_WEEKLY_COLUMNS', 'RS_VIEW_COLUMNS') for t in node.targets))]
ns = {'pd': pd, 'np': np}
exec(compile(ast.Module(body=parts, type_ignores=[]), '<signals>', 'exec'), ns)
frame = ns['rs_daily_frame']
signals = ns['rs_line_signals']


def candles(n=300, slope=1):
    return [{'datetime': date.strftime('%Y-%m-%d'), 'close': 100 + i * slope,
             'high': 101 + i * slope, 'volume': 1000}
            for i, date in enumerate(pd.bdate_range('2024-01-01', periods=n))]


class SignalTests(unittest.TestCase):

    def evaluate(self, stock=None, benchmark=None):
        return signals(stock if stock is not None else candles(), frame(benchmark if benchmark is not None else candles(slope=0)), 'TEST')

    def test_new_rs_high_before_intraday_price_high(self):
        stock = candles()
        stock[-10]['high'] = 500
        out = self.evaluate(stock)
        self.assertTrue(out['RS_Blue_Dot'])
        self.assertTrue(out['RS_Before_Price'])
        self.assertEqual(out['Days_Since_Dot'], 0)
        self.assertEqual(out['Dots_Last_20_Days'], 20)
        self.assertEqual(out['RS_Bars_Used'], 250)
        self.assertEqual(out['RS_Signal_Status'], 'OK')

    def test_tied_rs_high_counts_but_tied_price_high_does_not(self):
        stock = candles(slope=0)
        out = self.evaluate(stock)
        self.assertTrue(out['RS_Blue_Dot'])
        self.assertFalse(out['RS_Before_Price'])

    def test_250_window_includes_today_and_excludes_250_bars_ago(self):
        stock = candles()
        stock[-251]['close'] = 10000
        stock[-251]['high'] = 10001
        self.assertTrue(self.evaluate(stock)['RS_Blue_Dot'])
        stock[-250]['close'] = 10000
        stock[-250]['high'] = 10001
        self.assertFalse(self.evaluate(stock)['RS_Blue_Dot'])

    def test_days_since_last_dot(self):
        stock = candles()
        for candle in stock[-3:]:
            candle['close'] = 350
            candle['high'] = 351
        out = self.evaluate(stock)
        self.assertFalse(out['RS_Blue_Dot'])
        self.assertEqual(out['Days_Since_Dot'], 3)
        self.assertEqual(out['Dots_Last_20_Days'], 17)

    def test_date_alignment_does_not_zip_rows(self):
        benchmark = candles(slope=0)
        del benchmark[-20]
        out = self.evaluate(benchmark=benchmark)
        self.assertIsNone(out['RS_Blue_Dot'])
        self.assertEqual(out['RS_Signal_Status'], 'MISSING_RS_WINDOW_DATA')

    def test_latest_session_mismatch_is_explicit(self):
        out = self.evaluate(stock=candles()[:-1])
        self.assertIsNone(out['RS_Before_Price'])
        self.assertEqual(out['RS_Signal_Status'], 'LATEST_DATES_DIFFER')

    def test_missing_high_is_unknown_not_false(self):
        stock = candles()
        stock[-5].pop('high')
        out = self.evaluate(stock)
        self.assertTrue(out['RS_Blue_Dot'])
        self.assertIsNone(out['RS_Before_Price'])
        self.assertEqual(out['RS_Signal_Status'], 'MISSING_HIGH_WINDOW_DATA')

    def test_short_history_matches_pine_available_bars(self):
        out = self.evaluate(candles(120), candles(120, slope=0))
        self.assertTrue(out['RS_Blue_Dot'])
        self.assertEqual(out['RS_Bars_Used'], 120)
        self.assertEqual(out['RS_Signal_Status'], 'SHORT_HISTORY')

    def test_unordered_candles_and_epoch_milliseconds(self):
        stock = candles()
        expected = self.evaluate(stock)
        for row in stock:
            row['datetime'] = int(pd.Timestamp(row['datetime'], tz='UTC').timestamp() * 1000)
        self.assertEqual(self.evaluate(list(reversed(stock))), expected)

    def test_invalid_dates_and_duplicates(self):
        stock = candles()
        stock[-1]['datetime'] = stock[-2]['datetime']
        self.assertEqual(self.evaluate(stock)['RS_Signal_Status'], 'DUPLICATE_DAILY_CANDLES')
        stock[-1]['datetime'] = 'invalid'
        self.assertEqual(self.evaluate(stock)['RS_Signal_Status'], 'INVALID_CANDLE_DATES')
        self.assertEqual(self.evaluate([{'close': 10}])['RS_Signal_Status'], 'MISSING_CANDLE_DATES')

    def test_missing_benchmark(self):
        self.assertEqual(signals(candles(), None, 'SPX')['RS_Signal_Status'], 'BENCHMARK_UNAVAILABLE')

    def test_reference_against_literal_pine_rules(self):
        rng = np.random.default_rng(17)
        stock = candles(400)
        bench = candles(400)
        for rows in (stock, bench):
            prices = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(rows))))
            for row, price in zip(rows, prices):
                row['close'] = float(price)
                row['high'] = float(price * rng.uniform(1, 1.03))
        for endpoint in (120, 250, 251, 300, 400):
            ratios = [s['close'] / b['close'] for s, b in zip(stock[:endpoint], bench[:endpoint])]
            flags = [value == max(ratios[max(0, i - 249):i + 1]) for i, value in enumerate(ratios)]
            out = self.evaluate(stock[:endpoint], bench[:endpoint])
            self.assertEqual(out['RS_Blue_Dot'], flags[-1])
            self.assertEqual(out['RS_Before_Price'], flags[-1] and stock[endpoint - 1]['high'] < max((r['high'] for r in stock[max(0, endpoint - 250):endpoint])))
            self.assertEqual(out['Dots_Last_20_Days'], sum(flags[-20:]))
            self.assertEqual(out['Days_Since_Dot'], len(flags) - 1 - max((i for i, flag in enumerate(flags) if flag)))


class ProjectIntegrationTests(unittest.TestCase):
    def data_functions(self, benchmark='^GSPC', all_stocks=True):
        source = ast.parse((ROOT / 'rs_data.py').read_text(encoding='utf-8-sig'))
        selected = [node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('escape_ticker', 'parse_batch_download',
                                      'candles_from_df_row', 'get_resolved_securities')]
        namespace = {'pd': pd, 'REFERENCE_TICKER': 'SPY',
                     'RS_LINE_REFERENCE_TICKER': benchmark, 'ALL_STOCKS': all_stocks,
                     'REF_TICKER': {'ticker': 'SPY'}}
        def tickers(initial):
            return {**initial, 'TEST': {'ticker': 'TEST'}}
        namespace['get_tickers_from_nasdaq'] = tickers
        namespace['get_tickers_from_wikipedia'] = tickers
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<data>', 'exec'), namespace)
        return namespace

    def test_separate_index_is_downloaded_for_both_universe_modes(self):
        for all_stocks in (True, False):
            resolved = self.data_functions(all_stocks=all_stocks)['get_resolved_securities']()
            self.assertIn('^GSPC', resolved)
            self.assertIn('SPY', resolved)
            self.assertIn('TEST', resolved)

    def test_shared_benchmark_is_not_duplicated(self):
        self.assertEqual(set(self.data_functions('SPY')['get_resolved_securities']()), {'SPY', 'TEST'})

    def test_real_multi_ticker_candle_parser_feeds_signal(self):
        namespace = self.data_functions()
        index = pd.bdate_range('2024-01-01', periods=300)
        raw = pd.DataFrame(index=index)
        for ticker in ('TEST', '^GSPC'):
            close = 100 + np.arange(300) if ticker == 'TEST' else np.full(300, 100)
            for field in ('Open', 'Close', 'Low', 'High'):
                raw[(field, ticker)] = close.astype(float)
            raw[('Volume', ticker)] = 1000
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        raw.loc[index[-20], ('High', 'TEST')] = 500
        parsed = namespace['parse_batch_download'](raw, ['TEST', '^GSPC'])
        self.assertEqual(parsed['TEST'][-1]['datetime'], int(index[-1].timestamp()))
        result = signals(parsed['TEST'], frame(parsed['^GSPC']), '^GSPC')
        self.assertTrue(result['RS_Before_Price'])
        self.assertEqual(result['RS_Signal_Status'], 'OK')

    def test_flat_single_ticker_parser(self):
        index = pd.bdate_range('2024-01-01', periods=3)
        raw = pd.DataFrame({'Open': [1., 2., 3.], 'Close': [1., 2., 3.],
                            'Low': [1., 2., 3.], 'High': [2., 3., 4.],
                            'Volume': [100., 100., 100.]}, index=index)
        parsed = self.data_functions()['parse_batch_download'](raw, ['TEST'])
        self.assertEqual(len(parsed['TEST']), 3)
        self.assertEqual(frame(parsed['TEST']).index[-1], index[-1].tz_localize('UTC'))


class WeeklySignalTests(unittest.TestCase):
    def evaluate(self, stock, bench=None):
        if bench is None:
            bench = [dict(row, close=100, high=101) for row in stock]
        return signals(stock, frame(bench), '^GSPC')

    def weekly_candles(self, weeks=60, end='2026-11-11'):
        dates = pd.bdate_range(end=end, periods=weeks * 5)
        rows = []
        for i, date in enumerate(dates):
            rows.append({'datetime': date.strftime('%Y-%m-%d'),
                         'close': 100 + i, 'high': 101 + i})
        return rows

    def test_weekly_before_price_uses_weekly_intraday_high(self):
        stock = self.weekly_candles()
        stock[-25]['high'] = 1000
        result = self.evaluate(stock)
        self.assertTrue(result['RS_New_High_Weekly'])
        self.assertTrue(result['RS_Before_Price_Weekly'])
        self.assertEqual(result['RS_Weekly_Bars_Used'], 52)
        self.assertEqual(result['RS_Weekly_Signal_Date'], '2026-11-11')
        self.assertEqual(result['RS_Weekly_Week_Start'], '2026-11-09')
        self.assertEqual(result['RS_Weekly_Signal_Status'], 'OK')

    def test_weekly_close_is_not_highest_daily_ratio(self):
        stock = self.weekly_candles(end='2026-11-13')
        # A midweek spike is below the 250-daily-bar high now, but is not a
        # previous weekly CLOSE high. A weekly signal still qualifies.
        stock[-8]['close'] = 2000
        stock[-8]['high'] = 2001
        result = self.evaluate(stock)
        self.assertFalse(result['RS_Blue_Dot'])
        self.assertTrue(result['RS_New_High_Weekly'])
        self.assertTrue(result['RS_Before_Price_Weekly'])

    def test_weekly_is_not_a_daily_signal_copied_over(self):
        stock = self.weekly_candles(end='2026-11-13')
        # A weekly closing high 51 weeks ago falls outside the 250-day window.
        stock[-256]['close'] = 2000
        stock[-256]['high'] = 2001
        result = self.evaluate(stock)
        self.assertTrue(result['RS_Blue_Dot'])
        self.assertFalse(result['RS_New_High_Weekly'])

    def test_weekly_ties_count_and_tied_price_high_excludes_before_price(self):
        stock = self.weekly_candles()
        stock = [dict(row, close=100, high=101) for row in stock]
        result = self.evaluate(stock)
        self.assertTrue(result['RS_New_High_Weekly'])
        self.assertFalse(result['RS_Before_Price_Weekly'])

    def test_weekly_resampling_preserves_last_session_missing_close(self):
        stock = self.weekly_candles(end='2026-11-13')
        stock[-1]['close'] = None
        result = self.evaluate(stock)
        self.assertIsNone(result['RS_New_High_Weekly'])
        self.assertEqual(result['RS_Weekly_Signal_Status'], 'MISSING_RS_WINDOW_DATA')

    def test_weekly_high_unknown_when_one_daily_high_is_unknown(self):
        stock = self.weekly_candles()
        stock[-20]['high'] = None
        result = self.evaluate(stock)
        self.assertTrue(result['RS_New_High_Weekly'])
        self.assertIsNone(result['RS_Before_Price_Weekly'])
        self.assertEqual(result['RS_Weekly_Signal_Status'], 'MISSING_HIGH_WINDOW_DATA')

    def test_previous_week_close_dates_must_match(self):
        stock = self.weekly_candles(end='2026-11-13')
        # A missing Friday stock session must not compare Thursday's stock
        # close against Friday's benchmark close, even with valid current data.
        benchmark = [dict(row, close=100, high=101) for row in stock]
        del stock[-6]
        result = self.evaluate(stock, benchmark)
        self.assertIsNone(result['RS_New_High_Weekly'])
        self.assertEqual(result['RS_Weekly_Signal_Status'], 'MISSING_RS_WINDOW_DATA')

    def test_matching_thursday_holiday_week_close_is_valid(self):
        stock = self.weekly_candles(end='2026-11-13')
        del stock[-6]
        # Both series omit Friday (e.g. a holiday): Thursday is their common
        # closing session. No external holiday-calendar assumption is needed.
        result = self.evaluate(stock)
        self.assertTrue(result['RS_New_High_Weekly'])
        self.assertEqual(result['RS_Weekly_Signal_Status'], 'OK')

    def test_short_weekly_history_is_marked(self):
        result = self.evaluate(self.weekly_candles(25))
        self.assertTrue(result['RS_New_High_Weekly'])
        self.assertEqual(result['RS_Weekly_Signal_Status'], 'SHORT_HISTORY')
        self.assertLess(result['RS_Weekly_Bars_Used'], 52)

    def test_reference_rule_for_52_week_window(self):
        rng = np.random.default_rng(92)
        dates = pd.bdate_range('2025-01-01', periods=400)
        stock, benchmark = [], []
        sp = 100 * np.exp(np.cumsum(rng.normal(.002, .025, len(dates))))
        bp = 100 * np.exp(np.cumsum(rng.normal(.001, .01, len(dates))))
        for i, date in enumerate(dates):
            stock.append({'datetime': str(date.date()), 'close': sp[i], 'high': sp[i] * 1.02})
            benchmark.append({'datetime': str(date.date()), 'close': bp[i], 'high': bp[i] * 1.01})
        for length in (120, 260, 300, 400):
            # Independent weekly grouping: calendar Monday keys, last close,
            # maximum daily high, and current partial week retained.
            groups = {}
            for s, b in zip(stock[:length], benchmark[:length]):
                date = pd.Timestamp(s['datetime'])
                monday = date - pd.Timedelta(days=date.weekday())
                group = groups.setdefault(monday, {'high': s['high']})
                group.update(ratio=s['close']/b['close'], high=max(group['high'], s['high']))
            weeks = list(groups.values())[-52:]
            expected = weeks[-1]['ratio'] == max(w['ratio'] for w in weeks)
            result = self.evaluate(stock[:length], benchmark[:length])
            self.assertEqual(result['RS_New_High_Weekly'], expected)
            self.assertEqual(result['RS_Before_Price_Weekly'], expected and weeks[-1]['high'] < max(w['high'] for w in weeks))

    def test_four_dot_view_column_order_and_unknowns(self):
        stock = self.weekly_candles()
        stock[-10]['high'] = 1000
        result = self.evaluate(stock)
        row = {'Rank': 1, 'Ticker': 'TEST', 'Price': 399, 'Percentile': 99,
               'Sector': 'Technology', 'Industry': 'Software', 'Exchange': 'NASDAQ',
               'RS_Benchmark': '^GSPC', 'Days_Since_Dot': 0, 'Dots_Last_20_Days': 20}
        view = ns['rs_signals_view'](pd.DataFrame([row]), {'TEST': result})
        self.assertEqual(view.columns[4:8].tolist(), ['RS_Before_Price_Daily',
            'RS_Before_Price_Weekly', 'RS_New_High_Daily', 'RS_New_High_Weekly'])
        self.assertEqual(view.iloc[0, 4:8].tolist(), ['🔵'] * 4)
        self.assertEqual(len(view.columns), 24)
        result['RS_Before_Price_Weekly'] = None
        result['RS_Weekly_Signal_Status'] = 'MISSING_HIGH_WINDOW_DATA'
        view = ns['rs_signals_view'](pd.DataFrame([row]), {'TEST': result})
        self.assertEqual(view.iloc[0]['RS_Before_Price_Weekly'], '')
        self.assertEqual(view.iloc[0]['Weekly_Status'], 'MISSING_HIGH_WINDOW_DATA')


if __name__ == '__main__':
    unittest.main(verbosity=2)
