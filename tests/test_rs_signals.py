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
        'rs_daily_frame', 'rs_line_signals', 'rs_frame_signals', 'rs_weekly_frame', 'rs_signals_view', 'rs_history_summary', 'rs_history_view')) or (
    isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
        'RS_SIGNAL_COLUMNS', 'RS_WEEKLY_COLUMNS', 'RS_VIEW_COLUMNS', 'RS_HISTORY_COLUMNS', 'RS_HISTORY_CORE_COLUMNS') for t in node.targets))]
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


class HistoricalSignalTests(unittest.TestCase):
    def summarize(self, age, recent=20, unknown_last=False):
        index = pd.bdate_range('2025-01-01', periods=400, tz='UTC')
        stock = pd.DataFrame({'close':100., 'high':101.}, index=index)
        events = pd.Series(False, index=index)
        events.iloc[-1-age] = True
        valid = pd.Series(True, index=index)
        if unknown_last:
            valid.iloc[-1] = False
        return ns['rs_history_summary'](stock, events, valid, events, valid, 250, recent)

    def test_daily_window_boundary_includes_19_bars_ago_not_20(self):
        for age, expected in ((0,True),(19,True),(20,False),(40,False)):
            result = self.summarize(age)
            self.assertEqual(result['RS_History_New_High'], expected)
            self.assertEqual(result['RS_History_Before_Price'], expected)
            self.assertEqual(result['RS_Bars_Since_New_High'], age)
            self.assertEqual(result['RS_History_Window_Bars'], 20)

    def test_weekly_window_boundary_includes_7_bars_ago_not_8(self):
        self.assertTrue(self.summarize(7, recent=8)['RS_History_New_High'])
        self.assertFalse(self.summarize(8, recent=8)['RS_History_New_High'])

    def test_observed_recent_event_survives_unknown_later_bar_but_age_is_unknown(self):
        result = self.summarize(4, unknown_last=True)
        self.assertTrue(result['RS_History_New_High'])
        self.assertIsNone(result['RS_Last_New_High_Date'])
        self.assertIsNone(result['RS_Bars_Since_New_High'])
        self.assertIn('RS_WINDOW_HAS_GAPS', result['RS_History_Status'])

    def test_no_observed_recent_event_and_a_gap_is_unknown_not_false(self):
        result = self.summarize(30, unknown_last=True)
        self.assertIsNone(result['RS_History_New_High'])
        self.assertIsNone(result['RS_History_Before_Price'])

    def test_current_false_but_historical_true(self):
        stock = candles(400)
        stock[-80]['high'] = 1500
        stock[-20]['close'] = 1000
        stock[-20]['high'] = 1100
        result = signals(stock, frame(candles(400, slope=0)), '^GSPC')
        self.assertFalse(result['RS_Blue_Dot'])
        self.assertFalse(result['RS_Before_Price'])
        self.assertTrue(result['Historical_RS_New_High_Daily'])
        self.assertTrue(result['Historical_RS_Before_Price_Daily'])
        self.assertEqual(result['Trading_Days_Since_RS_New_High'], 19)
        self.assertEqual(result['Trading_Days_Since_RS_Before_Price'], 19)
        self.assertEqual(result['Last_RS_New_High_Daily_Date'], stock[-20]['datetime'])

    def test_last_before_price_date_is_independent_of_plain_rs_date(self):
        stock = candles(400)
        stock[-80]['high'] = 1500
        stock[-1]['high'] = 2000
        result = signals(stock, frame(candles(400, slope=0)), '^GSPC')
        self.assertTrue(result['RS_Blue_Dot'])
        self.assertFalse(result['RS_Before_Price'])
        self.assertEqual(result['Trading_Days_Since_RS_New_High'], 0)
        self.assertEqual(result['Trading_Days_Since_RS_Before_Price'], 1)
        self.assertTrue(result['Historical_RS_Before_Price_Daily'])

    def test_latest_date_mismatch_does_not_flag_stale_recent_history(self):
        result = signals(candles(400)[:-1], frame(candles(400, slope=0)), '^GSPC')
        self.assertIsNone(result['Historical_RS_New_High_Daily'])
        self.assertEqual(result['Historical_Daily_Status'], 'LATEST_DATES_DIFFER')

    def test_short_ipo_history_is_visible(self):
        result = signals(candles(10), frame(candles(10, slope=0)), '^GSPC')
        self.assertTrue(result['Historical_RS_New_High_Daily'])
        self.assertEqual(result['Historical_Daily_Bars_Checked'], 10)
        self.assertIn('SHORT_HISTORY', result['Historical_Daily_Status'])
        self.assertIn('SHORT_HISTORY', result['Historical_Weekly_Status'])

    def test_past_events_use_their_own_rolling_high_not_todays_high(self):
        stock = candles(400)
        # Last session sets a fresh high above an older RS high, but the older
        # high must still remain an historical event in its own earlier window.
        stock[-10]['close'] = 1000
        stock[-10]['high'] = 1100
        stock[-1]['close'] = 2000
        stock[-1]['high'] = 2100
        result = signals(stock, frame(candles(400, slope=0)), '^GSPC')
        self.assertTrue(result['Historical_RS_New_High_Daily'])
        # Drop today's bar: the earlier event is then the last historical high.
        earlier = signals(stock[:-1], frame(candles(400, slope=0)[:-1]), '^GSPC')
        self.assertTrue(earlier['Historical_RS_New_High_Daily'])
        self.assertEqual(earlier['Last_RS_New_High_Daily_Date'], stock[-10]['datetime'])

    def test_daily_and_weekly_histories_against_independent_reference(self):
        rng = np.random.default_rng(122)
        dates = pd.bdate_range('2025-01-01', periods=400)
        prices = 100 * np.exp(np.cumsum(rng.normal(.001,.025,len(dates))))
        benchmarks = 100 * np.exp(np.cumsum(rng.normal(.001,.01,len(dates))))
        stock = [{'datetime':str(d.date()), 'close':float(p), 'high':float(p*1.04)}
                 for d,p in zip(dates,prices)]
        bench = [{'datetime':str(d.date()), 'close':float(p), 'high':float(p*1.01)}
                 for d,p in zip(dates,benchmarks)]
        result = signals(stock, frame(bench), '^GSPC')
        daily = [(s['datetime'],s['close']/b['close'],s['high']) for s,b in zip(stock,bench)]
        grouped = {}
        for record in daily:
            date = pd.Timestamp(record[0])
            week = date - pd.Timedelta(days=date.weekday())
            previous = grouped.get(week)
            grouped[week] = (record[0], record[1], max(record[2],previous[2]) if previous else record[2])
        for timeframe, records, lookback, recent, unit in (
            ('Daily',daily,250,20,'Trading_Days'),
            ('Weekly',list(grouped.values()),52,8,'Weekly_Bars')):
            plain, before = [], []
            for i, (_,line,high) in enumerate(records):
                window = records[max(0,i-lookback+1):i+1]
                at_high = line == max(r[1] for r in window)
                plain.append(at_high)
                before.append(at_high and high < max(r[2] for r in window))
            for kind, flags in (('New_High',plain),('Before_Price',before)):
                self.assertEqual(result[f'Historical_RS_{kind}_{timeframe}'], any(flags[-recent:]))
                found = [i for i, value in enumerate(flags) if value]
                if found:
                    last = found[-1]
                    self.assertEqual(result[f'Last_RS_{kind}_{timeframe}_Date'], records[last][0])
                    self.assertEqual(result[f'{unit}_Since_RS_{kind}'],len(records)-1-last)

    def test_history_sidecar_keeps_legacy_rs_view_at_24_columns(self):
        stock = candles(400)
        stock[-10]['high'] = 1000
        result = signals(stock, frame(candles(400,slope=0)), '^GSPC')
        stocks = pd.DataFrame([{'Rank':1,'Ticker':'NA'}])
        legacy = ns['rs_signals_view'](stocks, {'NA':result})
        history = ns['rs_history_view'](stocks, {'NA':result})
        self.assertEqual(len(legacy.columns),24)
        self.assertEqual(len(history.columns),18)
        self.assertEqual(history.loc[0,'Ticker'],'NA')
        self.assertEqual(history.loc[0,'Historical_RS_New_High_Daily'],'🔵')
        self.assertEqual(history.loc[0,'History_As_Of'],stock[-1]['datetime'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
