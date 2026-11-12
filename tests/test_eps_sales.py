"""Offline financial-data tests; no Yahoo network calls or rs_data imports."""
from datetime import date, datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eps_sales_under_test', ROOT / 'eps_sales.py')
fund = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fund)
NOW = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)


def statement():
    # Deliberately unsorted; source includes a negative and a genuine zero.
    ends = pd.to_datetime(['2026-03-31','2026-09-30','2025-09-30','2026-06-30','2025-12-31'])
    return pd.DataFrame([[.6, 1.5, 1., -0.2, 0.],
                         [60e6, 150e6, 100e6, 80e6, 40e6]],
                        index=['DilutedEPS', 'TotalRevenue'], columns=ends)


def entry():
    return {'periods': fund.normalize_statement(statement(), NOW.date()), 'currency': 'USD',
            'last_success': fund.timestamp(NOW), 'last_attempt': fund.timestamp(NOW), 'last_error': None}


def stock_frame(count=10):
    return pd.DataFrame([{'Rank': i+1, 'Ticker': f'T{i}', 'MarketCap': 1e9,
                          **{c: '🔵' if i < 8 else None for c in fund.SIGNAL_COLUMNS}}
                         for i in range(count)])


class FundamentalsTests(unittest.TestCase):
    def test_exact_diluted_eps_and_revenue_with_sorted_dates(self):
        periods = fund.normalize_statement(statement(), NOW.date())
        self.assertEqual(len(periods), 5)
        self.assertEqual(periods[0], {'end':'2026-09-30', 'eps_diluted':1.5, 'sales':150e6})
        self.assertEqual(periods[1]['eps_diluted'], -.2)
        self.assertEqual(periods[3]['eps_diluted'], 0)

    def test_no_basic_eps_fallback(self):
        source = statement().rename(index={'DilutedEPS':'BasicEPS'})
        periods = fund.normalize_statement(source, NOW.date())
        self.assertTrue(all(p['eps_diluted'] is None for p in periods))
        output = fund.financial_columns(dict(entry(), periods=periods), NOW.date())
        self.assertIsNone(output['EPS_Diluted_Q0'])
        self.assertIn('EPS_MISSING', output['Financials_Period_Status'])

    def test_only_latest_yoy_and_millions(self):
        output = fund.financial_columns(entry(), NOW.date())
        self.assertEqual(output['EPS_YoY_Latest_Pct'], 50)
        self.assertEqual(output['Sales_YoY_Latest_Pct'], 50)
        self.assertEqual(output['Sales_Millions_Q0'], 150)
        self.assertEqual(output['Quarter_End_Q4'], '2025-09-30')
        self.assertEqual(output['Financials_Period_Status'], 'OK')
        self.assertNotIn('EPS_YoY_Q1', output)

    def test_negative_base_uses_absolute_denominator(self):
        self.assertEqual(fund.growth(-.5, -1), 50)
        self.assertEqual(fund.growth(1, -.5), 300)
        self.assertEqual(fund.growth(-1, 1), -200)

    def test_zero_or_missing_base_does_not_become_zero_growth(self):
        self.assertIsNone(fund.growth(2, 0))
        self.assertIsNone(fund.growth(None, 1))
        data = entry()
        data['periods'][-1]['eps_diluted'] = 0
        output = fund.financial_columns(data, NOW.date())
        self.assertIsNone(output['EPS_YoY_Latest_Pct'])
        self.assertIn('ZERO_BASE_EPS', output['Financials_Period_Status'])

    def test_missing_quarter_leaves_gap_instead_of_shifting(self):
        data = entry()
        data['periods'] = [p for p in data['periods'] if p['end'] != '2026-06-30']
        output = fund.financial_columns(data, NOW.date())
        self.assertIsNone(output['Quarter_End_Q1'])
        self.assertEqual(output['Quarter_End_Q2'], '2026-03-31')
        self.assertEqual(output['EPS_YoY_Latest_Pct'], 50)
        self.assertIn('MISSING_QUARTER', output['Financials_Period_Status'])

    def test_short_history_no_false_yoy(self):
        data = dict(entry(), periods=entry()['periods'][:3])
        output = fund.financial_columns(data, NOW.date())
        self.assertIsNone(output['EPS_YoY_Latest_Pct'])
        self.assertIn('YOY_BASE_MISSING', output['Financials_Period_Status'])
        self.assertEqual(output['Financials_Quarters_Available'], 3)

    def test_52_53_week_fiscal_dates_match(self):
        data = entry()
        for p, end in zip(data['periods'], ['2026-09-26','2026-06-27','2026-03-28','2025-12-27','2025-09-27']):
            p['end'] = end
        output = fund.financial_columns(data, NOW.date())
        self.assertEqual(output['Quarter_End_Q4'], '2025-09-27')
        self.assertEqual(output['EPS_YoY_Latest_Pct'], 50)

    def test_wrong_previous_year_period_not_used(self):
        data = entry()
        data['periods'][-1]['end'] = '2025-08-15'
        output = fund.financial_columns(data, NOW.date())
        self.assertIsNone(output['EPS_YoY_Latest_Pct'])

    def test_no_currency_guessing_or_currency_unknown_growth(self):
        data = dict(entry(), currency=None)
        output = fund.financial_columns(data, NOW.date())
        self.assertIsNone(output['Financial_Currency'])
        self.assertIsNone(output['EPS_YoY_Latest_Pct'])
        self.assertIn('CURRENCY_UNKNOWN', output['Financials_Period_Status'])

    def test_nonfinite_future_and_ttm_values(self):
        source = statement()
        source.loc['DilutedEPS', pd.Timestamp('2026-09-30')] = float('inf')
        source['TTM'] = [99,99]
        source[pd.Timestamp('2027-03-31')] = [99,99]
        periods = fund.normalize_statement(source, NOW.date())
        self.assertEqual(len(periods), 5)
        self.assertIsNone(periods[0]['eps_diluted'])
        self.assertEqual(periods[0]['end'], '2026-09-30')

    def test_empty_duplicate_and_unusable_data_fail_explicitly(self):
        for source in (pd.DataFrame(), pd.DataFrame([[1]], index=['BasicEPS'], columns=[pd.Timestamp('2026-09-30')])):
            with self.assertRaises(ValueError):
                fund.normalize_statement(source, NOW.date())
        source = statement()
        source.columns = [pd.Timestamp('2026-09-30')] * 5
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_FISCAL_PERIODS'):
            fund.normalize_statement(source, NOW.date())
        source = statement()
        source[pd.Timestamp('2026-09-29')] = [2, 200e6]
        with self.assertRaisesRegex(ValueError, 'OVERLAPPING'):
            fund.normalize_statement(source, NOW.date())

    def test_yahoo_adapter_queries_quarterly_and_financial_currency(self):
        seen = []
        class FakeTicker:
            def __init__(self, symbol): seen.append(symbol)
            def get_income_stmt(self, **kwargs):
                seen.append(kwargs)
                return statement()
            def get_info(self): return {'financialCurrency':'eur', 'currency':'USD'}
        result = fund.fetch_financials('BRK.B', FakeTicker, NOW.date())
        self.assertEqual(seen, ['BRK-B', {'freq':'quarterly'}])
        self.assertEqual(result['currency'], 'EUR')

    def test_no_quote_currency_fallback(self):
        class FakeTicker:
            def __init__(self, symbol): pass
            def get_income_stmt(self, **kwargs): return statement()
            def get_info(self): return {'currency':'USD'}
        self.assertIsNone(fund.fetch_financials('TEST', FakeTicker, NOW.date())['currency'])

    def test_existing_rs_columns_and_row_order_preserved(self):
        stocks = stock_frame()
        result = fund.append_financials(stocks, {'tickers':{'T0':entry()}}, NOW.date())
        pd.testing.assert_frame_equal(stocks, result[stocks.columns])
        self.assertEqual(result.iloc[0]['EPS_Diluted_Q0'], 1.5)
        self.assertEqual(result.iloc[1]['Financials_Refresh_Status'], 'NOT_FETCHED')
        self.assertTrue(pd.isna(result.iloc[1]['Quarter_End_Q0']))
        with self.assertRaises(ValueError):
            fund.append_financials(pd.concat([stocks, stocks.iloc[[0]]]), {'tickers':{}})

    def test_refresh_queue_reserves_general_coverage(self):
        stocks = stock_frame()
        cache = {'version':1, 'tickers':{}}
        chosen = fund.select_refresh(stocks, cache, NOW, 5)
        # 4 of the five slots are dot priority; another is oldest general.
        self.assertEqual(len(chosen), 5)
        cache['tickers'] = {f'T{i}':entry() for i in range(8)}
        chosen = fund.select_refresh(stocks, cache, NOW, 5)
        self.assertEqual(chosen, ['T8','T9'])
        # When every dot is due, an OLD general row outranks fresher dots
        # for the reserved slot.
        cache['tickers'] = {f'T{i}': dict(entry(), last_success=fund.timestamp(NOW-timedelta(days=4))) for i in range(8)}
        self.assertIn('T8', fund.select_refresh(stocks, cache, NOW, 5))

    def test_recent_failed_attempt_does_not_hammer_provider(self):
        data = dict(entry(), last_success=None, last_error='RATE_LIMIT')
        cache = {'tickers':{'T0':data}}
        self.assertEqual(fund.select_refresh(stock_frame(1), cache, NOW, 5), [])

    def test_failed_refresh_keeps_old_values_and_marks_status(self):
        cache = {'version':1, 'tickers':{'T0':dict(entry(), last_success=fund.timestamp(NOW-timedelta(days=4)))}}
        def failing(*args, **kwargs): raise RuntimeError('Temporary issue')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.json'
            fund.run_refresh(stock_frame(1), cache, path, fetcher=failing, now=NOW, pause=lambda _:None)
            saved = fund.load_cache(path)
            output = fund.financial_columns(saved['tickers']['T0'], NOW.date())
            self.assertEqual(output['EPS_Diluted_Q0'], 1.5)
            self.assertEqual(output['Financials_Refresh_Status'], 'ERROR_CACHED')

    def test_rate_limit_stops_batch_without_destroying_cache(self):
        class YFRateLimitError(Exception): pass
        def limited(*args, **kwargs): raise YFRateLimitError()
        cache = {'version':1, 'tickers':{}}
        with tempfile.TemporaryDirectory() as directory:
            count = fund.run_refresh(stock_frame(), cache, Path(directory)/'cache.json', fetcher=limited,
                                     now=NOW, pause=lambda _:None)
        self.assertEqual(count, 1)
        self.assertEqual(cache['tickers']['T0']['last_error'], 'RATE_LIMIT')

    def test_budget_and_cache_roundtrip(self):
        cache = {'version':1, 'tickers':{}}
        timer = iter([0,0,100,100])
        def good(*args, **kwargs): return {'periods':entry()['periods'], 'currency':'USD'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.json'
            count = fund.run_refresh(stock_frame(), cache, path, budget_seconds=10, fetcher=good,
                                     now=NOW, monotonic=lambda:next(timer), pause=lambda _:None)
            self.assertEqual(count, 1)
            self.assertEqual(fund.load_cache(path), cache)
            path.write_text('{"version":99,"tickers":{}}')
            with self.assertRaises(ValueError): fund.load_cache(path)

    def test_csv_pipeline_retains_24_rs_columns_and_adds_26_financial_columns(self):
        columns = ['Rank','Ticker','Price','Percentile', *fund.SIGNAL_COLUMNS,
                   'Sector','Industry','Exchange','PctFrom52WkHigh','MarketCap',
                   'AvgVol50','RS_Benchmark','Daily_Signal_Date','Daily_Bars_Used',
                   'Daily_Status','Weekly_Signal_Date','Weekly_Week_Start',
                   'Weekly_Bars_Used','Weekly_Status','Days_Since_Dot','Dots_Last_20_Days']
        rows = []
        for i in range(3):
            row = dict.fromkeys(columns)
            row.update(Rank=i+1, Ticker=f'T{i}', Price=50+i, Percentile=99-i,
                       RS_Before_Price_Daily='🔵', RS_Benchmark='^GSPC')
            rows.append(row)
        stocks = pd.DataFrame(rows, columns=columns)
        cache = {'version':1, 'tickers':{}}
        def good(*args, **kwargs): return {'periods':entry()['periods'], 'currency':'USD'}
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory)/'cache.json'
            fund.run_refresh(stocks, cache, cache_file, max_symbols=2, fetcher=good,
                             now=NOW, pause=lambda _:None)
            output = fund.append_financials(stocks, cache, NOW.date())
            csv_file = Path(directory)/'combined.csv'
            output.to_csv(csv_file, index=False, encoding='utf-8')
            reloaded = pd.read_csv(csv_file)
            self.assertEqual(reloaded.shape, (3,50))
            self.assertEqual(reloaded.columns[:24].tolist(), columns)
            self.assertEqual(reloaded.iloc[0,25], 1.5)  # Z: latest EPS
            self.assertEqual(reloaded.iloc[0,26], 150)  # AA: latest sales, millions
            self.assertEqual(reloaded.iloc[0,39], 50)   # AN: latest EPS YoY
            self.assertEqual(reloaded.iloc[0,40], 50)   # AO: latest sales YoY
            self.assertEqual(reloaded.iloc[2]['Financials_Refresh_Status'], 'NOT_FETCHED')
            pd.testing.assert_frame_equal(stocks, output[columns])
            export_dir = Path(directory)/'output'
            fund.write_exports(output, export_dir)
            parts = [pd.read_csv(export_dir/'rs_signals_fundamentals_1.csv')]
            parts.extend(pd.read_csv(export_dir/f'rs_signals_fundamentals_{i}.csv',
                                     header=None, names=reloaded.columns) for i in range(2,7))
            joined = pd.concat([p.loc[p['Ticker'].notna()] for p in parts], ignore_index=True)
            pd.testing.assert_frame_equal(reloaded, joined, check_dtype=False)
            self.assertTrue(all(p.columns.tolist() == reloaded.columns.tolist() for p in parts))


if __name__ == '__main__':
    unittest.main(verbosity=2)
