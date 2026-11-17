import ast
import importlib.util
import contextlib
import datetime as dt
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rotation_under_test',ROOT/'market_rotation.py')
rotation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotation)
tree=ast.parse((ROOT/'rs_ranking.py').read_text(encoding='utf-8'))
functions={'rs_daily_frame','rs_weekly_frame','rs_frame_signals','rs_line_signals','rs_history_summary'}
constants={'RS_SIGNAL_COLUMNS','RS_WEEKLY_COLUMNS','RS_HISTORY_COLUMNS','RS_HISTORY_CORE_COLUMNS'}
parts=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions or
       isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
ns={'pd':pd,'np':np}
exec(compile(ast.Module(body=parts,type_ignores=[]),'<rs helpers>','exec'),ns)


def candles(n=400, slope=1, base=100):
    return [{'datetime':str(d.date()),'close':base+i*slope,'high':base+i*slope+2}
            for i,d in enumerate(pd.bdate_range('2025-01-01',periods=n))]


def sector_prices():
    return {'SPY':candles(slope=0),**{symbol:candles(slope=.1*(i+1))
            for i,symbol in enumerate(rotation.SECTORS)}}


def sectors(prices):
    return rotation.build_sector_table(prices,ns['rs_daily_frame'],ns['rs_line_signals'],'TEST_UTC')


def industry_inputs():
    industries=pd.DataFrame([{'Rank':1,'Industry':'Software','Sector':'Technology',
        'Relative Strength':110,'Percentile':70,'1M_RS_Percentile':30,
        '3M_RS_Percentile':20,'6M_RS_Percentile':10}])
    stocks=pd.DataFrame([{'Ticker':t,'Industry':'Software','Sector':'Technology',
        'Percentile':p,'Daily_Signal_Date':'2026-11-13','Weekly_Signal_Date':'2026-11-13',
        'Daily_Status':'OK','Weekly_Status':'OK','RS_New_High_Daily':dot,
        'RS_Before_Price_Daily':dot,'RS_New_High_Weekly':dot,'RS_Before_Price_Weekly':dot}
        for t,p,dot in [('NA',80,'🔵'),('B',60,''),('C',40,''),('D',20,'')]])
    return industries,stocks


class SectorRotationTests(unittest.TestCase):
    def test_full_table_and_original_blue_dot_rules(self):
        prices=sector_prices()
        for symbol in rotation.SECTORS:prices[symbol][-10]['high']=10000
        out=sectors(prices)
        self.assertEqual(out.shape,(11,30))
        self.assertEqual(set(out['ETF']),set(rotation.SECTORS))
        self.assertTrue(out['Daily_RS_New_High'].eq('🔵').all())
        self.assertTrue(out['Daily_RS_Before_Price'].eq('🔵').all())
        self.assertTrue(out['Weekly_RS_New_High'].eq('🔵').all())
        self.assertTrue(out['Weekly_RS_Before_Price'].eq('🔵').all())
        self.assertTrue(out['Benchmark'].eq('SPY').all())

    def test_relative_return_uses_ratio_not_percentage_point_subtraction(self):
        prices=sector_prices()
        for symbol in rotation.SECTORS:
            prices[symbol][-22]['close']=100
            prices[symbol][-1]['close']=120
        prices['SPY'][-22]['close']=100
        prices['SPY'][-1]['close']=110
        out=sectors(prices)
        self.assertAlmostEqual(out.iloc[0]['RS_1M_Pct'],(1.2/1.1-1)*100,places=5)
        self.assertAlmostEqual(out.iloc[0]['ETF_Return_1M_Pct'],20)

    def test_relative_strength_can_rise_while_sector_price_falls(self):
        prices={'SPY':candles(base=1000,slope=-1.4),
                **{s:candles(base=500,slope=-.3) for s in rotation.SECTORS}}
        out=sectors(prices)
        self.assertTrue(out['ETF_Return_1M_Pct'].lt(0).all())
        self.assertTrue(out['RS_1M_Pct'].gt(0).all())

    def test_stale_etf_does_not_get_current_metrics(self):
        prices=sector_prices()
        prices['XLK']=prices['XLK'][:-1]
        row=sectors(prices).set_index('ETF').loc['XLK']
        self.assertEqual(row['Daily_Status'],'LATEST_DATES_DIFFER')
        self.assertTrue(pd.isna(row['RS_1M_Pct']))
        self.assertEqual(row['Daily_RS_New_High'],'')

    def test_missing_etf_is_visible_and_rank_change_is_withheld(self):
        prices=sector_prices()
        del prices['XLE']
        out=sectors(prices)
        row=out.set_index('ETF').loc['XLE']
        self.assertEqual(len(out),11)
        self.assertEqual(row['Daily_Status'],'NO_CANDLES')
        self.assertTrue(out['Rank_Change_5D'].isna().all())

    def test_rank_change_against_independent_past_endpoints(self):
        prices=sector_prices()
        prices['XLC'][-1]['close']*=2
        out=sectors(prices).set_index('ETF')
        current={s:prices[s][-1]['close']/prices[s][-22]['close'] for s in rotation.SECTORS}
        earlier={s:prices[s][-6]['close']/prices[s][-27]['close'] for s in rotation.SECTORS}
        for s in rotation.SECTORS:
            now=1+sum(v>current[s] for v in current.values())
            old=1+sum(v>earlier[s] for v in earlier.values())
            self.assertEqual(out.loc[s,'Rank_1M'],now)
            self.assertEqual(out.loc[s,'Rank_Change_5D'],old-now)

    def test_interior_gap_is_not_filled_into_a_blue_dot(self):
        prices=sector_prices()
        prices['XLK'][-10]['close']=None
        row=sectors(prices).set_index('ETF').loc['XLK']
        self.assertEqual(row['Daily_Status'],'MISSING_RS_WINDOW_DATA')
        self.assertEqual(row['Daily_RS_New_High'],'')

    def test_bad_latest_benchmark_is_fatal(self):
        prices=sector_prices()
        prices['SPY'][-1]['close']=None
        with self.assertRaisesRegex(ValueError,'Benchmark'):sectors(prices)

    def test_exact_trading_interval_and_missing_endpoints(self):
        self.assertIsNone(rotation.trailing_change(pd.Series([1,2]),2))
        self.assertEqual(rotation.trailing_change(pd.Series([1,2,3]),2),200)
        self.assertIsNone(rotation.trailing_change(pd.Series([1,2,np.nan]),2))
        self.assertIsNone(rotation.trailing_change(pd.Series([0,2,3]),2))

    def test_rank_ties_are_not_arbitrarily_broken(self):
        prices=sector_prices()
        for s in rotation.SECTORS:prices[s]=candles(slope=1)
        out=sectors(prices)
        self.assertTrue(out['Rank_1M'].eq(1).all())
        self.assertTrue(out['Rank_Change_5D'].eq(0).all())


class DownloadParsingTests(unittest.TestCase):
    def test_both_multiindex_column_orders(self):
        index=pd.bdate_range('2026-01-01',periods=3)
        frame=pd.DataFrame([[1,2,3,4]]*3,index=index,
            columns=pd.MultiIndex.from_product([['Close','High'],['SPY','XLK']]))
        normal=rotation.extract_candles(frame,['SPY','XLK'])
        swapped=rotation.extract_candles(frame.swaplevel(axis=1),['SPY','XLK'])
        self.assertEqual(normal,swapped)
        self.assertEqual(normal['SPY'][0]['close'],1)

    def test_missing_session_is_preserved(self):
        frame=pd.DataFrame({'Close':[1,np.nan,3],'High':[2,np.nan,4]},index=pd.bdate_range('2026-01-01',periods=3))
        out=rotation.extract_candles(frame,['SPY'])['SPY']
        self.assertEqual(len(out),3)
        self.assertIsNone(out[1]['close'])

    def test_empty_download_produces_explicit_missing_symbols(self):
        self.assertEqual(rotation.extract_candles(pd.DataFrame(),['SPY','XLK']),{'SPY':[],'XLK':[]})


class FullSectorLookbackTests(unittest.TestCase):
    def test_initial_partial_daily_and_weekly_events_are_not_published(self):
        prices=sector_prices()
        rows=prices['XLY']
        for i,row in enumerate(rows):
            row['close']=100+5*i if i<=20 else 200-.1*(i-20)
            row['high']=1000 if i==0 else row['close']+2
        benchmark=ns['rs_daily_frame'](prices['SPY'])
        original=ns['rs_line_signals'](rows,benchmark,'SPY')
        self.assertIn('RS_LAST_EVENT_SHORT_LOOKBACK',original['Historical_Daily_Status'])
        self.assertIn('BEFORE_PRICE_LAST_EVENT_SHORT_LOOKBACK',original['Historical_Daily_Status'])
        self.assertIn('RS_LAST_EVENT_SHORT_LOOKBACK',original['Historical_Weekly_Status'])
        out=sectors(prices).set_index('ETF').loc['XLY']
        for column in ('Last_Daily_RS_New_High_Date','Last_Daily_RS_Before_Price_Date',
                       'Last_Weekly_RS_New_High_Date','Last_Weekly_RS_Before_Price_Date',
                       'Daily_Bars_Since_Before_Price','Weekly_Bars_Since_Before_Price'):
            self.assertTrue(pd.isna(out[column]),column)
        self.assertIn('RS_NO_FULL_LOOKBACK_EVENT_FOUND',out.Historical_Daily_Status)
        self.assertIn('BEFORE_PRICE_NO_FULL_LOOKBACK_EVENT_FOUND',out.Historical_Weekly_Status)
        self.assertEqual(out.Historical_Daily_RS_New_High,'')
        self.assertEqual(out.Data_As_Of,rows[-1]['datetime'])
        self.assertTrue(pd.notna(out.RS_1M_Pct))
        # The original snapshot and shared stock helper retain their IPO policy.
        self.assertIsNotNone(original['Last_RS_New_High_Daily_Date'])

    def test_old_fully_qualified_event_is_retained(self):
        prices=sector_prices()
        for i,row in enumerate(prices['XLY']):
            close=(200-.2*i if i<250 else
                   151+4*(i-250) if i<=270 else 231-.2*(i-270))
            row.update(close=close,high=1000 if i==260 else close+2)
        row=sectors(prices).set_index('ETF').loc['XLY']
        self.assertEqual(row.Last_Daily_RS_New_High_Date,prices['XLY'][270]['datetime'])
        self.assertEqual(row.Last_Daily_RS_Before_Price_Date,prices['XLY'][270]['datetime'])
        self.assertEqual(row.Daily_Bars_Since_Before_Price,129)
        self.assertEqual(row.Historical_Daily_RS_New_High,'')
        self.assertEqual(row.Daily_Status,'OK')

    def test_current_short_daily_and_weekly_signals_are_unavailable(self):
        prices={'SPY':candles(n=20,slope=0),
                **{s:candles(n=20,slope=1) for s in rotation.SECTORS}}
        out=sectors(prices)
        self.assertTrue(out.Daily_Status.eq('SHORT_HISTORY').all())
        self.assertTrue(out.Weekly_Status.eq('SHORT_HISTORY').all())
        for column in rotation.SECTOR_COLUMNS[9:17]:
            self.assertTrue(out[column].eq('').all(),column)

    def test_a_full_daily_event_inside_a_short_recent_window_is_kept(self):
        prices={'SPY':candles(n=255,slope=0),
                **{s:candles(n=255,slope=1) for s in rotation.SECTORS}}
        out=sectors(prices)
        self.assertTrue(out.Historical_Daily_RS_New_High.eq('\U0001f535').all())
        self.assertTrue(out.Last_Daily_RS_New_High_Date.eq(prices['SPY'][-1]['datetime']).all())
        # This calendar starts midweek: 255 sessions span 52 weekly bars.
        self.assertTrue(out.Weekly_Status.eq('OK').all())
        self.assertTrue(out.Weekly_RS_New_High.eq('\U0001f535').all())

    def test_no_partial_event_is_misreported_as_a_known_negative_recent_window(self):
        prices={'SPY':candles(n=255,slope=0),
                **{s:candles(n=255,slope=-.1) for s in rotation.SECTORS}}
        snapshot=ns['rs_line_signals'](prices['XLY'],ns['rs_daily_frame'](prices['SPY']),'SPY')
        fixed=rotation.qualify_sector_signals(snapshot)
        self.assertIsNone(fixed['Historical_RS_New_High_Daily'])
        self.assertIsNone(fixed['Last_RS_New_High_Daily_Date'])

    def test_missing_later_observations_keep_the_uncertain_event_warning(self):
        prices=sector_prices()
        prices['XLY'][-10]['close']=None
        row=sectors(prices).set_index('ETF').loc['XLY']
        self.assertIn('RS_LAST_EVENT_UNCERTAIN',row.Historical_Daily_Status)
        self.assertTrue(pd.isna(row.Last_Daily_RS_New_High_Date))
        self.assertEqual(row.Daily_Status,'MISSING_RS_WINDOW_DATA')


class IndustryRotationTests(unittest.TestCase):
    def test_member_breadth_not_industry_index_blue_dot(self):
        ind,stocks=industry_inputs()
        row=rotation.build_industry_table(ind,stocks,'TEST').iloc[0]
        self.assertEqual(row['Sector_ETF_Proxy'],'XLK')
        self.assertEqual(row['Member_Count'],4)
        self.assertEqual(row['Valid_Daily_Members'],4)
        self.assertEqual(row['Daily_RS_New_High_Pct'],25)
        self.assertEqual(row['Weekly_RS_Before_Price_Pct'],25)
        self.assertEqual(row['Median_Member_RS_Percentile'],50)
        self.assertEqual(row['Member_Tickers'].split(',')[0],'NA')

    def test_stale_and_short_history_members_are_excluded_from_breadth_denominator(self):
        ind,stocks=industry_inputs()
        stocks.loc[2,'Daily_Signal_Date']='2026-11-12'
        stocks.loc[2,'Weekly_Signal_Date']='2026-11-12'
        stocks.loc[3,'Daily_Status']='SHORT_HISTORY'
        stocks.loc[3,'Weekly_Status']='SHORT_HISTORY'
        row=rotation.build_industry_table(ind,stocks,'TEST').iloc[0]
        self.assertEqual(row['Member_Count'],4)
        self.assertEqual(row['Valid_Daily_Members'],2)
        self.assertEqual(row['Valid_Weekly_Members'],2)
        self.assertEqual(row['Daily_RS_New_High_Pct'],50)
        self.assertEqual(row['Older_Or_Missing_Date_Members'],1)

    def test_no_eligible_members_means_unavailable_not_zero(self):
        ind,stocks=industry_inputs()
        stocks['Daily_Status']='SHORT_HISTORY'
        stocks['Weekly_Status']='SHORT_HISTORY'
        row=rotation.build_industry_table(ind,stocks,'TEST').iloc[0]
        self.assertEqual(row['Valid_Daily_Members'],0)
        self.assertIsNone(row['Daily_RS_New_High_Pct'])

    def test_known_no_dots_is_real_zero(self):
        ind,stocks=industry_inputs()
        for c in ['RS_New_High_Daily','RS_Before_Price_Daily','RS_New_High_Weekly','RS_Before_Price_Weekly']:stocks[c]=''
        row=rotation.build_industry_table(ind,stocks,'TEST').iloc[0]
        self.assertEqual(row['Daily_RS_New_High_Pct'],0)

    def test_older_columns_are_lagged_recalculations(self):
        ind,stocks=industry_inputs()
        row=rotation.build_industry_table(ind,stocks,'TEST').iloc[0]
        self.assertEqual(row['RS_Percentile_1M_Ago_Recalc'],30)
        self.assertEqual(row['RS_Percentile_Change_1M_Recalc'],40)
        self.assertEqual(row['Classification'],'Yahoo Finance')

    def test_classification_mismatch_does_not_guess_sector_proxy(self):
        ind,stocks=industry_inputs()
        stocks.loc[1,'Sector']='Energy'
        row=rotation.build_industry_table(ind,stocks,'TEST').iloc[0]
        self.assertEqual(row['Sector_ETF_Proxy'],'')
        self.assertEqual(row['Group_Status'],'SECTOR_MEMBERSHIP_MISMATCH')

    def test_duplicate_tickers_and_industries_are_rejected(self):
        ind,stocks=industry_inputs()
        with self.assertRaisesRegex(ValueError,'ticker'):
            rotation.build_industry_table(ind,pd.concat([stocks,stocks.iloc[:1]]),'TEST')
        with self.assertRaisesRegex(ValueError,'Industry names'):
            rotation.build_industry_table(pd.concat([ind,ind]),stocks,'TEST')

    def test_missing_schema_and_unknown_stock_dates_are_rejected(self):
        ind,stocks=industry_inputs()
        with self.assertRaises(ValueError):rotation.build_industry_table(ind.drop(columns='Sector'),stocks,'TEST')
        stocks['Daily_Signal_Date']='invalid'
        with self.assertRaisesRegex(ValueError,'dated'):rotation.build_industry_table(ind,stocks,'TEST')


class PublishingIntegrationTests(unittest.TestCase):
    def exercise(self, mismatch=False):
        prices=sector_prices()
        latest=prices['SPY'][-1]['datetime']
        ind,stocks=industry_inputs()
        stocks['Daily_Signal_Date']=latest
        stocks['Weekly_Signal_Date']=latest
        dates=pd.to_datetime([r['datetime'] for r in prices['SPY']])
        downloaded=pd.DataFrame(index=dates)
        for ticker,rows in prices.items():
            for field,key in [('Close','close'),('High','high')]:
                downloaded[(field,ticker)]=[r[key] for r in rows]
        downloaded.columns=pd.MultiIndex.from_tuples(downloaded.columns)
        if mismatch:downloaded=downloaded.iloc[:-1]
        calls=[]
        def fake_download(**kwargs):calls.append(kwargs);return downloaded
        fake_rs=types.SimpleNamespace(rs_daily_frame=ns['rs_daily_frame'],rs_line_signals=ns['rs_line_signals'])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'output').mkdir()
            ind.to_csv(root/'output/rs_industries.csv',index=False)
            stocks.to_csv(root/'output/rs_signals.csv',index=False)
            before=(root/'output/rs_signals.csv').read_bytes()
            with patch.object(rotation,'ROOT',root),patch.dict(sys.modules,{
                'yfinance':types.SimpleNamespace(download=fake_download),'rs_ranking':fake_rs}),contextlib.redirect_stdout(io.StringIO()):
                if mismatch:
                    with self.assertRaisesRegex(ValueError,'different dates'):rotation.main()
                else:rotation.main()
            if mismatch:
                self.assertFalse((root/'output/sector_rotation.csv').exists())
                self.assertFalse((root/'output/industry_rotation.csv').exists())
            else:
                sectors_out=pd.read_csv(root/'output/sector_rotation.csv',keep_default_na=False)
                industry_out=pd.read_csv(root/'output/industry_rotation.csv',keep_default_na=False)
                self.assertEqual(sectors_out.shape,(11,30))
                self.assertEqual(industry_out.shape,(1,24))
                self.assertEqual(industry_out.loc[0,'Member_Tickers'].split(',')[0],'NA')
                self.assertEqual(sectors_out.Benchmark_As_Of.iloc[0],latest)
                self.assertEqual(calls[0]['end'],(pd.Timestamp(latest)+pd.Timedelta(days=1)).date())
                self.assertEqual(calls[0]['start'],(pd.Timestamp(latest)-pd.DateOffset(years=3)).date())
            self.assertEqual((root/'output/rs_signals.csv').read_bytes(),before)

    def test_real_cli_writes_both_tables_without_changing_stock_inputs(self):self.exercise()

    def test_mismatched_sector_and_stock_snapshots_do_not_publish(self):self.exercise(True)

    def test_stock_downloader_includes_run_date_with_exclusive_end_api(self):
        source=ast.parse((ROOT/'rs_data.py').read_text(encoding='utf-8-sig'))
        selected=[node for node in source.body if isinstance(node,ast.FunctionDef) and
                  node.name in ('load_prices_from_yahoo','parse_batch_download','escape_ticker','candles_from_df_row')]
        for run_date in [dt.date(2026,11,13),dt.date(2026,11,16),dt.date(2026,12,31)]:
            with self.subTest(run_date=run_date):
                class FixedDate(dt.date):
                    @classmethod
                    def today(cls):return run_date
                index=pd.bdate_range(end=run_date,periods=3)
                frame=pd.DataFrame({'Open':1.,'High':2.,'Low':1.,'Close':1.,'Volume':1000.},index=index)
                def exclusive_download(**kwargs):return frame.loc[frame.index<pd.Timestamp(kwargs['end'])]
                environment={'pd':pd,'date':FixedDate,'dt':dt,'np':np,'REFERENCE_TICKER':'SPY',
                    'BATCH_SIZE':100,'time':types.SimpleNamespace(time=lambda:1.),
                    'yf':types.SimpleNamespace(download=exclusive_download),
                    'TICKER_INFO_DICT':{'SPY':{'info':{'industry':'Index','sector':'Index'}}},
                    'needs_refresh':lambda *args:False,'UNKNOWN':'Unknown',
                    'enrich_ticker_data':lambda out,sec:out.update(universe='NYSE'),
                    'write_price_history_file':lambda *args:None,'write_ticker_info_file':lambda *args:None}
                exec(compile(ast.Module(body=selected,type_ignores=[]),'<downloader>','exec'),environment)
                with contextlib.redirect_stdout(io.StringIO()):
                    out=environment['load_prices_from_yahoo']([{'ticker':'SPY'}])
                last=pd.Timestamp(out['SPY']['candles'][-1]['datetime'],unit='s').date()
                self.assertEqual(last,run_date)


if __name__=='__main__':unittest.main()
