import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT/'rs_ranking.py').read_text(encoding='utf-8'))
nodes = [n for n in tree.body if (
    isinstance(n, ast.FunctionDef) and n.name in
    ('rs_daily_frame','dollar_volume_summary','rs_liquidity_view')) or (
    isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and
    t.id=='LIQUIDITY_COLUMNS' for t in n.targets))]
ns = {'pd':pd, 'np':np}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<liquidity>','exec'),ns)
calculate = ns['dollar_volume_summary']


def candles(n=60):
    return [{'datetime':str(d.date()),'close':10+i,'high':12+i,'volume':100+i}
            for i,d in enumerate(pd.bdate_range('2026-01-01',periods=n))]


class DollarVolumeTests(unittest.TestCase):
    def test_average_pairs_not_latest_price_times_average_volume(self):
        rows=candles()
        out=calculate(rows)
        expected=round(sum(c['close']*c['volume'] for c in rows[-50:])/50,2)
        self.assertEqual(out['AvgDollarVolume50'],expected)
        self.assertNotEqual(out['AvgDollarVolume50'],rows[-1]['close']*np.mean([c['volume'] for c in rows[-50:]]))
        self.assertEqual(out['DollarVolume50_Status'],'OK')
        self.assertEqual(out['DollarVolume50_Bars_Used'],50)

    def test_trailing_window_ignores_older_invalid_values(self):
        rows=candles(51)
        rows[0]['volume']=None
        self.assertEqual(calculate(rows)['DollarVolume50_Status'],'OK')
        rows[1]['volume']=None
        self.assertIsNone(calculate(rows)['AvgDollarVolume50'])

    def test_short_window_is_not_relabelled_as_50_days(self):
        out=calculate(candles(49))
        self.assertIsNone(out['AvgDollarVolume50'])
        self.assertEqual(out['DollarVolume50_Bars_Used'],49)
        self.assertEqual(out['DollarVolume50_Status'],'SHORT_HISTORY')

    def test_zero_volume_is_real_observation(self):
        rows=candles()
        for row in rows: row['volume']=0
        out=calculate(rows)
        self.assertEqual(out['AvgDollarVolume50'],0)
        self.assertEqual(out['DollarVolume50_Bars_Used'],50)
        self.assertEqual(out['DollarVolume50_Status'],'OK')

    def test_invalid_inputs_are_not_silently_skipped(self):
        for field,values in [('volume',[None,-1,float('inf'),float('nan'),True,'bad']),
                             ('close',[None,0,-1,float('inf'),True,'bad'])]:
            for value in values:
                with self.subTest(field=field,value=value):
                    rows=candles()
                    rows[-3][field]=value
                    out=calculate(rows)
                    self.assertIsNone(out['AvgDollarVolume50'])
                    self.assertEqual(out['DollarVolume50_Bars_Used'],49)
                    self.assertIn('MISSING_OR_INVALID',out['DollarVolume50_Status'])

    def test_all_missing_volume_is_unavailable_not_zero(self):
        rows=candles()
        for row in rows: del row['volume']
        out=calculate(rows)
        self.assertEqual(out['DollarVolume50_Bars_Used'],0)
        self.assertIsNone(out['AvgDollarVolume50'])

    def test_date_sorting_preserves_price_volume_pairing(self):
        rows=candles()
        self.assertEqual(calculate(rows),calculate(list(reversed(rows))))

    def test_duplicate_and_invalid_dates_have_explicit_status(self):
        rows=candles()
        rows[-1]['datetime']=rows[-2]['datetime']
        out=calculate(rows)
        self.assertEqual(out['DollarVolume50_Status'],'DUPLICATE_DAILY_CANDLES')
        self.assertIsNone(out['Liquidity_As_Of'])
        rows[-1]['datetime']='invalid'
        self.assertEqual(calculate(rows)['DollarVolume50_Status'],'INVALID_CANDLE_DATES')

    def test_unix_seconds_dates_match_normal_dates(self):
        rows=candles()
        numeric=[dict(c,datetime=int(pd.Timestamp(c['datetime'],tz='UTC').timestamp())) for c in rows]
        self.assertEqual(calculate(rows),calculate(numeric))

    def test_empty_history_has_no_manufactured_date_or_average(self):
        out=calculate([])
        self.assertEqual(out['DollarVolume50_Status'],'NO_CANDLES')
        self.assertIsNone(out['AvgDollarVolume50'])
        self.assertIsNone(out['Liquidity_As_Of'])

    def test_pair_product_overflow_is_rejected(self):
        rows=candles()
        rows[-1].update(close=1e250,volume=1e250)
        with np.errstate(over='ignore'):
            out=calculate(rows)
        self.assertIsNone(out['AvgDollarVolume50'])
        self.assertEqual(out['DollarVolume50_Status'],'NONFINITE_DOLLAR_VOLUME')

    def test_snapshot_keys_order_and_na_ticker_are_preserved(self):
        rows=candles()
        stocks=pd.DataFrame({'Ticker':['NA','AAPL','MISSING']})
        out=ns['rs_liquidity_view'](stocks,{'AAPL':{'candles':rows},'NA':{'candles':rows}})
        self.assertEqual(out['Ticker'].tolist(),['NA','AAPL','MISSING'])
        self.assertEqual(out.shape,(3,5))
        self.assertEqual(out.loc[0,'Liquidity_As_Of'],rows[-1]['datetime'])
        self.assertEqual(out.loc[2,'DollarVolume50_Status'],'NO_CANDLES')
        self.assertTrue(pd.isna(out.loc[2,'AvgDollarVolume50']))


if __name__=='__main__':
    unittest.main()
