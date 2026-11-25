"""Sector price leadership and industry stock breadth; no fund-flow estimates.

Uses RS helpers from the installed Skyte/Fred-derived rs_ranking module in main.
Existing stock/financial CSVs are read only. No percentile mixing of ETFs/stocks.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BENCHMARK = 'SPY'
SECTOR_DOWNLOAD_YEARS = 3
SECTORS = {
    'XLC': 'Communication Services', 'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples', 'XLE': 'Energy', 'XLF': 'Financials',
    'XLV': 'Health Care', 'XLI': 'Industrials', 'XLB': 'Materials',
    'XLRE': 'Real Estate', 'XLK': 'Technology', 'XLU': 'Utilities',
}
YAHOO_TO_ETF = {
    'Communication Services': 'XLC', 'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP', 'Energy': 'XLE', 'Financial Services': 'XLF',
    'Healthcare': 'XLV', 'Industrials': 'XLI', 'Basic Materials': 'XLB',
    'Real Estate': 'XLRE', 'Technology': 'XLK', 'Utilities': 'XLU',
}
SECTOR_COLUMNS = [
    'Rank_1M', 'ETF', 'Sector', 'Data_As_Of', 'ETF_Return_1M_Pct',
    'RS_1W_Pct', 'RS_1M_Pct', 'RS_3M_Pct', 'Rank_Change_5D',
    'Daily_RS_New_High', 'Daily_RS_Before_Price',
    'Weekly_RS_New_High', 'Weekly_RS_Before_Price',
    'Historical_Daily_RS_New_High', 'Historical_Daily_RS_Before_Price',
    'Historical_Weekly_RS_New_High', 'Historical_Weekly_RS_Before_Price',
    'Daily_Bars_Since_Before_Price', 'Weekly_Bars_Since_Before_Price',
    'Last_Daily_RS_New_High_Date', 'Last_Daily_RS_Before_Price_Date',
    'Last_Weekly_RS_New_High_Date', 'Last_Weekly_RS_Before_Price_Date',
    'Daily_Status', 'Weekly_Status', 'Historical_Daily_Status',
    'Historical_Weekly_Status', 'Benchmark', 'Benchmark_As_Of', 'Generated_UTC',
]
INDUSTRY_COLUMNS = [
    'Rank', 'Industry', 'Yahoo_Sector', 'Sector_ETF_Proxy',
    'Industry_RS_Percentile', 'Member_Count', 'Avg_Member_RS_Score',
    'Median_Member_RS_Percentile', 'RS_Percentile_1M_Ago_Recalc',
    'RS_Percentile_3M_Ago_Recalc', 'RS_Percentile_6M_Ago_Recalc',
    'RS_Percentile_Change_1M_Recalc', 'Valid_Daily_Members',
    'Daily_RS_New_High_Pct', 'Daily_RS_Before_Price_Pct', 'Valid_Weekly_Members',
    'Weekly_RS_New_High_Pct', 'Weekly_RS_Before_Price_Pct', 'Stocks_As_Of',
    'Older_Or_Missing_Date_Members', 'Member_Tickers', 'Generated_UTC',
    'Classification', 'Group_Status',
    # Leader breadth: a big industry full of laggards can hide a cluster of
    # leaders behind a low average RS. Counts, not only averages.
    'Breadth_Rank', 'Leader_Count', 'RS90_Member_Count', 'Dot_20D_Member_Count',
    'Dot_20D_Pct', 'Before_Price_20D_Member_Count', 'Leader_Tickers',
]
LEADER_MIN_RS = 90      # RS rating for a "leader"
RECENT_DOT_BARS = 20    # a dot in the last 20 sessions counts as recent
SIGNAL_MAP = {
    'Daily_RS_New_High': 'RS_Blue_Dot', 'Daily_RS_Before_Price': 'RS_Before_Price',
    'Weekly_RS_New_High': 'RS_New_High_Weekly',
    'Weekly_RS_Before_Price': 'RS_Before_Price_Weekly',
    'Historical_Daily_RS_New_High': 'Historical_RS_New_High_Daily',
    'Historical_Daily_RS_Before_Price': 'Historical_RS_Before_Price_Daily',
    'Historical_Weekly_RS_New_High': 'Historical_RS_New_High_Weekly',
    'Historical_Weekly_RS_Before_Price': 'Historical_RS_Before_Price_Weekly',
    'Daily_Bars_Since_Before_Price': 'Trading_Days_Since_RS_Before_Price',
    'Weekly_Bars_Since_Before_Price': 'Weekly_Bars_Since_RS_Before_Price',
    'Last_Daily_RS_New_High_Date': 'Last_RS_New_High_Daily_Date',
    'Last_Daily_RS_Before_Price_Date': 'Last_RS_Before_Price_Daily_Date',
    'Last_Weekly_RS_New_High_Date': 'Last_RS_New_High_Weekly_Date',
    'Last_Weekly_RS_Before_Price_Date': 'Last_RS_Before_Price_Weekly_Date',
    'Daily_Status': 'RS_Signal_Status', 'Weekly_Status': 'RS_Weekly_Signal_Status',
    'Historical_Daily_Status': 'Historical_Daily_Status',
    'Historical_Weekly_Status': 'Historical_Weekly_Status',
}


def trailing_change(series, bars, offset=0):
    """End-point ratio change over exactly bars daily intervals."""
    if len(series) < bars + offset + 1:
        return None
    end = series.iloc[-1-offset]
    start = series.iloc[-1-offset-bars]
    if not (pd.notna(start) and pd.notna(end) and
            np.isfinite(start) and np.isfinite(end) and start > 0 and end > 0):
        return None
    result = (end / start - 1) * 100
    return round(float(result), 6) if np.isfinite(result) else None


def extract_candles(download, symbols):
    """Keep missing interior sessions; do not manufacture continuity."""
    output = {}
    fields = {'Close', 'High'}
    for symbol in symbols:
        if download is None or download.empty:
            output[symbol] = []
            continue
        if isinstance(download.columns, pd.MultiIndex):
            if symbol in download.columns.get_level_values(1):
                frame = download.xs(symbol, axis=1, level=1)
            elif symbol in download.columns.get_level_values(0):
                frame = download.xs(symbol, axis=1, level=0)
            else:
                output[symbol] = []
                continue
        elif len(symbols) == 1:
            frame = download
        else:
            raise ValueError('Multi-symbol download lacks ticker column levels')
        if not fields.issubset(frame.columns):
            output[symbol] = []
            continue
        if frame[['Close', 'High']].isna().all().all():
            output[symbol] = []
            continue
        rows = []
        for stamp, row in frame.iterrows():
            rows.append({
                'datetime': pd.Timestamp(stamp).strftime('%Y-%m-%d'),
                'close': None if pd.isna(row['Close']) else float(row['Close']),
                'high': None if pd.isna(row['High']) else float(row['High']),
            })
        output[symbol] = rows
    return output


def qualify_sector_signals(snapshot):
    """Require full lookbacks for ETF signals, preserving the stock IPO policy.

    The shared stock helpers allow partial initial windows and label them.
    A sector comparison must not present those as 250-day/52-week events.
    An old qualified event remains valid; blank dates never mean "never".
    """
    result = dict(snapshot)
    for timeframe, current_status, current_high, current_before in (
        ('Daily', 'RS_Signal_Status', 'RS_Blue_Dot', 'RS_Before_Price'),
        ('Weekly', 'RS_Weekly_Signal_Status', 'RS_New_High_Weekly',
         'RS_Before_Price_Weekly'),
    ):
        history_key = f'Historical_{timeframe}_Status'
        issues = [part for part in (result.get(history_key) or '').split(';')
                  if part and part != 'OK']
        short_recent = 'RECENT_WINDOW_SHORT_HISTORY' in issues
        short_current = result.get(current_status) == 'SHORT_HISTORY'
        if short_current:
            result[current_high] = None
            result[current_before] = None
        for prefix, kind, age_key in (
            ('RS', 'New_High', 'Trading_Days_Since_RS_New_High'
             if timeframe == 'Daily' else 'Weekly_Bars_Since_RS_New_High'),
            ('BEFORE_PRICE', 'Before_Price', 'Trading_Days_Since_RS_Before_Price'
             if timeframe == 'Daily' else 'Weekly_Bars_Since_RS_Before_Price'),
        ):
            date_key = f'Last_RS_{kind}_{timeframe}_Date'
            flag_key = f'Historical_RS_{kind}_{timeframe}'
            warning = prefix + '_LAST_EVENT_SHORT_LOOKBACK'
            if warning in issues:
                result[date_key] = None
                result[age_key] = None
                # The old date was the latest observed event, so no later
                # full-window event was found. Preserve unknown gap flags.
                if result.get(flag_key) is True:
                    result[flag_key] = None if short_recent else False
                issues.remove(warning)
                issues.append(prefix + '_NO_FULL_LOOKBACK_EVENT_FOUND')
            # If a recent window includes unqualified bars, only an observed
            # full-window event can establish a positive historical flag.
            if short_recent and not (
                result.get(flag_key) is True and result.get(date_key)
            ):
                result[flag_key] = None
        result[history_key] = ';'.join(issues) if issues else 'OK'
    return result


def build_sector_table(prices, frame_function, signal_function, generated):
    benchmark = frame_function(prices.get(BENCHMARK, []))
    if benchmark.empty or pd.isna(benchmark['close'].iloc[-1]):
        raise ValueError('Benchmark latest close unavailable')
    records = []
    for symbol, sector in SECTORS.items():
        candles = prices.get(symbol, [])
        snapshot = qualify_sector_signals(
            signal_function(candles, benchmark, BENCHMARK))
        row = dict.fromkeys(SECTOR_COLUMNS)
        row.update(ETF=symbol, Sector=sector, Benchmark=BENCHMARK,
                   Benchmark_As_Of=str(benchmark.index[-1].date()), Generated_UTC=generated)
        row.update({column: snapshot.get(key) for column, key in SIGNAL_MAP.items()})
        for column in list(SIGNAL_MAP)[:8]:
            row[column] = '🔵' if row[column] is True else ''
        row['_Previous_RS_1M'] = None
        try:
            frame = frame_function(candles)
        except ValueError:
            frame = None
        if frame is not None:
            row['Data_As_Of'] = str(frame.index[-1].date())
            if frame.index[-1] == benchmark.index[-1]:
                close = frame['close'].reindex(benchmark.index)
                ratio = close / benchmark['close']
                row['ETF_Return_1M_Pct'] = trailing_change(close, 21)
                for name, bars in [('RS_1W_Pct', 5), ('RS_1M_Pct', 21), ('RS_3M_Pct', 63)]:
                    row[name] = trailing_change(ratio, bars)
                row['_Previous_RS_1M'] = trailing_change(ratio, 21, offset=5)
        records.append(row)
    result = pd.DataFrame(records)
    result['Rank_1M'] = result['RS_1M_Pct'].rank(ascending=False, method='min')
    # Rank movement is not claimed when today/past coverage differs.
    if result['RS_1M_Pct'].notna().all() and result['_Previous_RS_1M'].notna().all():
        previous = result['_Previous_RS_1M'].rank(ascending=False, method='min')
        result['Rank_Change_5D'] = previous - result['Rank_1M']
    result = result.sort_values(['Rank_1M', 'ETF'], na_position='last')
    if not result['Daily_Status'].isin(['OK', 'SHORT_HISTORY']).any():
        raise ValueError('No usable sector RS snapshots; refuse to publish empty analysis')
    return result[SECTOR_COLUMNS].reset_index(drop=True)


def build_industry_table(industries, stocks, generated):
    required_industries = {'Rank', 'Industry', 'Sector', 'Relative Strength',
                          'Percentile', '1M_RS_Percentile', '3M_RS_Percentile', '6M_RS_Percentile'}
    required_stocks = {'Ticker', 'Industry', 'Sector', 'Percentile', 'Daily_Signal_Date',
                       'Weekly_Signal_Date', 'Daily_Status', 'Weekly_Status',
                       'RS_New_High_Daily', 'RS_Before_Price_Daily',
                       'RS_New_High_Weekly', 'RS_Before_Price_Weekly'}
    if industries.empty or not required_industries.issubset(industries.columns):
        raise ValueError('Industry input schema is missing or empty')
    if stocks.empty or not required_stocks.issubset(stocks.columns):
        raise ValueError('Stock input schema is missing or empty')
    if stocks['Ticker'].duplicated().any() or stocks['Ticker'].eq('').any() or stocks['Ticker'].isna().any():
        raise ValueError('Stock ticker keys must be unique and nonempty')
    if industries['Industry'].duplicated().any():
        raise ValueError('Industry names must be unique')
    stocks = stocks.copy()
    dates = pd.to_datetime(stocks['Daily_Signal_Date'], format='mixed', errors='coerce', utc=True)
    usable = stocks['Daily_Status'].isin(['OK', 'SHORT_HISTORY']) & dates.notna()
    if not usable.any():
        raise ValueError('No usable dated stock observations')
    as_of = dates[usable].max()
    stocks['_daily_date'] = dates
    stocks['_weekly_date'] = pd.to_datetime(stocks['Weekly_Signal_Date'], format='mixed', errors='coerce', utc=True)
    records = []
    for source in industries.to_dict('records'):
        members = stocks.loc[stocks['Industry'].eq(source['Industry'])].copy()
        row = dict.fromkeys(INDUSTRY_COLUMNS)
        pct = pd.to_numeric(pd.Series([source['Percentile']]), errors='coerce').iloc[0]
        previous = pd.to_numeric(pd.Series([source['1M_RS_Percentile']]), errors='coerce').iloc[0]
        sector = source['Sector']
        uniform_sector = members['Sector'].nunique() == 1 and members['Sector'].eq(sector).all()
        group_status = ('NO_CURRENT_MEMBERS' if members.empty else
                        'OK' if uniform_sector else 'SECTOR_MEMBERSHIP_MISMATCH')
        row.update(Rank=source['Rank'], Industry=source['Industry'], Yahoo_Sector=sector,
                   Sector_ETF_Proxy=YAHOO_TO_ETF.get(sector, '') if uniform_sector else '',
                   Industry_RS_Percentile=pct, Member_Count=len(members),
                   Avg_Member_RS_Score=source['Relative Strength'],
                   Median_Member_RS_Percentile=pd.to_numeric(members['Percentile'], errors='coerce').median(),
                   RS_Percentile_1M_Ago_Recalc=previous,
                   RS_Percentile_3M_Ago_Recalc=source['3M_RS_Percentile'],
                   RS_Percentile_6M_Ago_Recalc=source['6M_RS_Percentile'],
                   RS_Percentile_Change_1M_Recalc=pct-previous,
                   Stocks_As_Of=str(as_of.date()),
                   Older_Or_Missing_Date_Members=int(members['_daily_date'].ne(as_of).sum()),
                   Member_Tickers=','.join(members.sort_values('Percentile', ascending=False)['Ticker'].astype(str)),
                   Generated_UTC=generated, Classification='Yahoo Finance', Group_Status=group_status)
        for timeframe, prefix in [('Daily', 'Daily'), ('Weekly', 'Weekly')]:
            eligible = members.loc[members[f'{timeframe}_Status'].eq('OK') &
                                   members[f'_{timeframe.lower()}_date'].eq(as_of)]
            row[f'Valid_{prefix}_Members'] = len(eligible)
            for kind in ['New_High', 'Before_Price']:
                column = f'RS_{kind}_{timeframe}'
                row[f'{prefix}_RS_{kind}_Pct'] = (round(100*eligible[column].eq('🔵').mean(), 2)
                                                if len(eligible) else None)
        row.update(leader_breadth(members, as_of))
        records.append(row)
    result = pd.DataFrame(records, columns=INDUSTRY_COLUMNS)
    order = result.sort_values(['Leader_Count', 'Dot_20D_Pct', 'Rank'],
                               ascending=[False, False, True], na_position='last').index
    result.loc[order, 'Breadth_Rank'] = range(1, len(result) + 1)
    result['Breadth_Rank'] = result['Breadth_Rank'].astype('Int64')
    return result


def leader_breadth(members, as_of):
    """Leaders = RS >= LEADER_MIN_RS with a daily blue dot in the last
    RECENT_DOT_BARS sessions. Days-since values only count for members
    observed on the industry's snapshot date."""
    current = members['_daily_date'].eq(as_of) & members['Daily_Status'].eq('OK')
    pct = pd.to_numeric(members['Percentile'], errors='coerce')
    def recent(column):
        if column not in members:
            return pd.Series(False, index=members.index)
        days = pd.to_numeric(members[column], errors='coerce')
        return current & days.lt(RECENT_DOT_BARS)
    dot = recent('Days_Since_Dot')
    before = recent('Trading_Days_Since_RS_Before_Price')
    leaders = members.assign(_pct=pct).loc[pct.ge(LEADER_MIN_RS) & dot]
    leaders = leaders.sort_values('_pct', ascending=False)
    return {
        'Leader_Count': len(leaders),
        'RS90_Member_Count': int(pct.ge(LEADER_MIN_RS).sum()),
        'Dot_20D_Member_Count': int(dot.sum()),
        'Dot_20D_Pct': round(100*dot.sum()/current.sum(), 2) if current.sum() else None,
        'Before_Price_20D_Member_Count': (int(before.sum()) if
            'Trading_Days_Since_RS_Before_Price' in members else None),
        'Leader_Tickers': ','.join(leaders['Ticker'].astype(str)),
    }


def main():
    import yfinance as yf
    # Imported only after the normal stock job has created its input/cache files.
    from rs_ranking import rs_daily_frame, rs_line_signals
    generated = datetime.now(timezone.utc).isoformat(timespec='seconds')
    output = ROOT/'output'
    stocks = pd.read_csv(output/'rs_signals.csv', keep_default_na=False, na_values=[''])
    history = output/'rs_signals_history.csv'
    if history.exists():
        recent = pd.read_csv(history, keep_default_na=False, na_values=[''])
        column = 'Trading_Days_Since_RS_Before_Price'
        if {'Ticker', column}.issubset(recent.columns) and not recent['Ticker'].duplicated().any():
            stocks = stocks.merge(recent[['Ticker', column]], on='Ticker', how='left')
    industries = pd.read_csv(output/'rs_industries.csv', keep_default_na=False, na_values=[''])
    groups = build_industry_table(industries, stocks, generated)
    as_of = pd.Timestamp(groups.Stocks_As_Of.iloc[0])
    symbols = [*SECTORS, BENCHMARK]
    download = yf.download(tickers=symbols,
                           start=(as_of-pd.DateOffset(years=SECTOR_DOWNLOAD_YEARS)).date(),
                           end=(as_of+pd.Timedelta(days=1)).date(),
                           auto_adjust=True, keepna=True, progress=False,
                           threads=True, group_by='column', ignore_tz=True)
    prices = extract_candles(download, symbols)
    sectors = build_sector_table(prices, rs_daily_frame, rs_line_signals, generated)
    if sectors.Benchmark_As_Of.iloc[0] != groups.Stocks_As_Of.iloc[0]:
        raise ValueError('ETF benchmark and stock snapshots are on different dates; no rotation files published')
    output.mkdir(exist_ok=True)
    pending = []
    try:
        for name, frame in [('sector_rotation.csv', sectors), ('industry_rotation.csv', groups)]:
            path = output/name
            temporary = path.with_suffix('.csv.tmp')
            frame.to_csv(temporary, index=False, encoding='utf-8')
            pending.append((temporary, path))
        for temporary, path in pending:
            temporary.replace(path)
    finally:
        for temporary, path in pending:
            temporary.unlink(missing_ok=True)
    print(f'Sector rows: {len(sectors)}; benchmark session: {sectors.Benchmark_As_Of.iloc[0]}')
    print(f'Industry rows: {len(groups)}; stock session: {groups.Stocks_As_Of.iloc[0]}')


if __name__ == '__main__':
    main()
