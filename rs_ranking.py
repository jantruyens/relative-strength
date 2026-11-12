#!/usr/bin/env python
# rs_ranking.py — Updated version
# New columns in rs_stocks.csv:
#   Price, MarketCap, Float, ShortFloatPct, 52WkHigh, 52WkLow, PctFrom52WkHigh,
#   AvgVol10, AvgVol30, AvgVol50, RevenueGrowth
# Avg volumes calculated from candles (no extra API call).
# Market data from ticker_info.json cache (populated by rs_data.py).

import sys
import pandas as pd
import numpy as np
import json
import os
from datetime import date
from scipy.stats import linregress
import yaml
from rs_data import TD_API, cfg, read_json
from functools import reduce
import datetime

DIR = os.path.dirname(os.path.realpath(__file__))

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

try:
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
    print(exc)

PRICE_DATA = os.path.join(DIR, "data", "price_history.json")
MIN_PERCENTILE = cfg("MIN_PERCENTILE")
POS_COUNT_TARGET = cfg("POSITIONS_COUNT_TARGET")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
ALL_STOCKS = cfg("USE_ALL_LISTED_STOCKS")
TICKER_INFO_FILE = os.path.join(DIR, "data_persist", "ticker_info.json")
TICKER_INFO_DICT = read_json(TICKER_INFO_FILE)

# ── Column titles ─────────────────────────────────────────────────────────────

TITLE_RANK       = "Rank"
TITLE_TICKER     = "Ticker"
TITLE_TICKERS    = "Tickers"
TITLE_SECTOR     = "Sector"
TITLE_INDUSTRY   = "Industry"
TITLE_UNIVERSE   = "Universe" if not ALL_STOCKS else "Exchange"
TITLE_RS         = "Relative Strength"
TITLE_PERCENTILE = "Percentile"
TITLE_1M         = "1M_RS_Percentile"
TITLE_3M         = "3M_RS_Percentile"
TITLE_6M         = "6M_RS_Percentile"
TITLE_PRICE      = "Price"
TITLE_MKTCAP     = "MarketCap"
TITLE_FLOAT      = "Float"
TITLE_SHORT      = "ShortFloatPct"
TITLE_PCT_52WH   = "PctFrom52WkHigh"
TITLE_AVGVOL10   = "AvgVol10"
TITLE_AVGVOL30   = "AvgVol30"
TITLE_AVGVOL50   = "AvgVol50"
TITLE_REVGROWTH  = "RevenueGrowth"

if not os.path.exists('output'):
    os.makedirs('output')

# ── RS calculation ────────────────────────────────────────────────────────────

def relative_strength(closes: pd.Series, closes_ref: pd.Series):
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)
    rs = (1 + rs_stock) / (1 + rs_ref) * 100
    rs = int(rs * 100) / 100
    return rs

def strength(closes: pd.Series):
    """Yearly performance, most recent quarter weighted double."""
    try:
        q1 = quarters_perf(closes, 1)
        q2 = quarters_perf(closes, 2)
        q3 = quarters_perf(closes, 3)
        q4 = quarters_perf(closes, 4)
        return 0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4
    except:
        return 0

def quarters_perf(closes: pd.Series, n):
    length = min(len(closes), n * int(252 / 4))
    prices = closes.tail(length)
    pct_chg = prices.pct_change().dropna()
    perf_cum = (pct_chg + 1).cumprod() - 1
    return perf_cum.tail(1).item()

# ── Market data helpers ───────────────────────────────────────────────────────

def avg_volume(candles, days):
    """Average daily volume over the last N trading days from candle data."""
    try:
        vols = [c["volume"] for c in candles[-days:] if c.get("volume") is not None]
        return int(sum(vols) / len(vols)) if vols else None
    except Exception:
        return None

def safe_info(ticker, field):
    """Safely retrieve a field from the ticker_info cache."""
    try:
        return TICKER_INFO_DICT[ticker]["info"].get(field)
    except (KeyError, TypeError):
        return None

def pct_from_52wk_high(price, high):
    """% distance from 52-week high. Negative = below high."""
    try:
        if price and high and high > 0:
            return round((price / high - 1) * 100, 2)
        return None
    except Exception:
        return None

# ── TradingView CSV ───────────────────────────────────────────────────────────


# RS-line signals: 250 daily bars including the current session.
# Ties count, and "before price" compares intraday highs, matching the
# uploaded Fred6724 Pine Script. No cross-sectional RS rating is used here.
RS_SIGNAL_COLUMNS = [
    "RS_Blue_Dot", "RS_Before_Price", "Days_Since_Dot",
    "Dots_Last_20_Days", "RS_Signal_Date", "RS_Benchmark",
    "RS_Bars_Used", "RS_Signal_Status",
]
RS_LINE_REFERENCE_TICKER = cfg("RS_LINE_REFERENCE_TICKER") or REFERENCE_TICKER


def rs_daily_frame(candles):
    """Validate daily dates and preserve missing prices; never align by row."""
    frame = pd.DataFrame(candles)
    if frame.empty:
        raise ValueError("NO_CANDLES")
    date_key = next((key for key in ("datetime", "date", "timestamp", "time")
                     if key in frame.columns), None)
    if date_key is None:
        raise ValueError("MISSING_CANDLE_DATES")
    values = frame[date_key]
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().all():
        # Common API formats: YYYYMMDD, Unix seconds, milliseconds, microseconds.
        magnitude = numbers.abs().median()
        if 19000101 <= magnitude <= 21001231:
            dates = pd.to_datetime(values.astype(str), format="%Y%m%d",
                                   errors="coerce", utc=True)
        else:
            unit = "ns" if magnitude > 1e17 else (
                "us" if magnitude > 1e14 else ("ms" if magnitude > 1e11 else "s"))
            dates = pd.to_datetime(numbers, unit=unit, errors="coerce", utc=True)
    else:
        dates = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("INVALID_CANDLE_DATES")
    frame.index = pd.DatetimeIndex(dates).normalize()
    if frame.index.has_duplicates:
        raise ValueError("DUPLICATE_DAILY_CANDLES")
    for field in ("close", "high"):
        if field not in frame:
            frame[field] = float("nan")
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame[field] = frame[field].where(np.isfinite(frame[field]) & (frame[field] > 0))
    return frame.sort_index()


def rs_frame_signals(stock, benchmark, benchmark_name, lookback=250):
    """Return current flags plus counts within the supplied daily history.

    Short IPO histories use available bars, as ta.highest does in Pine.
    Unknown signals are blank in the CSV and explained by RS_Signal_Status.
    Days_Since_Dot and Dots_Last_20_Days refer to ANY RS-line high, not only
    the more selective before-price signal.
    """
    result = dict.fromkeys(RS_SIGNAL_COLUMNS)
    result["RS_Benchmark"] = benchmark_name
    if benchmark is None or benchmark.empty:
        result["RS_Signal_Status"] = "BENCHMARK_UNAVAILABLE"
        return result
    result["RS_Signal_Date"] = stock.index[-1].strftime("%Y-%m-%d")
    result["RS_Bars_Used"] = min(lookback, len(stock))
    if stock.index[-1] != benchmark.index[-1]:
        result["RS_Signal_Status"] = "LATEST_DATES_DIFFER"
        return result

    # Reindex leaves unmatched benchmark sessions missing instead of silently
    # comparing prices from different dates or filling missing sessions.
    ref_close = benchmark["close"].reindex(stock.index)
    if "last_session" in stock.columns:
        ref_sessions = benchmark["last_session"].reindex(stock.index)
        if stock["last_session"].iloc[-1] != ref_sessions.iloc[-1]:
            result["RS_Signal_Status"] = "LATEST_DATES_DIFFER"
            return result
        # Do not pair a Thursday stock close with a Friday benchmark close
        # merely because both belong to the same calendar week.
        ref_close = ref_close.where(stock["last_session"].eq(ref_sessions))
        result["RS_Signal_Date"] = stock["last_session"].iloc[-1].strftime("%Y-%m-%d")
    line = stock["close"] / ref_close
    window_size = pd.Series(np.minimum(np.arange(len(stock)) + 1, lookback),
                            index=stock.index)
    line_valid = line.rolling(lookback, min_periods=1).count().eq(window_size)
    highs_valid = stock["high"].rolling(lookback, min_periods=1).count().eq(window_size)
    line_max = line.rolling(lookback, min_periods=1).max()
    price_max = stock["high"].rolling(lookback, min_periods=1).max()
    dots = line.eq(line_max) & line_valid
    if not bool(line_valid.iloc[-1]):
        result["RS_Signal_Status"] = "MISSING_RS_WINDOW_DATA"
        return result

    result["RS_Blue_Dot"] = bool(dots.iloc[-1])
    if bool(highs_valid.iloc[-1]):
        result["RS_Before_Price"] = bool(
            dots.iloc[-1] and stock["high"].iloc[-1] < price_max.iloc[-1])
    found = np.flatnonzero(dots.to_numpy())
    if len(found):
        last_dot = int(found[-1])
        # A missing later observation could hide a more recent dot.
        if bool(line_valid.iloc[last_dot:].all()):
            result["Days_Since_Dot"] = len(stock) - 1 - last_dot
    if bool(line_valid.tail(20).all()):
        result["Dots_Last_20_Days"] = int(dots.tail(20).sum())
    result["RS_Signal_Status"] = (
        "MISSING_HIGH_WINDOW_DATA" if not bool(highs_valid.iloc[-1]) else
        ("SHORT_HISTORY" if len(stock) < lookback else "OK")
    )
    return result



RS_WEEKLY_COLUMNS = [
    "RS_New_High_Weekly", "RS_Before_Price_Weekly", "RS_Weekly_Signal_Date",
    "RS_Weekly_Week_Start", "RS_Weekly_Bars_Used", "RS_Weekly_Signal_Status",
]
RS_VIEW_COLUMNS = [
    "Rank", "Ticker", "Price", "Percentile",
    "RS_Before_Price_Daily", "RS_Before_Price_Weekly",
    "RS_New_High_Daily", "RS_New_High_Weekly", "Sector", "Industry",
    "Exchange", "PctFrom52WkHigh", "MarketCap", "AvgVol50", "RS_Benchmark",
    "Daily_Signal_Date", "Daily_Bars_Used", "Daily_Status",
    "Weekly_Signal_Date", "Weekly_Week_Start", "Weekly_Bars_Used",
    "Weekly_Status", "Days_Since_Dot", "Dots_Last_20_Days",
]


def rs_weekly_frame(daily):
    """Build calendar-week candles, including the current week-to-date.

    Weekly close is the LAST session's close, never a previous valid close.
    Weekly high is the maximum daily high, unknown if a daily high is missing.
    last_session makes stock/benchmark weekly closes date-comparable.
    """
    sessions = pd.Series(daily.index, index=daily.index)
    last_sessions = sessions.resample("W-FRI", closed="right", label="right").last().dropna()
    weekly = pd.DataFrame(index=last_sessions.index)
    weekly["last_session"] = last_sessions
    last_dates = pd.DatetimeIndex(last_sessions.array)
    weekly["close"] = daily["close"].reindex(last_dates).to_numpy()
    groups = daily["high"].resample("W-FRI", closed="right", label="right")
    highs = groups.max()
    all_highs_known = groups.count().eq(groups.size())
    weekly["high"] = highs.where(all_highs_known).reindex(weekly.index)
    return weekly


def rs_line_signals(candles, benchmark, benchmark_name, lookback=250, weekly_benchmark=None):
    """Daily 250-bar and weekly 52-bar signals from the same dated candles.

    Weekly flags use this week's latest daily close and week-to-date high;
    they are provisional before the weekly candle closes. The old daily
    CSV schema is preserved; the combined four-dot view has its own CSV.
    """
    try:
        stock = rs_daily_frame(candles)
    except ValueError as exc:
        result = dict.fromkeys(RS_SIGNAL_COLUMNS + RS_WEEKLY_COLUMNS)
        result["RS_Benchmark"] = benchmark_name
        result["RS_Signal_Status"] = str(exc)
        result["RS_Weekly_Signal_Status"] = str(exc)
        return result
    result = rs_frame_signals(stock, benchmark, benchmark_name, lookback)
    stock_weekly = rs_weekly_frame(stock)
    if weekly_benchmark is None and benchmark is not None and not benchmark.empty:
        weekly_benchmark = rs_weekly_frame(benchmark)
    weekly = rs_frame_signals(stock_weekly, weekly_benchmark, benchmark_name, 52)
    result.update({
        "RS_New_High_Weekly": weekly["RS_Blue_Dot"],
        "RS_Before_Price_Weekly": weekly["RS_Before_Price"],
        "RS_Weekly_Signal_Date": weekly["RS_Signal_Date"],
        "RS_Weekly_Week_Start": (stock_weekly.index[-1] - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
        "RS_Weekly_Bars_Used": weekly["RS_Bars_Used"],
        "RS_Weekly_Signal_Status": weekly["RS_Signal_Status"],
    })
    return result


def rs_signals_view(stocks, snapshots):
    """An independent export with four visible dot columns and diagnostics."""
    def dot(value):
        return "🔵" if value is not None and not pd.isna(value) and bool(value) else ""

    records = []
    for _, stock in stocks.iterrows():
        signal = snapshots[stock["Ticker"]]
        record = {column: stock.get(column) for column in RS_VIEW_COLUMNS}
        record["Exchange"] = stock.get("Exchange", stock.get("Universe"))
        record.update({
            "RS_Before_Price_Daily": dot(signal["RS_Before_Price"]),
            "RS_Before_Price_Weekly": dot(signal["RS_Before_Price_Weekly"]),
            "RS_New_High_Daily": dot(signal["RS_Blue_Dot"]),
            "RS_New_High_Weekly": dot(signal["RS_New_High_Weekly"]),
            "Daily_Signal_Date": signal["RS_Signal_Date"],
            "Daily_Bars_Used": signal["RS_Bars_Used"],
            "Daily_Status": signal["RS_Signal_Status"],
            "Weekly_Signal_Date": signal["RS_Weekly_Signal_Date"],
            "Weekly_Week_Start": signal["RS_Weekly_Week_Start"],
            "Weekly_Bars_Used": signal["RS_Weekly_Bars_Used"],
            "Weekly_Status": signal["RS_Weekly_Signal_Status"],
        })
        records.append(record)
    return pd.DataFrame(records, columns=RS_VIEW_COLUMNS)


def generate_tradingview_csv(percentile_values, first_rs_values):
    lines = []
    trading_days = 0
    yesterday = datetime.date.today() - datetime.timedelta(days=1)

    for percentile in sorted(percentile_values):
        rs_value = first_rs_values[percentile]
        for _ in range(5):
            trading_date = yesterday - datetime.timedelta(days=trading_days)
            date_str = trading_date.strftime("%Y%m%dT")
            lines.append(f"{date_str},0,1000,0,{rs_value},0\n")
            trading_days += 1

    return ''.join(reversed(lines))

# ── Rankings ──────────────────────────────────────────────────────────────────

def rankings():
    """Returns DataFrames with percentile rankings + enriched market data."""
    price_data = read_json(PRICE_DATA)
    rows = []
    ranks = []
    industries = {}
    ind_ranks = []
    stock_rs = {}
    ref = price_data[REFERENCE_TICKER]
    signal_snapshots = {}
    signal_reference = price_data.get(RS_LINE_REFERENCE_TICKER, {})
    try:
        signal_benchmark = rs_daily_frame(signal_reference.get("candles", []))
    except ValueError as exc:
        signal_benchmark = None
        print(f"RS-line benchmark {RS_LINE_REFERENCE_TICKER}: {exc}")

    signal_benchmark_weekly = (
        rs_weekly_frame(signal_benchmark) if signal_benchmark is not None else None
    )

    for ticker in price_data:
        # The separate RS-line benchmark is data for the signal, not another
        # stock in the cross-sectional percentile universe.
        if ticker == RS_LINE_REFERENCE_TICKER and ticker != REFERENCE_TICKER:
            continue
        if not cfg("SP500") and price_data[ticker].get("universe") == "S&P 500":
            continue
        if not cfg("SP400") and price_data[ticker].get("universe") == "S&P 400":
            continue
        if not cfg("SP600") and price_data[ticker].get("universe") == "S&P 600":
            continue
        if not cfg("NQ100") and price_data[ticker].get("universe") == "Nasdaq 100":
            continue
        try:
            candles    = price_data[ticker]["candles"]
            closes     = [c["close"] for c in candles]
            closes_ref = [c["close"] for c in ref["candles"]]

            industry = (
                TICKER_INFO_DICT[ticker]["info"]["industry"]
                if price_data[ticker].get("industry") == "unknown"
                else price_data[ticker].get("industry", "unknown")
            )
            sector = (
                TICKER_INFO_DICT[ticker]["info"]["sector"]
                if price_data[ticker].get("sector") == "unknown"
                else price_data[ticker].get("sector", "unknown")
            )

            # Skip tickers with no sector AND no industry — likely ETFs or
            # structured products that slipped through the NASDAQ ETF filter
            no_data = ("unknown", "n/a", "Unknown", "N/A", None, "")
            if sector in no_data and industry in no_data:
                continue

            if len(closes) >= 6 * 20:
                cs  = pd.Series(closes)
                csr = pd.Series(closes_ref)
                rs   = relative_strength(cs, csr)
                m    = 20
                rs1m = relative_strength(cs.head(-1 * m), csr.head(-1 * m))
                rs3m = relative_strength(cs.head(-3 * m), csr.head(-3 * m))
                rs6m = relative_strength(cs.head(-6 * m), csr.head(-6 * m))

                if rs < 590:
                    # Price from last candle (most reliable — from our own data)
                    price = round(closes[-1], 2) if closes[-1] else None

                    # Avg volumes from candles — no extra API call
                    av10 = avg_volume(candles, 10)
                    av30 = avg_volume(candles, 30)
                    av50 = avg_volume(candles, 50)
                    rev_growth = safe_info(ticker, "revenueGrowth")

                    # Market data from ticker_info cache
                    mktcap = safe_info(ticker, "marketCap")
                    flt    = safe_info(ticker, "floatShares")
                    short  = safe_info(ticker, "shortPercentOfFloat")
                    wk52h  = safe_info(ticker, "fiftyTwoWeekHigh")  # kept for pct calc only
                    pct52h = pct_from_52wk_high(price, wk52h)

                    signals = rs_line_signals(
                        candles, signal_benchmark, RS_LINE_REFERENCE_TICKER,
                        weekly_benchmark=signal_benchmark_weekly
                    )
                    signal_snapshots[ticker] = signals
                    ranks.append(len(ranks) + 1)
                    rows.append((
                        0,                                        # Rank placeholder
                        ticker, sector, industry,
                        price_data[ticker].get("universe", ""),
                        rs, 100,                                  # Percentile placeholder
                        rs1m, rs3m, rs6m,
                        price, mktcap, flt, short,
                        pct52h,
                        av10, av30, av50, rev_growth,
                        *(signals[column] for column in RS_SIGNAL_COLUMNS)
                    ))
                    stock_rs[ticker] = rs

                    # Industries aggregation
                    if industry not in industries:
                        industries[industry] = {
                            "info": (0, industry, sector, 0, 99, 1, 3, 6),
                            TITLE_RS: [], TITLE_1M: [], TITLE_3M: [],
                            TITLE_6M: [], TITLE_TICKERS: []
                        }
                        ind_ranks.append(len(ind_ranks) + 1)
                    industries[industry][TITLE_RS].append(rs)
                    industries[industry][TITLE_1M].append(rs1m)
                    industries[industry][TITLE_3M].append(rs3m)
                    industries[industry][TITLE_6M].append(rs6m)
                    industries[industry][TITLE_TICKERS].append(ticker)

        except KeyError:
            print(f'Ticker {ticker} has corrupted data.')

    dfs = []

    # ── Stocks DataFrame ──────────────────────────────────────────────────────
    cols = [
        TITLE_RANK, TITLE_TICKER, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_UNIVERSE,
        TITLE_RS, TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M,
        TITLE_PRICE, TITLE_MKTCAP, TITLE_FLOAT, TITLE_SHORT,
        TITLE_PCT_52WH,
        TITLE_AVGVOL10, TITLE_AVGVOL30, TITLE_AVGVOL50, TITLE_REVGROWTH,
        *RS_SIGNAL_COLUMNS
    ]
    df = pd.DataFrame(rows, columns=cols)

    df[TITLE_PERCENTILE] = pd.qcut(df[TITLE_RS],  100, labels=False, duplicates="drop")
    df[TITLE_1M]         = pd.qcut(df[TITLE_1M],  100, labels=False, duplicates="drop")
    df[TITLE_3M]         = pd.qcut(df[TITLE_3M],  100, labels=False, duplicates="drop")
    df[TITLE_6M]         = pd.qcut(df[TITLE_6M],  100, labels=False, duplicates="drop")

    df = df.sort_values([TITLE_RS], ascending=False)
    df[TITLE_RANK] = list(range(1, len(df) + 1))

    out_tickers_count = int((df[TITLE_PERCENTILE] >= MIN_PERCENTILE).sum())
    df = df.head(out_tickers_count)

    # ── TradingView RSRATING.csv ──────────────────────────────────────────────
    percentile_values = [98, 89, 69, 49, 29, 9, 1]
    first_rs_values = {}

    for percentile in percentile_values:
        matching = df[df[TITLE_PERCENTILE] == percentile]
        if matching.empty:
            available = df[TITLE_PERCENTILE].dropna().unique()
            if len(available) == 0:
                continue
            nearest = min(available, key=lambda x: abs(x - percentile))
            matching = df[df[TITLE_PERCENTILE] == nearest]
        if not matching.empty:
            first_rs_values[percentile] = matching.iloc[0][TITLE_RS]

    if len(first_rs_values) == len(percentile_values):
        tv_csv = generate_tradingview_csv(percentile_values, first_rs_values)
        with open(os.path.join(DIR, "output", "RSRATING.csv"), "w") as f:
            f.write(tv_csv)
        print("✓ RSRATING.csv generated for TradingView.")
    else:
        print("⚠ Could not generate RSRATING.csv — not enough percentile data points.")

    # ── Split into two CSVs to stay under GitHub 500KB display limit ─────────
    # rs_stocks_1.csv → Percentile 50-99 (strongest stocks, most useful)
    # rs_stocks_2.csv → Percentile 0-49  (rest)
    df_top = df[df[TITLE_PERCENTILE] >= 50].copy()
    df_bot = df[df[TITLE_PERCENTILE] <  50].copy()

    df_top.to_csv(os.path.join(DIR, "output", "rs_stocks_1.csv"), index=False)
    df_bot.to_csv(os.path.join(DIR, "output", "rs_stocks_2.csv"), index=False)

    # Full file kept for programmatic use (not rendered by GitHub but downloadable)
    df.to_csv(os.path.join(DIR, "output", "rs_stocks.csv"), index=False)
    # Separate export: do not expand the existing 27-column import in Sheets.
    rs_signals_view(df, signal_snapshots).to_csv(
        os.path.join(DIR, "output", "rs_signals.csv"), index=False, encoding="utf-8"
    )

    print(f"✓ rs_stocks_1.csv: {len(df_top)} tickers (Percentile 50-99)")
    print(f"✓ rs_stocks_2.csv: {len(df_bot)} tickers (Percentile 0-49)")
    dfs.append(df)

    # ── Industries DataFrame ──────────────────────────────────────────────────
    def getDfView(entry):
        return entry["info"]

    def rs_sum(a, b):
        return a + b

    def getRsAverage(ind_dict, industry, column):
        vals = ind_dict[industry][column]
        return int((reduce(rs_sum, vals) / len(vals)) * 100) / 100

    def rs_for_stock(t):
        return stock_rs.get(t, 0)

    def getTickers(ind_dict, industry):
        return ",".join(sorted(ind_dict[industry][TITLE_TICKERS], key=rs_for_stock, reverse=True))

    filtered = list(filter(lambda i: len(i[TITLE_TICKERS]) > 1, list(industries.values())))
    df_ind = pd.DataFrame(
        map(getDfView, filtered),
        columns=[TITLE_RANK, TITLE_INDUSTRY, TITLE_SECTOR, TITLE_RS,
                 TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M]
    )

    df_ind[TITLE_RS]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_RS),  axis=1)
    df_ind[TITLE_1M]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_1M),  axis=1)
    df_ind[TITLE_3M]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_3M),  axis=1)
    df_ind[TITLE_6M]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_6M),  axis=1)

    df_ind[TITLE_PERCENTILE] = pd.qcut(df_ind[TITLE_RS], 100, labels=False, duplicates="drop")
    df_ind[TITLE_1M]         = pd.qcut(df_ind[TITLE_1M], 100, labels=False, duplicates="drop")
    df_ind[TITLE_3M]         = pd.qcut(df_ind[TITLE_3M], 100, labels=False, duplicates="drop")
    df_ind[TITLE_6M]         = pd.qcut(df_ind[TITLE_6M], 100, labels=False, duplicates="drop")

    df_ind[TITLE_TICKERS] = df_ind.apply(lambda r: getTickers(industries, r[TITLE_INDUSTRY]), axis=1)
    df_ind = df_ind.sort_values([TITLE_RS], ascending=False)
    df_ind[TITLE_RANK] = list(range(1, len(df_ind) + 1))

    df_ind.to_csv(os.path.join(DIR, "output", 'rs_industries.csv'), index=False)
    dfs.append(df_ind)

    return dfs


def main(skipEnter=False):
    ranks = rankings()
    print(ranks[0])
    print("***\nYour 'rs_stocks.csv' is in the output folder.\n***")
    if not skipEnter and cfg("EXIT_WAIT_FOR_ENTER"):
        input("Press Enter key to exit...")

if __name__ == "__main__":
    main()