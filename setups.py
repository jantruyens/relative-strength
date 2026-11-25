#!/usr/bin/env python
"""Pullback scan: "weakness after strength" in RS leaders.

Runs after relative-strength.py, reusing data/price_history.json (no extra
downloads). Writes output/setups.csv with one row per stock at or above
SETUP_MIN_RS, flagged when it shows the pattern:

  Strength   at least SETUP_MIN_DOTS daily RS-line new highs (blue dots) in the
             last SETUP_DOT_WINDOW sessions
  Weakness   the latest dot is SETUP_MIN_DAYS_SINCE_DOT to
             SETUP_MAX_DAYS_SINCE_DOT sessions old (pullback without new dots)
  Support    close within SETUP_NEAR_PCT % of the daily 21 EMA or the
             8-week EMA (weekly closes)
  Trend      8-week EMA rising and close above the 50-day SMA

A blue dot uses the same definition as rs_ranking.rs_frame_signals: the RS
line (close / S&P 500 close) equals its highest value of the last 250 bars.
tests/test_setups.py checks that both give the same dots.

The 8-week EMA uses weekly closes, with the current week's latest close as a
provisional weekly close (as on a live TradingView weekly chart).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

DIR = os.path.dirname(os.path.realpath(__file__))
PRICE_DATA = os.path.join(DIR, "data", "price_history.json")
RS_STOCKS = os.path.join(DIR, "output", "rs_signals.csv")
OUTPUT = os.path.join(DIR, "output", "setups.csv")

LOOKBACK = 250  # RS-line new-high lookback, same as the blue dot
DEFAULTS = {
    "SETUP_MIN_RS": 80,              # only scan stocks with RS rating >= this
    "SETUP_DOT_WINDOW": 40,          # sessions counted for "strength"
    "SETUP_MIN_DOTS": 3,             # dots needed in that window
    "SETUP_MIN_DAYS_SINCE_DOT": 5,   # pullback has lasted at least this long
    "SETUP_MAX_DAYS_SINCE_DOT": 20,  # ...but not longer than this
    "SETUP_NEAR_PCT": 3.0,           # max distance to 21 EMA / 8-week EMA, %
}

SETUP_COLUMNS = [
    "Ticker", "Setup", "Near_Support", "RS_Rating", "Price",
    "Dots_In_Window", "Before_Price_Dots_In_Window", "Days_Since_Dot",
    "Weekly_RS_Before_Price", "Pct_From_EMA21", "Pct_From_EMA8W",
    "EMA8W_Rising", "Above_SMA50", "Pct_From_52W_High", "Inside_Day",
    "Vol_5D_vs_50D", "HV1_Date", "HV1_Sessions_Ago", "HV1_High",
    "Pct_From_HV1_High", "EMA21", "EMA8W", "SMA50", "AvgDollarVol50_M",
    "Sector", "Industry", "MarketCap", "As_Of", "Status",
]


def daily_frame(candles):
    """Candles -> date-indexed OHLCV frame, oldest first.

    Like rs_daily_frame, rows are kept when a price is missing (NaN) so the
    250-bar windows line up with the blue-dot calculation.
    """
    frame = pd.DataFrame(candles)
    if frame.empty or "datetime" not in frame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    stamps = frame["datetime"]
    numbers = pd.to_numeric(stamps, errors="coerce")
    if numbers.notna().all():
        magnitude = numbers.abs().median()
        if 19000101 <= magnitude <= 21001231:
            dates = pd.to_datetime(stamps.astype(str), format="%Y%m%d", errors="coerce")
        else:
            unit = "ns" if magnitude > 1e17 else (
                "us" if magnitude > 1e14 else ("ms" if magnitude > 1e11 else "s"))
            dates = pd.to_datetime(numbers, unit=unit, errors="coerce", utc=True)
    else:
        dates = pd.to_datetime(stamps, format="mixed", errors="coerce", utc=True)
    dates = pd.DatetimeIndex(dates)
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    frame.index = dates.normalize()
    frame = frame[frame.index.notna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].where(frame[column] > 0)
    return frame[["open", "high", "low", "close", "volume"]]


def blue_dots(stock, benchmark, lookback=LOOKBACK):
    """Daily RS-line new highs (any) and new highs before price, per bar.

    Mirrors rs_frame_signals: windows hold up to `lookback` bars, a window
    with a missing value gives no dot.
    """
    ref_close = benchmark["close"].reindex(stock.index)
    line = stock["close"] / ref_close
    size = pd.Series(np.minimum(np.arange(len(stock)) + 1, lookback), index=stock.index)
    line_valid = line.rolling(lookback, min_periods=1).count().eq(size)
    highs_valid = stock["high"].rolling(lookback, min_periods=1).count().eq(size)
    dots = line.eq(line.rolling(lookback, min_periods=1).max()) & line_valid
    before = dots & stock["high"].lt(
        stock["high"].rolling(lookback, min_periods=1).max()) & highs_valid
    return dots, before


def weekly_ema(closes, span=8):
    """EMA of weekly closes; the current week uses its latest close."""
    weekly = closes.dropna().resample("W-FRI").last().dropna()
    return weekly.ewm(span=span, adjust=False).mean()


def _round(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def _pct(price, level):
    if price is None or level is None or not np.isfinite(level) or level <= 0:
        return None
    return _round((price / level - 1) * 100)


def scan_ticker(candles, benchmark, settings=DEFAULTS):
    """Setup metrics for one stock; returns a dict (Status explains gaps)."""
    stock = daily_frame(candles)
    if stock.empty or stock["close"].notna().sum() < 60:
        return {"Status": "SHORT_HISTORY"}
    if benchmark.empty or stock.index[-1] != benchmark.index[-1]:
        return {"Status": "LATEST_DATES_DIFFER"}
    closes = stock["close"]
    price = closes.iloc[-1]
    if not np.isfinite(price):
        return {"Status": "MISSING_LATEST_CLOSE"}

    window = int(settings["SETUP_DOT_WINDOW"])
    dots, before = blue_dots(stock, benchmark)
    found = np.flatnonzero(dots.to_numpy())
    days_since = int(len(stock) - 1 - found[-1]) if len(found) else None
    dots_window = int(dots.tail(window).sum())
    before_window = int(before.tail(window).sum())

    clean = closes.dropna()
    ema21 = clean.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50 = clean.tail(50).mean() if len(clean) >= 50 else np.nan
    ema8w_series = weekly_ema(closes)
    ema8w = ema8w_series.iloc[-1] if len(ema8w_series) >= 8 else np.nan
    ema8w_rising = (bool(ema8w_series.iloc[-1] > ema8w_series.iloc[-2])
                    if len(ema8w_series) >= 9 else None)
    high_250 = stock["high"].tail(LOOKBACK).max()

    last, prev = stock.iloc[-1], stock.iloc[-2]
    inside = (bool(last["high"] <= prev["high"] and last["low"] >= prev["low"])
              if pd.notna([last["high"], last["low"], prev["high"], prev["low"]]).all()
              else None)

    volume = stock["volume"]
    vol50 = volume.tail(50).mean()
    vol_ratio = _round(volume.tail(5).mean() / vol50) if vol50 and vol50 > 0 else None
    dollar50 = (closes * volume).tail(50).mean()

    year = stock.tail(LOOKBACK)
    hv1_date = hv1_high = hv1_ago = None
    if year["volume"].notna().any():
        hv1_date = year["volume"].idxmax()
        hv1_high = year.loc[hv1_date, "high"]
        hv1_ago = int(len(stock) - 1 - stock.index.get_loc(hv1_date))

    pct21 = _pct(price, ema21)
    pct8w = _pct(price, ema8w)
    near = float(settings["SETUP_NEAR_PCT"])
    support = [name for name, pct in (("21 EMA", pct21), ("8W EMA", pct8w))
               if pct is not None and abs(pct) <= near]
    above50 = bool(price > sma50) if np.isfinite(sma50) else None

    strength = dots_window >= int(settings["SETUP_MIN_DOTS"])
    weakness = (days_since is not None and
                settings["SETUP_MIN_DAYS_SINCE_DOT"] <= days_since
                <= settings["SETUP_MAX_DAYS_SINCE_DOT"])
    trend = bool(ema8w_rising) and bool(above50)
    setup = "Pullback" if strength and weakness and support and trend else ""

    return {
        "Setup": setup, "Near_Support": " + ".join(support),
        "Price": _round(price), "Dots_In_Window": dots_window,
        "Before_Price_Dots_In_Window": before_window, "Days_Since_Dot": days_since,
        "Pct_From_EMA21": pct21, "Pct_From_EMA8W": pct8w,
        "EMA8W_Rising": ema8w_rising, "Above_SMA50": above50,
        "Pct_From_52W_High": _pct(price, high_250), "Inside_Day": inside,
        "Vol_5D_vs_50D": vol_ratio,
        "HV1_Date": hv1_date.strftime("%Y-%m-%d") if hv1_date is not None else None,
        "HV1_Sessions_Ago": hv1_ago, "HV1_High": _round(hv1_high),
        "Pct_From_HV1_High": _pct(price, hv1_high),
        "EMA21": _round(ema21), "EMA8W": _round(ema8w), "SMA50": _round(sma50),
        "AvgDollarVol50_M": _round(dollar50 / 1e6, 1) if np.isfinite(dollar50) else None,
        "As_Of": stock.index[-1].strftime("%Y-%m-%d"), "Status": "OK",
    }


def setups_view(price_data, stocks, benchmark_ticker="^GSPC", settings=DEFAULTS):
    """Scan stocks with RS rating >= SETUP_MIN_RS; returns the setups.csv frame."""
    benchmark = daily_frame(price_data.get(benchmark_ticker, {}).get("candles", []))
    stocks = stocks.copy()
    stocks["Percentile"] = pd.to_numeric(stocks["Percentile"], errors="coerce")
    stocks = stocks[stocks["Percentile"] >= settings["SETUP_MIN_RS"]]
    rows = []
    for info in stocks.to_dict("records"):
        ticker = info["Ticker"]
        candles = price_data.get(ticker, {}).get("candles", [])
        try:
            metrics = scan_ticker(candles, benchmark, settings)
        except Exception as exc:  # one bad series must not sink the scan
            print(f"setups: {ticker} skipped ({exc})")
            continue
        rows.append({
            "Ticker": ticker, **metrics, "RS_Rating": info.get("Percentile"),
            "Weekly_RS_Before_Price": info.get("RS_Before_Price_Weekly", ""),
            "Sector": info.get("Sector"), "Industry": info.get("Industry"),
            "MarketCap": info.get("MarketCap"),
        })
    frame = pd.DataFrame(rows, columns=SETUP_COLUMNS)
    if not frame.empty:
        frame["_setup"] = frame["Setup"].eq("Pullback")
        frame = frame.sort_values(["_setup", "RS_Rating"], ascending=False).drop(columns="_setup")
    return frame.reset_index(drop=True)


def load_settings():
    settings = dict(DEFAULTS)
    try:
        with open(os.path.join(DIR, "config.yaml"), "r") as stream:
            config = yaml.safe_load(stream) or {}
        for key in DEFAULTS:
            if config.get(key) is not None:
                settings[key] = float(config[key])
    except FileNotFoundError:
        pass
    return settings


def main():
    with open(PRICE_DATA, "r", encoding="utf-8") as fp:
        price_data = json.load(fp)
    stocks = pd.read_csv(RS_STOCKS, keep_default_na=False, na_values=[""])
    benchmark = "^GSPC"
    try:
        with open(os.path.join(DIR, "config.yaml"), "r") as stream:
            benchmark = (yaml.safe_load(stream) or {}).get("RS_LINE_REFERENCE_TICKER", benchmark)
    except FileNotFoundError:
        pass
    frame = setups_view(price_data, stocks, benchmark, load_settings())
    frame.to_csv(OUTPUT, index=False, encoding="utf-8")
    hits = frame[frame["Setup"].eq("Pullback")]
    print(f"✓ setups.csv: {len(frame)} RS leaders scanned, {len(hits)} pullback setups")
    if len(hits):
        print(",".join(hits["Ticker"].astype(str)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
