#!/usr/bin/env python
"""Daily gap-up scan (Oliver Kell's gapper criteria) from the stored candles.

Runs after relative-strength.py, reusing data/price_history.json, so it costs
no extra downloads. Writes output/gappers.csv: one row per stock that gapped
up on heavy volume in the last GAP_LOOKBACK_DAYS sessions (its latest such gap).

A gap here is the session open vs the previous close (the same definition as
Finviz's Gap filter). Long-only: gap downs are ignored.

Gap type is judged from where the stock stood BEFORE the gap, in ATR units
above its 50-day average (the sheet's "ATR x from 50-day"):
  Breakaway        <= 2x  gapped out of a base / from near the 50-day
  Continuation     2-4x   gapped inside an existing uptrend
  Exhaustion risk  > 4x   gapped when already stretched
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

DIR = os.path.dirname(os.path.realpath(__file__))
PRICE_DATA = os.path.join(DIR, "data", "price_history.json")
RS_STOCKS = os.path.join(DIR, "output", "rs_stocks.csv")
OUTPUT = os.path.join(DIR, "output", "gappers.csv")

DEFAULTS = {
    "GAP_MIN_PCT": 3.0,          # open at least 3% above the previous close
    "GAP_MIN_RVOL": 3.0,         # gap-day volume >= 3x the prior 50-day average
    "GAP_MIN_PRICE": 20.0,       # gap-day close
    "GAP_MIN_VOLUME": 500000,    # gap-day shares traded
    "GAP_LOOKBACK_DAYS": 10,     # sessions to look back (two weeks)
}

GAPPER_COLUMNS = [
    "Ticker", "Gap_Date", "Sessions_Ago", "Gap_Pct", "RVOL", "Gap_Day_Volume",
    "Gap_Day_Change_Pct", "Close_In_Range", "Close_Quality", "Held_Gap_Intraday",
    "Pre_Gap_ATR_Ext", "Gap_Type", "Pre_Gap_Pct_From_52W_High",
    "Pre_Gap_Above_200", "Pre_Gap_200_Rising", "Gap_Fill_Level", "Gap_Filled",
    "Price", "Pct_Since_Gap_Close", "ATR_Ext_Now", "Gaps_In_Window",
    "Sector", "Industry", "RS_Rating", "MarketCap", "AvgDollarVol50_M", "As_Of",
]


def daily_frame(candles):
    """Candles -> date-indexed OHLCV frame, oldest first, rows without a close dropped."""
    frame = pd.DataFrame(candles)
    if frame.empty or "close" not in frame or "datetime" not in frame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    stamps = frame["datetime"]
    if pd.api.types.is_numeric_dtype(stamps):
        dates = pd.to_datetime(stamps, unit="s", utc=True).dt.tz_localize(None)
    else:
        dates = pd.to_datetime(stamps, errors="coerce")
    frame = frame.assign(date=dates.dt.normalize())
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0]
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    return frame.set_index("date")[["open", "high", "low", "close", "volume"]]


def true_range(frame):
    prev = frame["close"].shift(1)
    ranges = pd.concat([frame["high"] - frame["low"],
                        (frame["high"] - prev).abs(),
                        (frame["low"] - prev).abs()], axis=1)
    return ranges.max(axis=1, skipna=False)


def gap_type(ext):
    if ext is None or not np.isfinite(ext):
        return "Unknown"
    if ext > 4:
        return "Exhaustion risk"
    if ext <= 2:
        return "Breakaway"
    return "Continuation"


def close_quality(close_in_range, change_from_open):
    if close_in_range is None:
        return "Unknown"
    if close_in_range >= 0.75 and change_from_open >= 0:
        return "Strong close"
    if close_in_range < 0.4 or change_from_open < 0:
        return "Faded"
    return "OK"


def _round(value, digits):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def scan_ticker(candles, settings=DEFAULTS):
    """Latest qualifying gap-up for one stock in the lookback window, or None."""
    frame = daily_frame(candles)
    n = len(frame)
    if n < 51:  # need 50 prior sessions for the volume average and 50-day
        return None
    o, h, l = frame["open"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy()
    c, v = frame["close"].to_numpy(), frame["volume"].to_numpy()
    tr = true_range(frame).to_numpy()
    dates = frame.index

    lookback = int(settings["GAP_LOOKBACK_DAYS"])
    hits = []
    for i in range(max(50, n - lookback), n):
        prev_close = c[i - 1]
        if not (np.isfinite(o[i]) and np.isfinite(v[i]) and prev_close > 0):
            continue
        gap = o[i] / prev_close - 1
        prior_vol = v[i - 50:i]
        prior_vol = prior_vol[np.isfinite(prior_vol)]
        if len(prior_vol) < 40 or prior_vol.mean() <= 0:
            continue
        rvol = v[i] / prior_vol.mean()
        if (gap * 100 >= settings["GAP_MIN_PCT"] and rvol >= settings["GAP_MIN_RVOL"]
                and c[i] >= settings["GAP_MIN_PRICE"] and v[i] >= settings["GAP_MIN_VOLUME"]):
            hits.append((i, gap, rvol))
    if not hits:
        return None

    i, gap, rvol = hits[-1]
    prev_close = c[i - 1]
    sma50_before = np.nanmean(c[i - 50:i])
    atr_before = np.nanmean(tr[max(1, i - 14):i])
    ext_before = (prev_close - sma50_before) / atr_before if atr_before > 0 else None

    day_range = h[i] - l[i]
    in_range = (c[i] - l[i]) / day_range if day_range > 0 else 1.0
    change_from_open = c[i] / o[i] - 1 if o[i] > 0 else 0.0

    above_200 = rising_200 = None
    if i >= 220:
        sma200 = np.nanmean(c[i - 200:i])
        sma200_ago = np.nanmean(c[i - 220:i - 20])
        above_200 = bool(prev_close > sma200)
        rising_200 = bool(sma200 > sma200_ago)
    high52 = np.nanmax(h[max(0, i - 252):i])
    from_high = (prev_close / high52 - 1) * 100 if high52 > 0 else None

    # Unfilled gap: no close back below the pre-gap close since (gap day included).
    filled = bool((c[i:] < prev_close).any())

    dollar_vol = np.nanmean((c * v)[-50:]) / 1e6

    sma50_now = np.nanmean(c[-50:])
    atr_now = np.nanmean(tr[-14:])
    ext_now = (c[-1] - sma50_now) / atr_now if atr_now > 0 else None

    return {
        "Gap_Date": dates[i].strftime("%Y-%m-%d"),
        "Sessions_Ago": n - 1 - i,
        "Gap_Pct": _round(gap * 100, 2),
        "RVOL": _round(rvol, 2),
        "Gap_Day_Volume": int(v[i]),
        "Gap_Day_Change_Pct": _round((c[i] / prev_close - 1) * 100, 2),
        "Close_In_Range": _round(in_range, 2),
        "Close_Quality": close_quality(in_range, change_from_open),
        "Held_Gap_Intraday": bool(l[i] > prev_close) if np.isfinite(l[i]) else None,
        "Pre_Gap_ATR_Ext": _round(ext_before, 2),
        "Gap_Type": gap_type(ext_before),
        "Pre_Gap_Pct_From_52W_High": _round(from_high, 2),
        "Pre_Gap_Above_200": above_200,
        "Pre_Gap_200_Rising": rising_200,
        "Gap_Fill_Level": _round(prev_close, 2),
        "Gap_Filled": filled,
        "Price": _round(c[-1], 2),
        "Pct_Since_Gap_Close": _round((c[-1] / c[i] - 1) * 100, 2),
        "ATR_Ext_Now": _round(ext_now, 2),
        "Gaps_In_Window": len(hits),
        "AvgDollarVol50_M": _round(dollar_vol, 1),
        "As_Of": dates[-1].strftime("%Y-%m-%d"),
    }


def gapper_view(price_data, stocks, settings=DEFAULTS):
    """Scan every stock in rs_stocks.csv; returns the gappers.csv frame."""
    meta = stocks.set_index("Ticker")
    rows = []
    for ticker in meta.index:
        candles = price_data.get(ticker, {}).get("candles", [])
        try:
            hit = scan_ticker(candles, settings)
        except Exception as exc:  # one bad series must not sink the scan
            print(f"gappers: {ticker} skipped ({exc})")
            continue
        if hit is None:
            continue
        info = meta.loc[ticker]
        rows.append({
            "Ticker": ticker, **hit,
            "Sector": info.get("Sector"), "Industry": info.get("Industry"),
            "RS_Rating": info.get("Percentile"), "MarketCap": info.get("MarketCap"),
        })
    frame = pd.DataFrame(rows, columns=GAPPER_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["Sessions_Ago", "RVOL"], ascending=[True, False])
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
    frame = gapper_view(price_data, stocks, load_settings())
    frame.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"✓ gappers.csv: {len(frame)} gap-ups in the last "
          f"{int(load_settings()['GAP_LOOKBACK_DAYS'])} sessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
