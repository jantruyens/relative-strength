"""Append five Yahoo statement quarters to the existing four-RS-signal CSV.

EPS is DilutedEPS from quarterly statements, not adjusted earnings.actual.
Existing RS exports remain untouched. Run this after an RS CSV is published.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE_VERSION = 1
QUARTER_COLUMNS = [name for n in range(5) for name in (
    f"Quarter_End_Q{n}", f"EPS_Diluted_Q{n}", f"Sales_Millions_Q{n}")]
FUND_COLUMNS = QUARTER_COLUMNS + [
    "EPS_YoY_Latest_Pct", "Sales_YoY_Latest_Pct", "Financials_Source",
    "EPS_Basis", "Financial_Currency", "Financials_Last_Fetched_UTC",
    "Financials_Last_Attempt_UTC", "Financials_Refresh_Status",
    "Financials_Refresh_Error", "Financials_Quarters_Available",
    "Financials_Period_Status",
]
SIGNAL_COLUMNS = ["RS_Before_Price_Daily", "RS_Before_Price_Weekly",
                  "RS_New_High_Daily", "RS_New_High_Weekly"]


HISTORY_COLUMNS = ['Historical_RS_New_High_Daily', 'Historical_RS_Before_Price_Daily', 'Historical_RS_New_High_Weekly', 'Historical_RS_Before_Price_Weekly', 'Last_RS_New_High_Daily_Date', 'Last_RS_Before_Price_Daily_Date', 'Last_RS_New_High_Weekly_Date', 'Last_RS_Before_Price_Weekly_Date', 'Trading_Days_Since_RS_New_High', 'Trading_Days_Since_RS_Before_Price', 'Weekly_Bars_Since_RS_New_High', 'Weekly_Bars_Since_RS_Before_Price', 'Historical_Daily_Status', 'Historical_Weekly_Status', 'Historical_Daily_Bars_Checked', 'Historical_Weekly_Bars_Checked']


def attach_history(stocks, history):
    """Append 16 keyed historical fields without moving the first 50 columns.

    A sidecar from a different RS snapshot must not silently be joined to the
    current prices. An absent sidecar is supported during installation.
    """
    if set(stocks.columns).intersection(HISTORY_COLUMNS):
        raise ValueError("Historical columns are already attached.")
    output = stocks.copy()
    for column in HISTORY_COLUMNS:
        output[column] = None
    if history is None:
        output["Historical_Daily_Status"] = "HISTORY_NOT_AVAILABLE"
        output["Historical_Weekly_Status"] = "HISTORY_NOT_AVAILABLE"
        return output
    required = {"Ticker", "History_As_Of", *HISTORY_COLUMNS}
    if not required.issubset(history.columns) or history["Ticker"].isna().any() or history["Ticker"].duplicated().any():
        raise ValueError("Historical sidecar schema/ticker keys are invalid.")
    matched = history.set_index("Ticker").reindex(stocks["Ticker"])
    for column in HISTORY_COLUMNS:
        output[column] = matched[column].to_numpy()
    current_dates = pd.to_datetime(stocks["Daily_Signal_Date"], errors="coerce", utc=True)
    sidecar_dates = pd.to_datetime(matched["History_As_Of"].to_numpy(), errors="coerce", utc=True)
    sidecar_dates = pd.Series(sidecar_dates, index=stocks.index)
    missing = ~stocks["Ticker"].isin(history["Ticker"])
    unavailable = current_dates.isna() | sidecar_dates.isna()
    stale = current_dates.ne(sidecar_dates) & ~unavailable
    for mask, reason in ((unavailable, "SNAPSHOT_DATE_UNAVAILABLE"),
                         (stale, "STALE_HISTORY_SNAPSHOT"),
                         (missing, "HISTORY_NOT_AVAILABLE")):
        output.loc[mask, HISTORY_COLUMNS] = None
        output.loc[mask, "Historical_Daily_Status"] = reason
        output.loc[mask, "Historical_Weekly_Status"] = reason
    return output


def utc_now():
    return datetime.now(timezone.utc)


def timestamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (AttributeError, ValueError, TypeError):
        return None


def finite_number(value):
    """Retain genuine negative/zero values; reject booleans and NaN/inf."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def valid_currency(value):
    if isinstance(value, str) and len(value) == 3 and value.isalpha():
        return value.upper()
    return None


def normalize_statement(statement, as_of=None):
    """Extract actual quarterly DilutedEPS/TotalRevenue, latest five dates.

    No basic-EPS fallback, annual/TTM substitutions, or zero-filling.
    We retain period dates even when either requested metric is missing.
    """
    if statement is None or not isinstance(statement, pd.DataFrame) or statement.empty:
        raise ValueError("NO_QUARTERLY_STATEMENT")
    as_of = as_of or utc_now().date()
    if not statement.index.is_unique:
        raise ValueError("DUPLICATE_STATEMENT_ROWS")
    periods = {}
    for position, raw_date in enumerate(statement.columns):
        if isinstance(raw_date, (int, float, bool)):
            continue
        try:
            end = pd.Timestamp(raw_date).date()
        except (TypeError, ValueError):
            continue
        if pd.isna(end) or end > as_of:
            continue
        key = end.isoformat()
        if key in periods:
            raise ValueError("DUPLICATE_FISCAL_PERIODS")
        column = statement.iloc[:, position]
        periods[key] = {
            "end": key,
            "eps_diluted": finite_number(column.get("DilutedEPS")),
            "sales": finite_number(column.get("TotalRevenue")),
        }
    if not periods:
        raise ValueError("NO_VALID_FISCAL_PERIODS")
    ordered = [periods[key] for key in sorted(periods, reverse=True)[:5]]
    if any((date.fromisoformat(ordered[i]["end"]) - date.fromisoformat(ordered[i+1]["end"])).days < 45
           for i in range(len(ordered)-1)):
        raise ValueError("OVERLAPPING_QUARTER_DATES")
    if not any(p["eps_diluted"] is not None or p["sales"] is not None for p in ordered):
        raise ValueError("NO_USABLE_EPS_OR_SALES")
    return ordered


def fetch_financials(symbol, ticker_factory=None, as_of=None):
    """Use public yfinance methods; tests inject a fake ticker factory."""
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker
    ticker = ticker_factory(symbol.replace(".", "-"))
    periods = normalize_statement(ticker.get_income_stmt(freq="quarterly"), as_of)
    currency = None
    currency_error = None
    try:
        info = ticker.get_info()
        currency = valid_currency(info.get("financialCurrency"))
    except Exception as exc:
        if "ratelimit" in type(exc).__name__.lower() or "429" in str(exc):
            raise
        currency_error = "CURRENCY_LOOKUP_FAILED"
    return {"periods": periods, "currency": currency,
            "currency_error": currency_error}


def load_cache(path):
    if not path.exists():
        return {"version": CACHE_VERSION, "tickers": {}}
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("version") != CACHE_VERSION or not isinstance(result.get("tickers"), dict):
        raise ValueError("Unrecognized fundamentals cache; preserve it for inspection.")
    return result


def save_cache(cache, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def is_dot(value):
    return isinstance(value, str) and value.strip() == "🔵"


def select_refresh(stocks, cache, now, max_symbols):
    """Reserve 20% of each batch for general coverage; prioritize RS dots.

    Dot stocks are due every 3 days; other stocks every 14 days. A failed
    attempt is retried no sooner than 24h. Never-fetched/older rows sort first.
    The reserved general queue prevents non-dot stocks being starved forever.
    """
    if max_symbols < 1:
        return []
    due = []
    entries = cache["tickers"]
    for order, (_, stock) in enumerate(stocks.iterrows()):
        symbol = stock["Ticker"]
        entry = entries.get(symbol, {})
        success = parse_timestamp(entry.get("last_success"))
        attempt = parse_timestamp(entry.get("last_attempt"))
        has_dot = any(is_dot(stock.get(column)) for column in SIGNAL_COLUMNS)
        if entry.get("last_error") and attempt and now - attempt < timedelta(hours=24):
            continue
        if success and now - success < timedelta(days=3 if has_dot else 14):
            continue
        # No successful record precedes all cached records; use attempts for
        # fair ordering when a symbol repeatedly returns no statements.
        age_key = success or attempt or datetime.min.replace(tzinfo=timezone.utc)
        capitalization = finite_number(stock.get("MarketCap")) or 0
        due.append((symbol, has_dot, age_key, -capitalization, order))
    priority = sorted((r for r in due if r[1]), key=lambda r: (r[2], r[3], r[4]))
    priority_slots = max(1, int(max_symbols * .8))
    chosen = [r[0] for r in priority[:priority_slots]]
    selected = set(chosen)
    general = sorted((r for r in due if r[0] not in selected), key=lambda r: (r[2], r[3], r[4]))
    chosen.extend(r[0] for r in general[:max_symbols - len(chosen)])
    return chosen


def growth(current, previous):
    current, previous = finite_number(current), finite_number(previous)
    if current is None or previous is None or previous == 0:
        return None
    result = (current - previous) / abs(previous) * 100
    return round(result, 4) if math.isfinite(result) else None


def find_period(periods, target, tolerance_days=14):
    matches = [p for p in periods if abs((date.fromisoformat(p["end"]) - target).days) <= tolerance_days]
    return matches[0] if len(matches) == 1 else None


def financial_columns(entry, as_of=None):
    as_of = as_of or utc_now().date()
    output = dict.fromkeys(FUND_COLUMNS)
    output.update(Financials_Source="Yahoo Finance",
                  EPS_Basis="Statement diluted EPS (not adjusted reported EPS)")
    if not entry:
        output.update(Financials_Refresh_Status="NOT_FETCHED", Financials_Period_Status="NO_HISTORY",
                      Financials_Quarters_Available=0)
        return output
    output.update(Financial_Currency=entry.get("currency"),
                  Financials_Last_Fetched_UTC=entry.get("last_success"),
                  Financials_Last_Attempt_UTC=entry.get("last_attempt"),
                  Financials_Refresh_Error=entry.get("last_error"))
    periods = entry.get("periods", [])[:5]
    output["Financials_Quarters_Available"] = len(periods)
    output["Financials_Refresh_Status"] = (
        "ERROR_CACHED" if entry.get("last_error") and periods else
        "ERROR_NO_DATA" if entry.get("last_error") else "CACHED")
    if not periods:
        output["Financials_Period_Status"] = "NO_HISTORY"
        return output
    newest = max(date.fromisoformat(p["end"]) for p in periods)
    issues = set()
    if not entry.get("currency"):
        issues.add("CURRENCY_UNKNOWN")
    if len(periods) < 5:
        issues.add("SHORT_HISTORY")
    if (as_of - newest).days > 180:
        issues.add("OLD_FISCAL_PERIOD")
    for n in range(5):
        expected = (pd.Timestamp(newest) - pd.DateOffset(months=3*n)).date()
        period = find_period(periods, expected)
        if period is None:
            issues.add("MISSING_QUARTER")
            continue
        output[f"Quarter_End_Q{n}"] = period["end"]
        output[f"EPS_Diluted_Q{n}"] = period["eps_diluted"]
        output[f"Sales_Millions_Q{n}"] = (
            period["sales"] / 1_000_000 if period["sales"] is not None else None)
        if period["eps_diluted"] is None:
            issues.add("EPS_MISSING")
        if period["sales"] is None:
            issues.add("SALES_MISSING")
    output["Financials_Quarters_Available"] = sum(
        output[f"Quarter_End_Q{n}"] is not None for n in range(5))
    # Only latest-quarter YoY is exported. Five quarters cannot establish
    # five YoY comparisons. Date matching prevents comparing wrong quarters.
    current = find_period(periods, newest)
    year_before = (pd.Timestamp(newest) - pd.DateOffset(years=1)).date()
    baseline = find_period(periods, year_before)
    if baseline and entry.get("currency"):
        output["EPS_YoY_Latest_Pct"] = growth(current["eps_diluted"], baseline["eps_diluted"])
        output["Sales_YoY_Latest_Pct"] = growth(current["sales"], baseline["sales"])
        if baseline["eps_diluted"] == 0:
            issues.add("ZERO_BASE_EPS")
        if baseline["sales"] == 0:
            issues.add("ZERO_BASE_SALES")
    elif baseline is None:
        issues.add("YOY_BASE_MISSING")
    output["Financials_Period_Status"] = ";".join(sorted(issues)) if issues else "OK"
    return output


def append_financials(stocks, cache, as_of=None):
    if "Ticker" not in stocks or stocks["Ticker"].isna().any() or stocks["Ticker"].duplicated().any():
        raise ValueError("RS input requires unique, nonempty Ticker values.")
    overlap = set(stocks.columns).intersection(FUND_COLUMNS)
    if overlap:
        raise ValueError("RS input already has financial columns; use rs_signals.csv.")
    extra = pd.DataFrame([financial_columns(cache["tickers"].get(symbol), as_of)
                          for symbol in stocks["Ticker"]], columns=FUND_COLUMNS, index=stocks.index)
    return pd.concat([stocks, extra], axis=1)


def write_exports(frame, directory):
    """Full CSV plus six fixed-size partitions for smaller Sheets imports."""
    directory.mkdir(parents=True, exist_ok=True)
    names_and_frames = [("rs_signals_fundamentals.csv", frame)]
    size = max(1, math.ceil(len(frame) / 6))
    names_and_frames.extend((f"rs_signals_fundamentals_{n+1}.csv", frame.iloc[n*size:(n+1)*size])
                            for n in range(6))
    for index, (name, data) in enumerate(names_and_frames):
        destination = directory / name
        temporary = destination.with_suffix(".csv.tmp")
        # Full file and first partition have headers. Later partitions do not,
        # so their column-name strings cannot interfere with Sheets QUERY's
        # numeric type inference for sparsely populated EPS/sales columns.
        include_header = index <= 1
        if not include_header:
            blank = dict.fromkeys(frame.columns)
            blank["Rank"] = 0
            blank["Financials_Period_Status"] = "EMPTY_PART"
            if frame.columns[-1] == "Historical_Weekly_Bars_Checked":
                blank[frame.columns[-1]] = 0  # Retain all 66 columns, with a numeric sentinel.
            # A footer also retains width when all history values are blank.
            # The existing import formula removes rows with no Ticker.
            data = pd.concat([data, pd.DataFrame([blank], columns=frame.columns)], ignore_index=True)
        data.to_csv(temporary, index=False, encoding="utf-8", header=include_header)
        temporary.replace(destination)


def run_refresh(stocks, cache, cache_path, max_symbols=500, budget_seconds=1500,
                fetcher=fetch_financials, now=None, monotonic=time.monotonic,
                pause=time.sleep):
    now = now or utc_now()
    chosen = select_refresh(stocks, cache, now, max_symbols)
    started = monotonic()
    count = 0
    for symbol in chosen:
        if monotonic() - started >= budget_seconds:
            break
        entry = cache["tickers"].setdefault(symbol, {})
        entry["last_attempt"] = timestamp(now)
        stop_for_rate_limit = False
        try:
            value = fetcher(symbol, as_of=now.date())
            if not value.get("periods"):
                raise ValueError("NO_QUARTERLY_STATEMENT")
            entry.update(value, last_success=timestamp(now), last_error=None)
            print(f"Fundamentals {symbol}: {len(value['periods'])} quarters")
        except Exception as exc:
            stop_for_rate_limit = "ratelimit" in type(exc).__name__.lower() or "429" in str(exc)
            entry["last_error"] = "RATE_LIMIT" if stop_for_rate_limit else (
                str(exc) if isinstance(exc, ValueError) and str(exc).isupper() else type(exc).__name__)
            print(f"Fundamentals {symbol}: {entry['last_error']}; cached values retained")
        count += 1
        if count % 10 == 0 or stop_for_rate_limit:
            save_cache(cache, cache_path)
        if stop_for_rate_limit:
            break
        pause(.5)
    save_cache(cache, cache_path)
    print(f"Fundamentals attempted: {count}; daily cap: {max_symbols}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-symbols", type=int, default=500)
    parser.add_argument("--budget-minutes", type=float, default=25)
    args = parser.parse_args()
    if args.max_symbols < 1 or args.budget_minutes <= 0:
        parser.error("Refresh cap and time budget must be positive.")
    input_file = ROOT / "output/rs_signals.csv"
    # NA is a valid listed symbol. Only empty CSV fields mean missing here.
    stocks = pd.read_csv(input_file, keep_default_na=False, na_values=[""])
    if stocks.empty or not set(SIGNAL_COLUMNS).issubset(stocks.columns):
        raise ValueError("Run the four-RS-signal generator first.")
    cache_path = ROOT / "data_persist/fundamentals_cache.json"
    cache = load_cache(cache_path)
    run_refresh(stocks, cache, cache_path, args.max_symbols, args.budget_minutes * 60)
    enriched = append_financials(stocks, cache)
    history_path = ROOT / "output/rs_signals_history.csv"
    history = pd.read_csv(history_path, keep_default_na=False, na_values=[""]) if history_path.exists() else None
    enriched = attach_history(enriched, history)
    value_columns = [f"{field}_Q{n}" for n in range(5)
                     for field in ("EPS_Diluted", "Sales_Millions")]
    known = enriched[value_columns].notna().any(axis=1).sum()
    print(f"Published-file candidate: {known}/{len(enriched)} rows with quarterly history")
    if known == 0:
        raise SystemExit("No quarterly histories available. Inspect errors; existing RS files are untouched.")
    write_exports(enriched, ROOT / "output")


if __name__ == "__main__":
    main()
