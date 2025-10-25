# Backstarter_streamlit_improved_with_real_indicator.py
# Improved Backtester for Streamlit — expanded warmup + option to use real indicator values/prices
# להרצה: pip install streamlit yfinance pandas numpy matplotlib
# ואז: streamlit run Backstarter_streamlit_improved_with_real_indicator.py

import sys
import subprocess
import importlib

required = ["streamlit", "yfinance", "pandas", "numpy", "matplotlib"]
for pkg in required:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO, StringIO
import math
from datetime import date, timedelta

st.set_page_config(page_title="Indicator Backtester — Improved (real-indicator option)", layout="wide")

# ------------------------
# Safe numeric coercion helper (prevents TypeError on weird column types)
# ------------------------
def ensure_numeric_series(x, index=None):
    """
    Convert x into pd.Series of floats aligned to index if provided.
    Robust to: Series with list cells, DataFrame columns, arrays, lists, scalars.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            # try to find first numeric-like column
            for c in x.columns:
                try:
                    s = pd.to_numeric(x[c], errors='coerce')
                    if s.notna().any():
                        x = s
                        break
                except Exception:
                    continue
            else:
                x = x.iloc[:, 0]

    if isinstance(x, pd.Series):
        try:
            return pd.to_numeric(x, errors='coerce').astype(float)
        except Exception:
            out = []
            for v in x:
                if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                    try:
                        arr = np.asarray(v).ravel()
                        out.append(arr[0] if arr.size > 0 else np.nan)
                    except Exception:
                        out.append(np.nan)
                else:
                    out.append(v)
            s = pd.Series(out, index=x.index)
            try:
                return pd.to_numeric(s, errors='coerce').astype(float)
            except Exception:
                coerced = []
                for v in out:
                    try:
                        coerced.append(float(v))
                    except Exception:
                        coerced.append(np.nan)
                return pd.Series(coerced, index=x.index, dtype=float)

    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            s = pd.Series(x, index=index) if (index is not None and len(x) == len(index)) else pd.Series(x)
            return pd.to_numeric(s, errors='coerce').astype(float)
        except Exception:
            coerced = []
            for v in x:
                try:
                    coerced.append(float(v))
                except Exception:
                    coerced.append(np.nan)
            if index is not None and len(coerced) == len(index):
                return pd.Series(coerced, index=index, dtype=float)
            return pd.Series(coerced, dtype=float)

    try:
        val = float(x)
        if index is not None:
            return pd.Series([val] * len(index), index=index, dtype=float)
        return pd.Series([val], dtype=float)
    except Exception:
        length = len(index) if index is not None else 1
        return pd.Series([np.nan] * length, index=index if index is not None else None, dtype=float)


# ------------------------
# Indicator implementations
# ------------------------
def ema(series, period):
    s = ensure_numeric_series(series)
    return s.ewm(span=period, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    macd_line = ema(df['Close'], fast) - ema(df['Close'], slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series, period=14, method='sma'):
    s = ensure_numeric_series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    if method == 'wilder':
        avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    else:
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_v = 100 - (100 / (1 + rs))
    rsi_v = rsi_v.fillna(50.0)
    return rsi_v

def cci(df, period=20):
    close = ensure_numeric_series(df['Close'])
    high = ensure_numeric_series(df['High'])
    low = ensure_numeric_series(df['Low'])
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_v = (tp - sma_tp) / (0.015 * mad)
    return cci_v

# ------------------------
# Robust Wilder RSI (used in backtester) - SMA start and safe reshaping
# ------------------------
def rsi_wilder(prices, period: int = 14) -> pd.Series:
    """
    RSI calculation using Wilder's smoothing with SMA start.
    Robust to receiving a DataFrame (will use first column) or a Series.
    Returns a pandas.Series aligned to the input Series index (NaN where undefined).
    """
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 0:
            return pd.Series(dtype="float64")
        prices = prices.iloc[:, 0]

    prices = pd.Series(prices).copy()
    orig_index = prices.index

    # Convert to numeric and drop NA for calculation core
    prices_numeric = pd.to_numeric(prices, errors='coerce')
    prices_no_na = prices_numeric.dropna()
    n = len(prices_no_na)

    result = pd.Series(index=orig_index, data=np.nan, dtype="float64")
    if n <= period:
        return result

    delta = prices_no_na.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    gain_vals = gain.to_numpy()
    loss_vals = loss.to_numpy()

    rsi_vals = np.full(len(gain_vals), np.nan, dtype="float64")

    # initial SMA averages on first 'period' deltas (indices 1..period)
    first_gain_avg = np.mean(gain_vals[1: period + 1])
    first_loss_avg = np.mean(loss_vals[1: period + 1])

    avg_gain = float(first_gain_avg)
    avg_loss = float(first_loss_avg)

    if avg_loss == 0.0:
        rsi_vals[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_vals[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(gain_vals)):
        g = gain_vals[i] if not np.isnan(gain_vals[i]) else 0.0
        l = loss_vals[i] if not np.isnan(loss_vals[i]) else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

        if avg_loss == 0.0:
            rsi_vals[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[i] = 100.0 - (100.0 / (1.0 + rs))

    # Build Series for the no-na index, then map back to original index
    rsi_series_no_na = pd.Series(rsi_vals, index=prices_no_na.index)
    for idx, val in rsi_series_no_na.items():
        result.at[idx] = val

    return result

# ------------------------
# Warmup helpers (same idea as before)
# ------------------------
def compute_warmup_bars(indicator_name, indicator_period, rsi_method):
    p = int(max(1, int(indicator_period)))
    if indicator_name == "RSI":
        if rsi_method == 'wilder':
            return max(200, p * 10)
        else:
            return max(120, p * 6)
    elif indicator_name == "CCI":
        return max(120, p * 6)
    elif indicator_name == "MACD":
        slow = 26
        return max(200, slow * 8)
    else:
        return max(120, p * 6)

def compute_warmup_days(warmup_bars, interval):
    if interval == "1d":
        return int(math.ceil(warmup_bars * 1.8)) + 10
    else:
        trading_bars_per_day = 6.5
        days = int(math.ceil((warmup_bars / trading_bars_per_day) * 2.0)) + 5
        return max(days, 30)

# ------------------------
# Utilities & Backtest core
# ------------------------
def _to_scalar_safe(x):
    if x is None:
        raise ValueError("None cannot be converted to scalar")
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, pd.Series):
        non_na = x.dropna()
        if non_na.empty:
            raise ValueError("No non-NaN scalar in Series")
        return float(non_na.iloc[0])
    try:
        arr = np.asarray(x)
    except Exception:
        pass
    else:
        if arr.size == 0:
            raise ValueError("Empty array")
        flat = arr.ravel()
        non_nan_idx = np.where(~np.isnan(flat))[0]
        if non_nan_idx.size == 0:
            raise ValueError("All elements are NaN")
        return float(flat[non_nan_idx[0]])
    return float(x)

def _get_indicator_value_at(indicator_series, idx):
    try:
        val = indicator_series.iloc[idx]
    except Exception:
        return np.nan
    if isinstance(val, (pd.Series, np.ndarray, list, tuple)):
        try:
            arr = np.asarray(val)
            if arr.size == 0:
                return np.nan
            non_nan_idx = np.where(~np.isnan(arr))[0]
            if non_nan_idx.size == 0:
                return np.nan
            return float(arr[non_nan_idx[0]])
        except Exception:
            if isinstance(val, pd.Series):
                non_na = val.dropna()
                if non_na.empty:
                    return np.nan
                return float(non_na.iloc[0])
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

# ------------------------
# Updated run_backtest (supports 'adj_close' execution)
# ------------------------
def run_backtest(df, indicator_series, low_thresh, high_thresh,
                 entry_exec='close', exit_exec='close',
                 sizing_mode='fixed', fixed_amount=1000, initial_capital=10000,
                 allow_multiple_entries=False, commission_mode='none',
                 commission_value=0.0, exclude_incomplete=True,
                 defer_until_next_cross_with_price_higher=False,
                 debug_mode=False):
    """
    Main backtest loop with enhanced debugging.
    entry_exec / exit_exec supported values:
      - 'close'      -> use df['Close'] at current bar
      - 'next_open'  -> use df['next_open'] if present, else df['Open'].iloc[i+1] (if in range)
      - 'adj_close'  -> use df['Adj Close'] if present, else fallback to df['Close']
    Returns: trades_df, baseline_summary, equity_curve, open_positions_list, debug_log
    """
    if isinstance(indicator_series, pd.DataFrame):
        indicator_series = indicator_series.iloc[:, 0]
    try:
        indicator_series = pd.Series(indicator_series, index=df.index).reindex(df.index)
    except Exception:
        indicator_series = pd.Series(list(indicator_series), index=df.index).reindex(df.index)

    trades = []
    positions = []
    cumulative_realized_pnl = 0.0

    equity_curve = pd.Series(index=df.index, dtype=float)

    debug_log = [] if debug_mode else None

    def _estimate_periods_per_year(index):
        if len(index) < 2:
            return 252
        deltas = np.diff(index.astype('int64')) / 1e9
        median_delta = np.median(deltas)
        if median_delta >= 23 * 3600:
            return 252
        elif median_delta >= 3600:
            return int(252 * 6.5)
        else:
            return 252

    periods_per_year = _estimate_periods_per_year(df.index)

    # Pre-check whether Adj Close exists
    has_adj = "Adj Close" in df.columns

    for i in range(len(df)):
        date_idx = df.index[i]
        ind_val_float = _get_indicator_value_at(indicator_series, i)

        # update per-position max_price from previous bar's High
        for pos in positions:
            if i > pos['entry_i']:
                try:
                    prior_high = float(df['High'].iloc[i - 1])
                    pos['max_price'] = max(pos.get('max_price', pos['entry_price']), prior_high)
                except Exception:
                    pos['max_price'] = pos.get('max_price', pos['entry_price'])

        # compute previous indicator value early
        prev_ind_val_float = None
        if i > 0:
            prev_ind_val_float = _get_indicator_value_at(indicator_series, i - 1)
            if pd.isna(prev_ind_val_float):
                prev_ind_val_float = None

        if pd.isna(ind_val_float):
            # compute equity at this bar (no change in realized pnl)
            unrealized = 0.0
            for pos in positions:
                try:
                    current_price = float(df['Close'].iloc[i])
                    unrealized += (current_price - pos['entry_price']) * pos['shares']
                except Exception:
                    pass
            equity_curve.iloc[i] = initial_capital + cumulative_realized_pnl + unrealized
            if debug_mode:
                debug_log.append(f"BAR {i} {date_idx} - indicator=NaN - positions={len(positions)}")
            continue

        # ----- ENTRY LOGIC ----- #
        try:
            low_thresh_f = float(low_thresh)
        except Exception:
            low_thresh_f = _to_scalar_safe(low_thresh)
        try:
            high_thresh_f = float(high_thresh)
        except Exception:
            high_thresh_f = _to_scalar_safe(high_thresh)

        entered_this_bar = False

        if ind_val_float < low_thresh_f:
            # decide entry price
            entry_price_raw = None

            if entry_exec == 'close':
                entry_price_raw = df['Close'].iloc[i] if 'Close' in df.columns else None
            elif entry_exec == 'next_open':
                if 'next_open' in df.columns:
                    try:
                        entry_price_raw = df['next_open'].iloc[i]
                    except Exception:
                        entry_price_raw = None
                else:
                    # try Open of next bar if available
                    if (i + 1) < len(df) and 'Open' in df.columns:
                        try:
                            entry_price_raw = df['Open'].iloc[i + 1]
                        except Exception:
                            entry_price_raw = None
                    else:
                        entry_price_raw = None
            elif entry_exec == 'adj_close':
                if has_adj:
                    entry_price_raw = df['Adj Close'].iloc[i]
                else:
                    # fallback to Close if Adj not present
                    if 'Close' in df.columns:
                        entry_price_raw = df['Close'].iloc[i]
                        if debug_mode:
                            debug_log.append(f"BAR {i} {date_idx} - Adj Close not present, falling back to Close for entry")
                    else:
                        entry_price_raw = None
            else:
                # unknown mode: fallback to Close
                entry_price_raw = df['Close'].iloc[i] if 'Close' in df.columns else None
                if debug_mode:
                    debug_log.append(f"BAR {i} {date_idx} - unknown entry_exec '{entry_exec}', fallback to Close")

            try:
                entry_price = float(entry_price_raw) if entry_price_raw is not None else None
            except Exception:
                entry_price = None

            if entry_price is not None:
                if sizing_mode == 'fixed':
                    try:
                        shares = int(float(fixed_amount) // entry_price)
                    except Exception:
                        shares = int(math.floor(float(fixed_amount) / entry_price)) if entry_price != 0 else 0
                    if shares > 0:
                        invested = shares * entry_price
                        if commission_mode == 'percent':
                            commission_entry = invested * commission_value
                        elif commission_mode == 'absolute':
                            commission_entry = commission_value
                        else:
                            commission_entry = 0.0

                        if not allow_multiple_entries and len(positions) > 0:
                            if debug_mode:
                                debug_log.append(f"BAR {i} {date_idx} - entry skipped (pyramiding disabled)")
                        else:
                            pos = {
                                'entry_i': i,
                                'entry_date': date_idx,
                                'entry_price': entry_price,
                                'entry_indicator': ind_val_float,
                                'shares': shares,
                                'invested': invested,
                                'commission_entry': commission_entry,
                                'max_price': entry_price,
                                'deferred': False
                            }
                            positions.append(pos)
                            entered_this_bar = True
                            if debug_mode:
                                debug_log.append(f"BAR {i} {date_idx} - ENTER fixed shares={shares} invested={invested}")
                    else:
                        if debug_mode:
                            debug_log.append(f"BAR {i} {date_idx} - entry computed shares=0 (fixed_amount too small)")
                else:  # compound
                    if entry_price != 0:
                        current_unrealized = 0.0
                        for p in positions:
                            try:
                                current_unrealized += (float(df['Close'].iloc[i]) - p['entry_price']) * p['shares']
                            except Exception:
                                pass
                        current_equity = initial_capital + cumulative_realized_pnl + current_unrealized
                        if len(positions) == 0 and current_equity > 0:
                            shares = current_equity / entry_price
                            invested = current_equity
                            pos = {
                                'entry_i': i,
                                'entry_date': date_idx,
                                'entry_price': entry_price,
                                'entry_indicator': ind_val_float,
                                'shares': shares,
                                'invested': invested,
                                'commission_entry': 0.0,
                                'max_price': entry_price,
                                'deferred': False
                            }
                            if commission_mode == 'percent':
                                pos['commission_entry'] = invested * commission_value
                            elif commission_mode == 'absolute':
                                pos['commission_entry'] = commission_value
                            positions.append(pos)
                            entered_this_bar = True
                            if debug_mode:
                                debug_log.append(f"BAR {i} {date_idx} - ENTER compound shares={shares:.4f} invested={invested}")
                        else:
                            if debug_mode:
                                debug_log.append(f"BAR {i} {date_idx} - compound entry skipped (already in position or equity<=0)")
            else:
                if debug_mode:
                    debug_log.append(f"BAR {i} {date_idx} - entry_price unavailable (mode={entry_exec})")

        # ----- EXIT LOGIC ----- #
