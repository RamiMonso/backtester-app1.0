# backtester_streamlit_improved_with_real_indicator.py
# Improved Backtester for Streamlit — expanded warmup + option to use real indicator values/prices
# להרצה: pip install streamlit yfinance pandas numpy matplotlib
# ואז: streamlit run backtester_streamlit_improved_with_real_indicator.py

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

st.set_page_config(page_title="Indicator Backtester — Improved (real-indicator option)", layout="wide")

# ------------------------
# Utility: robust numeric coercion and column normalization
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


def normalize_df_columns(df):
    """
    Robustly detect important columns (Open/High/Low/Close/Adj Close/Volume), rename them consistently and coerce numeric.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("normalize_df_columns expects a DataFrame")
    df = df.copy()
    col_map = {}
    for c in df.columns:
        try:
            key = str(c).lower().replace(" ", "").replace("_", "").replace("-", "")
        except Exception:
            key = repr(c).lower()
        col_map[key] = c

    def find_col(possible_names):
        for name in possible_names:
            name_norm = name.lower().replace(" ", "").replace("_", "").replace("-", "")
            for k, orig in col_map.items():
                if name_norm in k:
                    return orig
        return None

    found_open = find_col(["open", "o"])
    found_high = find_col(["high", "h"])
    found_low = find_col(["low", "l"])
    found_close = find_col(["adjclose", "adj_close", "adj", "close", "c"])
    found_adj_close = None
    for k, orig in col_map.items():
        if 'adj' in k and 'close' in k:
            found_adj_close = orig
            break
    found_volume = find_col(["volume", "vol"])

    rename_map = {}
    if found_open is not None:
        rename_map[found_open] = "Open"
    if found_high is not None:
        rename_map[found_high] = "High"
    if found_low is not None:
        rename_map[found_low] = "Low"
    if found_close is not None:
        rename_map[found_close] = "Close"
    if found_adj_close is not None:
        rename_map[found_adj_close] = "Adj Close"
    if found_volume is not None:
        rename_map[found_volume] = "Volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = ensure_numeric_series(df[c], index=df.index)
    return df

# ------------------------
# Indicators
# ------------------------
def ema(series, period):
    s = ensure_numeric_series(series, index=series.index if isinstance(series, pd.Series) else None)
    return s.ewm(span=period, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    macd_line = ema(df['Close'], fast) - ema(df['Close'], slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series, period=14, method='sma'):
    s = ensure_numeric_series(series, index=series.index if isinstance(series, pd.Series) else None)
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
    close = ensure_numeric_series(df['Close'], index=df.index)
    high = ensure_numeric_series(df['High'], index=df.index)
    low = ensure_numeric_series(df['Low'], index=df.index)
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_v = (tp - sma_tp) / (0.015 * mad)
    return cci_v

# ------------------------
# Warmup helpers
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
    if isinstance(val, (pd.Series, np.ndarray)):
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

def run_backtest(df, indicator_series, low_thresh, high_thresh,
                 entry_exec='close', exit_exec='close',
                 sizing_mode='fixed', fixed_amount=1000, initial_capital=10000,
                 allow_multiple_entries=False, commission_mode='none',
                 commission_value=0.0, exclude_incomplete=True,
                 defer_until_next_cross_with_price_higher=False,
                 debug_mode=False):
    """
    Main backtest loop with enhanced debugging.
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

    for i in range(len(df)):
        date = df.index[i]
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
                debug_log.append(f"BAR {i} {date} - indicator=NaN - positions={len(positions)}")
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
            if entry_exec == 'close':
                entry_price_raw = df['Close'].iloc[i]
            else:  # next_open
                entry_price_raw = df['next_open'].iloc[i] if 'next_open' in df.columns else (df['Open'].iloc[i + 1] if (i + 1) < len(df) else None)

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
                                debug_log.append(f"BAR {i} {date} - entry skipped (pyramiding disabled)")
                        else:
                            pos = {
                                'entry_i': i,
                                'entry_date': date,
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
                                debug_log.append(f"BAR {i} {date} - ENTER fixed shares={shares} invested={invested}")
                    else:
                        if debug_mode:
                            debug_log.append(f"BAR {i} {date} - entry computed shares=0 (fixed_amount too small)")
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
                                'entry_date': date,
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
                                debug_log.append(f"BAR {i} {date} - ENTER compound shares={shares:.4f} invested={invested}")
                        else:
                            if debug_mode:
                                debug_log.append(f"BAR {i} {date} - compound entry skipped (already in position or equity<=0)")
            else:
                if debug_mode:
                    debug_log.append(f"BAR {i} {date} - entry_price unavailable (next_open beyond range?)")

        # ----- EXIT LOGIC ----- #
        for pos in positions[:]:
            if exit_exec == 'close':
                exit_price_raw = df['Close'].iloc[i]
            else:
                exit_price_raw = df['next_open'].iloc[i] if 'next_open' in df.columns else (df['Open'].iloc[i + 1] if (i + 1) < len(df) else None)
            try:
                exit_price = float(exit_price_raw) if exit_price_raw is not None else None
            except Exception:
                exit_price = None

            if defer_until_next_cross_with_price_higher:
                crossed = False
                if prev_ind_val_float is None:
                    try:
                        crossed = (ind_val_float > high_thresh_f)
                    except Exception:
                        crossed = False
                else:
                    try:
                        crossed = (prev_ind_val_float <= high_thresh_f and ind_val_float > high_thresh_f)
                    except Exception:
                        crossed = False

                immediate_by_price = False
                if exit_price is not None:
                    try:
                        entry_price_val = float(pos.get('entry_price', 0.0))
                    except Exception:
                        entry_price_val = 0.0
                    try:
                        immediate_by_price = (ind_val_float > high_thresh_f and exit_price > entry_price_val)
                    except Exception:
                        immediate_by_price = False

                if not (crossed or immediate_by_price):
                    if debug_mode:
                        debug_log.append(f"BAR {i} {date} - deferred not closed (crossed={crossed} immediate_by_price={immediate_by_price})")
                    continue

                if exit_price is None:
                    if debug_mode:
                        debug_log.append(f"BAR {i} {date} - exit_price None, cannot close")
                    continue

                try:
                    entry_price_val = float(pos.get('entry_price', 0.0))
                except Exception:
                    entry_price_val = 0.0

                if exit_price <= entry_price_val and not immediate_by_price:
                    if debug_mode:
                        debug_log.append(f"BAR {i} {date} - exit_price <= entry_price ({exit_price} <= {entry_price_val}) - keep deferred")
                    continue

            else:
                try:
                    if not (ind_val_float > high_thresh_f):
                        continue
                except Exception:
                    continue
                if exit_price is None:
                    continue

            shares = pos['shares']
            gross = shares * exit_price
            if commission_mode == 'percent':
                commission_exit = gross * commission_value
            elif commission_mode == 'absolute':
                commission_exit = commission_value
            else:
                commission_exit = 0.0

            total_commissions = pos.get('commission_entry', 0.0) + commission_exit
            pnl_amount = (shares * exit_price) - (pos['invested']) - total_commissions

            try:
                invested_val = _to_scalar_safe(pos.get('invested', 0.0))
            except Exception:
                try:
                    invested_val = float(pos.get('invested', 0.0))
                except Exception:
                    invested_val = 0.0

            pnl_percent = (pnl_amount / invested_val) * 100 if invested_val != 0 else 0.0

            trade = {
                'entry_date': pos['entry_date'],
                'entry_indicator': pos['entry_indicator'],
                'entry_price': pos['entry_price'],
                'exit_date': date,
                'exit_indicator': ind_val_float,
                'exit_price': exit_price,
                'shares': shares,
                'invested': pos['invested'],
                'commission_total': total_commissions,
                'pnl_amount': pnl_amount,
                'pnl_percent': pnl_percent
            }
            trades.append(trade)

            cumulative_realized_pnl += pnl_amount
            positions.remove(pos)
            if debug_mode:
                debug_log.append(f"BAR {i} {date} - CLOSED trade entry_i={pos['entry_i']} exit_i={i} pnl={pnl_amount}")

        unrealized = 0.0
        for pos in positions:
            try:
                current_price = float(df['Close'].iloc[i])
                unrealized += (current_price - pos['entry_price']) * pos['shares']
            except Exception:
                pass
        equity_curve.iloc[i] = initial_capital + cumulative_realized_pnl + unrealized

        if debug_mode and not entered_this_bar:
            debug_log.append(f"BAR {i} {date} - indicator={ind_val_float:.4f} low={low_thresh_f} entered={entered_this_bar} positions={len(positions)}")

    trades_df = pd.DataFrame(trades)

    open_positions_list = positions
    open_positions_count = len(open_positions_list)

    if equity_curve.dropna().empty:
        raw_final_equity = initial_capital
    else:
        raw_final_equity = float(equity_curve.dropna().iloc[-1])

    if exclude_incomplete and open_positions_count > 0:
        realized_pnl = trades_df['pnl_amount'].sum() if not trades_df.empty else 0.0
        final_equity_baseline = initial_capital + realized_pnl
    else:
        final_equity_baseline = raw_final_equity

    total_return_amount_baseline = final_equity_baseline - initial_capital
    total_return_pct_baseline = (total_return_amount_baseline / initial_capital) * 100 if initial_capital != 0 else 0.0

    total_trades = len(trades_df)
    wins = 0
    if total_trades > 0:
        wins = trades_df[trades_df['pnl_amount'] > 0].shape[0]
        win_rate = wins / total_trades * 100.0
    else:
        win_rate = np.nan

    baseline_summary = {
        'total_trades': total_trades,
        'wins': int(wins),
        'win_rate_percent': float(win_rate) if not pd.isna(win_rate) else None,
        'total_pnl_amount': trades_df['pnl_amount'].sum() if not trades_df.empty else 0.0,
        'total_return_amount': total_return_amount_baseline,
        'total_return_percent': total_return_pct_baseline,
        'open_positions_excluded': int(open_positions_count) if exclude_incomplete else 0
    }

    if sizing_mode == 'fixed':
        if total_trades > 0:
            fixed_total_invested = float(fixed_amount) * total_trades
            fixed_total_pnl = float(baseline_summary['total_pnl_amount'])
            fixed_return_pct = (fixed_total_pnl / fixed_total_invested) * 100.0 if fixed_total_invested != 0 else np.nan
            avg_return_pct_per_trade = fixed_return_pct
        else:
            fixed_total_invested = 0.0
            fixed_total_pnl = 0.0
            fixed_return_pct = np.nan
            avg_return_pct_per_trade = np.nan

        baseline_summary.update({
            'fixed_total_invested': fixed_total_invested,
            'fixed_total_pnl': fixed_total_pnl,
            'fixed_return_percent_total': fixed_return_pct,
            'fixed_avg_return_percent_per_trade': avg_return_pct_per_trade
        })

    if debug_mode:
        return trades_df, baseline_summary, equity_curve, open_positions_list, debug_log
    else:
        return trades_df, baseline_summary, equity_curve, open_positions_list, None

# ------------------------
# Streamlit UI
# ------------------------
st.title("Backtester לפי אינדיקטורים — RSI / CCI / MACD (משופר)")
st.markdown("מלא את הפרטים בצד שמאל ולחץ הרץ.")

with st.sidebar.form(key='params'):
    st.header("פרמטרים בסיסיים")
    ticker = st.text_input("1) שם מניה (לדוגמה: 'AAPL')", value="AAPL")
    indicator = st.selectbox("2) בחר אינדיקטור", options=["RSI", "CCI", "MACD"])
    st.write("3) טווח תאריכים")
    start_date = st.date_input("תאריך התחלה")
    end_date = st.date_input("תאריך סוף")
    timeframe = st.selectbox("בחירת תדירות", options=["Daily", "Hourly"])
    interval = "1d" if timeframe == "Daily" else "60m"

    st.markdown("5) מספר מדד נמוך לכניסה ו-6) מספר מדד גבוה ליציאה")
    low_thresh = st.number_input("מדד נמוך (כניסה)", value=40.0)
    high_thresh = st.number_input("מדד גבוה (יציאה)", value=60.0)

    st.markdown("6) תקופת אינדיקטור (לדוגמה RSI/CCI)")
    indicator_period = st.number_input("תקופת אינדיקטור", min_value=1, value=14)

    rsi_method = st.selectbox("RSI warmup method (אם בחרת RSI)", options=["sma", "wilder"], index=1,
                               format_func=lambda x: "SMA warmup (early values)" if x=="sma" else "Wilder smoothing")

    st.markdown("בחירת ביצוע (execution) למחיר")
    entry_exec = st.selectbox("מחיר כניסה", options=["close", "next_open"])
    exit_exec = st.selectbox("מחיר יציאה", options=["close", "next_open"])

    st.markdown("גודל פוזיציה / מודל sizing")
    sizing_mode = st.selectbox("שיטת חישוב תשואה", options=["fixed", "compound"], index=1,
                               format_func=lambda x: "fixed amount per entry" if x=="fixed" else "compound/all-in")
    fixed_amount = st.number_input("Fixed amount per entry (אם בחרת fixed)", value=1000.0, step=100.0)
    initial_capital = st.number_input("Initial capital (להשוואת compound / buy&hold)", value=10000.0)

    allow_multiple_entries = st.checkbox("אפשר כניסות כפולות (pyramiding) אם בחרת fixed", value=False)

    st.markdown("עמלות")
    commission_mode = st.selectbox("סוג עמלה", options=["none", "percent", "absolute"])
    commission_value = st.number_input("ערך עמלה (אחוז או סכום)", value=0.0,
                                       help="אם באחוז הזן כמו 0.001 עבור 0.1% , אם סכום הזן בערך בש\"ח/דולר")

    st.markdown("אופציות נוספות")
    exclude_incomplete = st.checkbox("לא לכלול פוזיציה פתוחה אחרונה (בחישוב הסיכום)", value=False)
    compare_buy_hold = st.checkbox("השווה ל-BUY & HOLD", value=True)

    defer_until_next_cross_with_price_higher = st.checkbox(
        "אם בעת יציאה המחיר נמוך מהכניסה — דחה סגירה; סגור רק בפעם הבאה שהאינדיקטור יחצה את הסף ותמח גבוה מהכניסה",
        value=True,
    )

    debug_mode = st.checkbox("הצג דיבאג של ערכי אינדיקטור (ראשונים)", value=False)

    use_real_indicators = st.checkbox("השתמש בערכי אינדיקטור אמיתיים מה-Yahoo / העלה CSV (Use real indicator values)", value=False,
                                      help="אם מסומן — המערכת תנסה להשתמש בעמודת אינדיקטור מתוך נתוני yfinance או בקובץ CSV שתעלה. יחד עם זאת גם שערי המניה יהיו מהנתונים שנמשכו.")
    uploaded_indicator_file = None
    if use_real_indicators:
        uploaded_indicator_file = st.file_uploader("העלה קובץ CSV עם עמודת תאריך ועמודת אינדיקטור (אופציונלי)", type=['csv'], help="עמודה אחת צריכה להיות תאריך, השנייה ערך אינדיקטור. אם לא תעלה — המערכת תחפש בעמודות שנמשכו משירות.")

    check_exits_after = st.checkbox("בדוק סגירות אחרי תאריך הסיום ועד היום", value=True,
                                   help="אם יש פוזיציה פתוחה בסוף התקופה — חפש אם התנאי ליציאה התקיים לאחר תום התקופה ועד היום.")
    include_after_in_summary = st.checkbox("כלול סגירות לאחר התקופה בחישובי הרווחים (אם נמצאו)", value=True)

    submitted = st.form_submit_button("הרץ backtest")

# helper: fetch data
def fetch_data(ticker, start_date, end_date, interval):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()

# ------------------------
# Main execution
# ------------------------
if submitted:
    warmup_bars = compute_warmup_bars(indicator, indicator_period, rsi_method)
    warmup_days = compute_warmup_days(warmup_bars, interval)
    st.info(f"Warmup: מביא כ-{warmup_bars} ברס (בערך {warmup_days} ימים) לפני תאריך ההתחלה כדי 'לחמם' את האינדיקטור — warmup מוגדל לשיפור דיוק.")

    with st.spinner("מושך נתונים מורחבים ומחשב/משתמש באינדיקטורים..."):
        fetch_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        if check_exits_after:
            fetch_end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
        df_full = fetch_data(ticker, extended_start, fetch_end, interval)

        if not isinstance(df_full, pd.DataFrame) or df_full.empty:
            st.error("לא הצלחנו לקבל טווח נתונים תקין מ-Yahoo. נסה לשנות טווח/תדירות או כבה את 'נתוני אמת'.")
            st.stop()

        try:
            df_full.index = pd.to_datetime(df_full.index)
        except Exception:
            pass

        # normalize and coerce numeric
        try:
            df_full = normalize_df_columns(df_full)
        except Exception:
            for c in df_full.columns:
                try:
                    df_full[c] = ensure_numeric_series(df_full[c], index=df_full.index)
                except Exception:
                    pass

        # compute computed indicators on full df
        if indicator == "RSI":
            ind_computed = rsi(df_full['Close'], period=int(indicator_period), method=rsi_method) if 'Close' in df_full.columns else pd.Series([np.nan]*len(df_full), index=df_full.index)
        elif indicator == "CCI":
            ind_computed = cci(df_full, period=int(indicator_period)) if all(x in df_full.columns for x in ['Close','High','Low']) else pd.Series([np.nan]*len(df_full), index=df_full.index)
        else:
            if 'Close' in df_full.columns:
                macd_line, signal_line, hist = macd(df_full, fast=12, slow=26, signal=9)
                ind_computed = macd_line
            else:
                ind_computed = pd.Series([np.nan]*len(df_full), index=df_full.index)

        # prepare indicator series (uploaded CSV or df column


