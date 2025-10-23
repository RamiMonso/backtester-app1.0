# backtester_streamlit_improved.py
# Improved Backtester for Streamlit — expanded warmup + defaults + summary additions
# להרצה: pip install streamlit yfinance pandas numpy matplotlib
# ואז: streamlit run backtester_streamlit_improved.py



import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import math

st.set_page_config(page_title="Indicator Backtester — Improved (expanded warmup)", layout="wide")

# ------------------------
# Indicator implementations
# ------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    macd_line = ema(df['Close'], fast) - ema(df['Close'], slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series, period=14, method='sma'):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    if method == 'wilder':
        # Wilder with min_periods=1 to produce early values but we rely on warmup for stability
        avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    else:
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi

def cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

# ------------------------
# Warmup helpers (expanded/stricter)
# ------------------------
def compute_warmup_bars(indicator_name, indicator_period, rsi_method):
    """
    Return a conservative number of bars to fetch before start_date to 'warm up' indicators.
    Increased relative to earlier versions to improve indicator stability at the first usable bar.
    """
    p = int(max(1, int(indicator_period)))
    # Make warmup significantly larger to avoid early transient/edge effects
    if indicator_name == "RSI":
        # Wilder smoothing needs more bars to stabilize; SMA also benefits but less so
        if rsi_method == 'wilder':
            return max(400, p * 10)        # very conservative for Wilder
        else:
            return max(120, p * 6)         # conservative for SMA warmup
    elif indicator_name == "CCI":
        return max(120, p * 6)             # CCI uses rolling and MAD -> needs many bars
    elif indicator_name == "MACD":
        slow = 26
        return max(200, slow * 8)          # MACD uses EMAs -> give ample history
    else:
        return max(120, p * 6)

def compute_warmup_days(warmup_bars, interval):
    """
    Convert warmup bars to calendar days to fetch from yfinance.
    Use a conservative multiplier to ensure enough history is retrieved.
    """
    if interval == "1d":
        # assume 1 bar per trading day, but fetch extra calendar days to account for weekends/holidays
        days = int(math.ceil(warmup_bars * 1.8)) + 10
    else:
        # for intraday, approximate trading bars per day ~6.5 hours -> 6.5 bars for 60m
        trading_bars_per_day = 6.5
        days = int(math.ceil((warmup_bars / trading_bars_per_day) * 2.0)) + 5
        days = max(days, 30)  # ensure a reasonable minimum for intraday
    return days

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
                entry_price_raw = df['Open'].iloc[i + 1] if (i + 1) < len(df) else None

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
                            # cannot enter due to pyramiding restriction
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
                        # compute current equity
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
            # default: determine exit_price according to execution
            if exit_exec == 'close':
                exit_price_raw = df['Close'].iloc[i]
            else:
                exit_price_raw = df['Open'].iloc[i + 1] if (i + 1) < len(df) else None
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
                # else proceed to close

            else:
                try:
                    if not (ind_val_float > high_thresh_f):
                        continue
                except Exception:
                    continue
                if exit_price is None:
                    continue
                # Note: in-period behavior historically allowed exit even if exit_price <= entry_price.
                # We keep that behavior inside the in-period loop to match historical runs.
                # Post-period logic (in the UI flow) will use stricter BOTH-conditions if selected.

            # compute PnL
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

            # update realized pnl and remove position
            cumulative_realized_pnl += pnl_amount
            positions.remove(pos)
            if debug_mode:
                debug_log.append(f"BAR {i} {date} - CLOSED trade entry_i={pos['entry_i']} exit_i={i} pnl={pnl_amount}")

        # compute equity at this bar (realized + unrealized)
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

    # Finished loop
    trades_df = pd.DataFrame(trades)

    # ---- Build baseline summary (apply exclude_incomplete as baseline) ----
    open_positions_list = positions  # remaining open positions
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

    # compute baseline win rate
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

    # If fixed sizing, compute fixed-mode returns as user requested:
    if sizing_mode == 'fixed':
        if total_trades > 0:
            fixed_total_invested = float(fixed_amount) * total_trades
            fixed_total_pnl = float(baseline_summary['total_pnl_amount'])
            fixed_return_pct = (fixed_total_pnl / fixed_total_invested) * 100.0 if fixed_total_invested != 0 else np.nan
            avg_return_pct_per_trade = fixed_return_pct  # same interpretation: total over total invested
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
st.title("Backtester לפי אינדיקטורים — RSI / CCI / MACD (משופר, warmup מוגדל)")
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
    # default low=40, high=60 as requested
    low_thresh = st.number_input("מדד נמוך (כניסה)", value=40.0)
    high_thresh = st.number_input("מדד גבוה (יציאה)", value=60.0)

    st.markdown("6) תקופת אינדיקטור (לדוגמה RSI/CCI)")
    indicator_period = st.number_input("תקופת אינדיקטור", min_value=1, value=14)

    # default warmup method = 'wilder'
    rsi_method = st.selectbox("RSI warmup method (אם בחרת RSI)", options=["sma", "wilder"], index=1,
                               format_func=lambda x: "SMA warmup (early values)" if x=="sma" else "Wilder smoothing")

    st.markdown("בחירת ביצוע (execution) למחיר")
    entry_exec = st.selectbox("מחיר כניסה", options=["close", "next_open"])
    exit_exec = st.selectbox("מחיר יציאה", options=["close", "next_open"])

    st.markdown("גודל פוזיציה / מודל sizing")
    # default sizing_mode = 'compound'
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
    # exclude_incomplete default = not checked (False)
    exclude_incomplete = st.checkbox("לא לכלול פוזיציה פתוחה אחרונה (בחישוב הסיכום)", value=False)
    compare_buy_hold = st.checkbox("השווה ל-BUY & HOLD", value=True)

    # defer_until_next_cross_with_price_higher default True
    defer_until_next_cross_with_price_higher = st.checkbox(
        "אם בעת יציאה המחיר נמוך מהכניסה — דחה סגירה; סגור רק בפעם הבאה שהאינדיקטור יחצה את הסף ותמח גבוה מהכניסה",
        value=True,
    )

    debug_mode = st.checkbox("הצג דיבאג של ערכי אינדיקטור (ראשונים)", value=False)

    # NEW: options for post-period exit search (keep default True as before)
    check_exits_after = st.checkbox("בדוק סגירות אחרי תאריך הסיום ועד היום", value=True,
                                   help="אם יש פוזיציה פתוחה בסוף התקופה — חפש אם התנאי ליציאה התקיים לאחר תום התקופה ועד היום.")
    include_after_in_summary = st.checkbox("כלול סגירות לאחר התקופה בחישובי הרווחים (אם נמצאו)", value=True,
                                          help="אם מצאנו סגירות אחרי תום התקופה — האם לכלול אותן בחישוב התשואה/מדדים?")

    submitted = st.form_submit_button("הרץ backtest")

def fetch_data(ticker, start_date, end_date, interval):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if df.empty:
        st.error("לא הצלחנו לקבל נתונים — בדוק סימול המניה/טווח וזמינות תדירות.")
    return df

if submitted:
    # compute expanded warmup
    warmup_bars = compute_warmup_bars(indicator, indicator_period, rsi_method)
    warmup_days = compute_warmup_days(warmup_bars, interval)
    st.info(f"Warmup: מביא כ-{warmup_bars} ברס (בערך {warmup_days} ימים) לפני תאריך ההתחלה כדי 'לחמם' את האינדיקטור — warmup מוגדל לשיפור דיוק.")

    with st.spinner("מושך נתונים מורחבים ומחשב אינדיקטורים..."):
        # decide fetch end: if user wants post-period checks, fetch until today; else until end_date+1
        fetch_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        if check_exits_after:
            fetch_end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)  # include today

        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
        df_full = fetch_data(ticker, extended_start, fetch_end, interval)
        if df_full is None or df_full.empty:
            st.stop()

        try:
            df_full.index = pd.to_datetime(df_full.index)
        except Exception:
            pass

        # compute indicator on full data (including post-period if any)
        if indicator == "RSI":
            ind_full = rsi(df_full['Close'], period=int(indicator_period), method=rsi_method)
        elif indicator == "CCI":
            ind_full = cci(df_full, period=int(indicator_period))
        else:
            macd_line, signal_line, hist = macd(df_full, fast=12, slow=26, signal=9)
            ind_full = macd_line

        # robust align
        try:
            if isinstance(ind_full, pd.Series):
                ind_full_series = ind_full.reindex(df_full.index)
            elif isinstance(ind_full, pd.DataFrame):
                ind_full_series = ind_full.iloc[:, 0].reindex(df_full.index)
            else:
                arr = np.asarray(ind_full)
                if arr.ndim == 0:
                    ind_full_series = pd.Series([np.nan] * len(df_full), index=df_full.index)
                else:
                    if arr.shape[0] == len(df_full):
                        ind_full_series = pd.Series(arr, index=df_full.index)
                    else:
                        try:
                            tmp = pd.Series(ind_full)
                            if tmp.shape[0] == len(df_full):
                                ind_full_series = pd.Series(tmp.values, index=df_full.index)
                            else:
                                ind_full_series = pd.Series([np.nan] * len(df_full), index=df_full.index)
                        except Exception:
                            ind_full_series = pd.Series([np.nan] * len(df_full), index=df_full.index)
        except Exception:
            ind_full_series = pd.Series([np.nan] * len(df_full), index=df_full.index)

        ind_full_series = pd.to_numeric(ind_full_series, errors='coerce')

        # trim to user-specified window for running the actual backtest (entries limited to that window)
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df = df_full.loc[start_ts:end_ts].copy()
        ind_series = ind_full_series.loc[df.index].copy()

        if df.empty:
            st.error("לא נותרו ברים בטווח שנבחר אחרי חיתוך ה-warmup — בדוק את תאריכי התחלה/סיום.")
            st.stop()

        if debug_mode:
            st.subheader("Debug - indicator head and first threshold hits")
            ind_s = ind_series
            st.write(f"indicator object type: {type(ind_s)} | length (aligned to df): {len(ind_s)}")
            try:
                types_sample = ind_s.head(10).apply(lambda x: type(x).__name__).to_list()
                st.write("Sample types for first 10 indicator values:", types_sample)
            except Exception:
                pass
            ind_df = ind_s.rename('indicator').reset_index()
            if len(ind_df.columns) >= 2:
                ind_df.columns = ['date', 'indicator']
            st.write("First 60 indicator values (aligned to price index):")
            st.dataframe(ind_df.head(60))
            st.write("First 60 Close prices:")
            st.dataframe(df['Close'].reset_index().head(60))
            below = ind_df[ind_df['indicator'] < low_thresh]
            above = ind_df[ind_df['indicator'] > high_thresh]
            if not below.empty:
                st.write("First below low_thresh:", below.iloc[0].to_dict())
            else:
                st.write("No value below low_thresh in sample")
            if not above.empty:
                st.write("First above high_thresh:", above.iloc[0].to_dict())
            else:
                st.write("No value above high_thresh in sample")

        # run backtest over requested period (entries are allowed only within start..end)
        trades_df, baseline_summary, equity_curve, open_positions, debug_log = run_backtest(
            df, ind_series, low_thresh, high_thresh,
            entry_exec=entry_exec, exit_exec=exit_exec,
            sizing_mode=sizing_mode, fixed_amount=fixed_amount,
            initial_capital=initial_capital,
            allow_multiple_entries=allow_multiple_entries,
            commission_mode=commission_mode, commission_value=commission_value,
            exclude_incomplete=exclude_incomplete,
            defer_until_next_cross_with_price_higher=defer_until_next_cross_with_price_higher,
            debug_mode=debug_mode
        )

        # ---------------------------------------------------------------------
        # POST-PERIOD EXIT SEARCH (strict/BOTH behavior retained from prior version)
        # ---------------------------------------------------------------------
        after_trades = []
        remaining_open_positions = list(open_positions)  # copy - we'll remove those closed by post-search

        if check_exits_after and remaining_open_positions:
            last_backtest_idx = df.index[-1]
            future_mask = ind_full_series.index > last_backtest_idx
            future_inds = ind_full_series.loc[future_mask]

            try:
                high_thresh_f = float(high_thresh)
            except Exception:
                high_thresh_f = _to_scalar_safe(high_thresh)

            # iterate each open position and then iterate chronologically over future bars;
            # close at the first future bar where BOTH conditions are met:
            #   - ind_at_bar > high_thresh
            #   - exit_price (according to execution) > entry_price
            for pos in list(open_positions):
                entry_price_val = None
                try:
                    entry_price_val = float(pos.get('entry_price', 0.0))
                except Exception:
                    entry_price_val = 0.0

                found = False
                for j, idx in enumerate(future_inds.index):
                    try:
                        ind_at = future_inds.iloc[j]
                    except Exception:
                        ind_at = np.nan

                    # get exit_price according to execution
                    exit_price = None
                    try:
                        if exit_exec == 'close':
                            exit_price = float(df_full['Close'].loc[idx])
                        else:  # next_open
                            pos_int = df_full.index.get_indexer([idx])[0]
                            if (pos_int + 1) < len(df_full):
                                exit_price = float(df_full['Open'].iloc[pos_int + 1])
                            else:
                                exit_price = None
                    except Exception:
                        exit_price = None

                    # check both conditions: indicator > threshold AND exit_price > entry_price
                    cond_ind = False
                    try:
                        cond_ind = (float(ind_at) > high_thresh_f)
                    except Exception:
                        cond_ind = False

                    cond_price = False
                    try:
                        if exit_price is not None and entry_price_val is not None:
                            cond_price = (float(exit_price) > float(entry_price_val))
                        else:
                            cond_price = False
                    except Exception:
                        cond_price = False

                    if cond_ind and cond_price:
                        # close here
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

                        after_trade = {
                            'entry_date': pos['entry_date'],
                            'entry_indicator': pos.get('entry_indicator', np.nan),
                            'entry_price': pos.get('entry_price', np.nan),
                            'exit_date': idx,
                            'exit_indicator': float(ind_full_series.loc[idx]) if pd.notna(ind_full_series.loc[idx]) else np.nan,
                            'exit_price': exit_price,
                            'shares': shares,
                            'invested': pos.get('invested', np.nan),
                            'commission_total': total_commissions,
                            'pnl_amount': pnl_amount,
                            'pnl_percent': pnl_percent,
                            'closed_after_period': True
                        }
                        after_trades.append(after_trade)
                        # mark removed
                        try:
                            remaining_open_positions.remove(pos)
                        except Exception:
                            pass
                        found = True
                        if debug_mode:
                            debug_log.append(f"POST-CLOSE pos entry {pos.get('entry_date')} closed at {idx} exit_price={exit_price} ind={ind_at}")
                        break
                    # else continue to next future bar

                if not found and debug_mode:
                    debug_log.append(f"POST-NOTFOUND pos entry {pos.get('entry_date')} - no future bar met BOTH conditions")

        # ---------------------------------------------------------------------
        # Build combined rows & summary adjustment (include after_trades optionally)
        # ---------------------------------------------------------------------
        closed_rows = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                # compute days between entry and exit
                try:
                    entry_dt = pd.to_datetime(row['entry_date'])
                    exit_dt = pd.to_datetime(row['exit_date'])
                    days = (exit_dt - entry_dt).days
                except Exception:
                    days = None
                closed_rows.append({
                    'entry_date': pd.to_datetime(row['entry_date']).strftime('%Y-%m-%d %H:%M'),
                    'entry_indicator': row.get('entry_indicator', np.nan),
                    'entry_price': row.get('entry_price', np.nan),
                    'exit_date': pd.to_datetime(row['exit_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('exit_date')) else '',
                    'exit_indicator': row.get('exit_indicator', np.nan),
                    'exit_price': row.get('exit_price', np.nan),
                    'pnl_percent': row.get('pnl_percent', np.nan),
                    'pnl_amount': row.get('pnl_amount', np.nan),
                    'days': days
                })

        after_rows = []
        for at in after_trades:
            try:
                exit_dt = pd.to_datetime(at['exit_date'])
                entry_dt = pd.to_datetime(at['entry_date'])
                exit_dt_s = exit_dt.strftime('%Y-%m-%d %H:%M')
                days = (exit_dt - entry_dt).days
            except Exception:
                exit_dt_s = str(at['exit_date'])
                days = None
            after_rows.append({
                'entry_date': pd.to_datetime(at['entry_date']).strftime('%Y-%m-%d %H:%M') if not pd.isna(at['entry_date']) else at['entry_date'],
                'entry_indicator': at.get('entry_indicator', np.nan),
                'entry_price': at.get('entry_price', np.nan),
                'exit_date': exit_dt_s,
                'exit_indicator': at.get('exit_indicator', np.nan),
                'exit_price': at.get('exit_price', np.nan),
                'pnl_percent': at.get('pnl_percent', np.nan),
                'pnl_amount': at.get('pnl_amount', np.nan),
                'days': days
            })

        open_rows = []
        # use the last in-period date as reference for open durations
        last_in_period = df.index[-1]
        for pos in remaining_open_positions:
            try:
                entry_dt = pd.to_datetime(pos.get('entry_date'))
                entry_dt_s = entry_dt.strftime('%Y-%m-%d %H:%M')
                days_open = (last_in_period - entry_dt).days
            except Exception:
                entry_dt_s = str(pos.get('entry_date'))
                days_open = None
            open_rows.append({
                'entry_date': entry_dt_s,
                'entry_indicator': pos.get('entry_indicator', np.nan),
                'entry_price': pos.get('entry_price', np.nan),
                'exit_date': "עדיין לא התקיימו התנאים ליציאה",
                'exit_indicator': np.nan,
                'exit_price': np.nan,
                'pnl_percent': np.nan,
                'pnl_amount': np.nan,
                'days': days_open
            })

        combined_rows = closed_rows + after_rows + open_rows

        # SUMMARY ADJUSTMENT
        summary = dict(baseline_summary)
        baseline_realized = float(summary.get('total_pnl_amount', 0.0))
        after_realized = sum([float(x.get('pnl_amount', 0.0)) for x in after_trades]) if after_trades else 0.0

        # compute wins including after_trades optionally
        baseline_wins = int(summary.get('wins', 0))
        baseline_trades = int(summary.get('total_trades', 0))
        after_wins = 0
        if after_trades:
            after_wins = sum(1 for x in after_trades if float(x.get('pnl_amount', 0.0)) > 0)

        if include_after_in_summary and after_trades:
            total_realized = baseline_realized + after_realized

            unrealized_remaining = 0.0
            for pos in remaining_open_positions:
                try:
                    last_price = float(df_full['Close'].iloc[-1])
                    unrealized_remaining += (last_price - pos.get('entry_price', 0.0)) * pos.get('shares', 0.0)
                except Exception:
                    pass

            if exclude_incomplete:
                final_equity_adj = initial_capital + total_realized
            else:
                final_equity_adj = initial_capital + total_realized + unrealized_remaining

            total_return_amount_adj = final_equity_adj - initial_capital
            total_return_pct_adj = (total_return_amount_adj / initial_capital) * 100 if initial_capital != 0 else 0.0

            eq_for_metrics = equity_curve.copy()
            try:
                last_idx = eq_for_metrics.dropna().index[-1]
                eq_for_metrics.loc[last_idx] = final_equity_adj
            except Exception:
                eq_for_metrics = pd.Series([final_equity_adj], index=[df.index[-1]])

            try:
                days = (df.index[-1] - df.index[0]).days
                years = days / 365.25 if days > 0 else 0
                if years > 0:
                    cagr = ((final_equity_adj / initial_capital) ** (1 / years) - 1) * 100
                else:
                    cagr = np.nan
            except Exception:
                cagr = np.nan

            try:
                eq = eq_for_metrics.dropna()
                returns = eq.pct_change().dropna()
                if returns.shape[0] > 1 and returns.std() > 0:
                    deltas = np.diff(df.index.astype('int64')) / 1e9
                    median_delta = np.median(deltas) if len(deltas) > 0 else 24*3600
                    if median_delta >= 23 * 3600:
                        ppy = 252
                    elif median_delta >= 3600:
                        ppy = int(252 * 6.5)
                    else:
                        ppy = 252
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(ppy)
                else:
                    sharpe = np.nan
            except Exception:
                sharpe = np.nan

            try:
                eq = eq_for_metrics.dropna()
                running_max = eq.cummax()
                drawdown = (eq / running_max - 1)
                max_drawdown = drawdown.min()
            except Exception:
                max_drawdown = np.nan

            # update win rate including after trades
            total_trades_incl_after = baseline_trades + len(after_trades)
            total_wins_incl_after = baseline_wins + after_wins
            win_rate_incl_after = (total_wins_incl_after / total_trades_incl_after * 100.0) if total_trades_incl_after > 0 else np.nan

            summary.update({
                'total_trades': int(summary.get('total_trades', 0)) + len(after_trades),
                'wins': int(total_wins_incl_after),
                'win_rate_percent': float(win_rate_incl_after) if not pd.isna(win_rate_incl_after) else None,
                'total_pnl_amount': float(baseline_realized) + after_realized,
                'total_return_amount': total_return_amount_adj,
                'total_return_percent': total_return_pct_adj,
                'cagr_percent': cagr,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'open_positions_excluded': int(len(remaining_open_positions)) if exclude_incomplete else 0,
                'after_period_closures_included': True,
                'after_period_closures_count': len(after_trades)
            })

            # if sizing_mode == fixed, update fixed totals
            if sizing_mode == 'fixed':
                fixed_total_invested = float(fixed_amount) * (baseline_trades + len(after_trades))
                fixed_total_pnl = float(summary.get('total_pnl_amount', 0.0))
                fixed_return_pct = (fixed_total_pnl / fixed_total_invested) * 100.0 if fixed_total_invested != 0 else np.nan
                summary.update({
                    'fixed_total_invested': fixed_total_invested,
                    'fixed_total_pnl': fixed_total_pnl,
                    'fixed_return_percent_total': fixed_return_pct,
                    'fixed_avg_return_percent_per_trade': fixed_return_pct
                })

        else:
            # no after trades included
            summary.update({
                'after_period_closures_found': len(after_trades),
                'after_period_closures_included': False,
                'open_positions_excluded': int(len(remaining_open_positions)) if exclude_incomplete else 0
            })
            # ensure cagr/sharpe/drawdown are computed based on equity_curve as before
            try:
                final_equity_baseline = (equity_curve.dropna().iloc[-1] if not equity_curve.dropna().empty else initial_capital)
                days = (df.index[-1] - df.index[0]).days
                years = days / 365.25 if days > 0 else 0
                cagr = ((final_equity_baseline / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan
            except Exception:
                cagr = np.nan
            try:
                eq = equity_curve.dropna()
                returns = eq.pct_change().dropna()
                if returns.shape[0] > 1 and returns.std() > 0:
                    deltas = np.diff(df.index.astype('int64')) / 1e9
                    median_delta = np.median(deltas) if len(deltas) > 0 else 24*3600
                    if median_delta >= 23 * 3600:
                        ppy = 252
                    elif median_delta >= 3600:
                        ppy = int(252 * 6.5)
                    else:
                        ppy = 252
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(ppy)
                else:
                    sharpe = np.nan
            except Exception:
                sharpe = np.nan
            try:
                eq = equity_curve.dropna()
                running_max = eq.cummax()
                drawdown = (eq / running_max - 1)
                max_drawdown = drawdown.min()
            except Exception:
                max_drawdown = np.nan
            summary.update({
                'cagr_percent': cagr,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown
            })

        # ensure win_rate is present if no after-trades included
        if not summary.get('win_rate_percent') and not pd.isna(win_rate):
            summary['win_rate_percent'] = float(win_rate)

    # ---- UI output: summary & trades ----
    st.subheader("תוצאות Backtest")
    st.write("סיכום כללי:")
    st.json(summary)

    if debug_mode and debug_log is not None:
        st.subheader("Debug log — מה קרה בשורות הראשונות")
        for line in debug_log[:500]:
            st.text(line)

    # ---- Trades table (closed during period, closed after period, open) ----
    st.subheader("טבלת עסקאות (נסגרו / נסגרו אחרי התאריך / פתוחות)")
    if len(combined_rows) == 0:
        st.warning("לא נרשמו עסקאות לפי הפרמטרים שנבחרו.")
    else:
        combined_df_display = pd.DataFrame(combined_rows)
        # ensure ordering of columns and include 'days'
        combined_df_display = combined_df_display[[
            'entry_date','entry_indicator','entry_price','exit_date','exit_indicator','exit_price','pnl_percent','pnl_amount','days'
        ]]
        combined_df_display = combined_df_display.rename(columns={
            'entry_date': 'תאריך כניסה',
            'entry_indicator': 'ערך אינדיקטור כניסה',
            'entry_price': 'מחיר כניסה',
            'exit_date': 'תאריך יציאה / מצב',
            'exit_indicator': 'ערך אינדיקטור יציאה',
            'exit_price': 'מחיר יציאה',
            'pnl_percent': 'אחוז רווח/הפסד',
            'pnl_amount': 'רווח/הפסד כספי',
            'days': 'מספר ימים'
        })
        st.dataframe(combined_df_display)

        # CSV downloads
        if not trades_df.empty:
            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("הורד CSV של עסקאות (נסגרו בתקופת הבדיקה)", data=csv, file_name=f"{ticker}_trades_closed.csv", mime="text/csv")
        if after_rows:
            after_csv = pd.DataFrame(after_rows).to_csv(index=False).encode('utf-8')
            st.download_button("הורד CSV של סגירות לאחר התקופה", data=after_csv, file_name=f"{ticker}_after_period_closures.csv", mime="text/csv")
        if open_rows:
            open_csv = pd.DataFrame(open_rows).to_csv(index=False).encode('utf-8')
            st.download_button("הורד CSV של פוזיציות פתוחות", data=open_csv, file_name=f"{ticker}_open_positions.csv", mime="text/csv")

    # Optionally show a detailed table just for open positions (raw fields)
    if remaining_open_positions:
        st.subheader("פוזיציות שעדיין פתוחות - פרטים")
        raw_open = []
        for pos in remaining_open_positions:
            raw_open.append({
                'entry_date': pos.get('entry_date'),
                'entry_indicator': pos.get('entry_indicator'),
                'entry_price': pos.get('entry_price'),
                'shares': pos.get('shares'),
                'invested': pos.get('invested'),
                'commission_entry': pos.get('commission_entry'),
                'max_price': pos.get('max_price'),
                'deferred': pos.get('deferred')
            })
        raw_open_df = pd.DataFrame(raw_open)
        try:
            raw_open_df['entry_date'] = pd.to_datetime(raw_open_df['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass
        st.dataframe(raw_open_df)

    # Plot price with markers (including after-period exits and open positions)
    st.subheader("גרף מחירים - כניסות/יציאות")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_full.index, df_full['Close'], label=f"{ticker} Close (fetched range)")
    # closed during period
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            entry_dt = pd.to_datetime(row['entry_date'])
            exit_dt = pd.to_datetime(row['exit_date'])
            ax.scatter(entry_dt, row['entry_price'], marker='^', s=80, color='green')
            ax.scatter(exit_dt, row['exit_price'], marker='v', s=80, color='red')
    # after_period closures
    for at in after_trades:
        try:
            entry_dt = pd.to_datetime(at['entry_date'])
            exit_dt = pd.to_datetime(at['exit_date'])
            ax.scatter(entry_dt, at['entry_price'], marker='^', s=80, color='green')
            ax.scatter(exit_dt, at['exit_price'], marker='x', s=100, color='blue', label='Exit after period')
        except Exception:
            pass
    # open positions
    for pos in remaining_open_positions:
        try:
            entry_dt = pd.to_datetime(pos.get('entry_date'))
            ax.scatter(entry_dt, pos.get('entry_price'), marker='^', s=120, facecolors='none', edgecolors='orange', linewidths=2, label='Open Position')
        except Exception:
            pass
    ax.set_title(f"{ticker} — Price with Entries/Exits (including post-period data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # equity curve (as originally computed for period)
    st.subheader("עקומת הון (תקופת הבדיקה)")
    if not equity_curve.dropna().empty:
        fig2, ax2 = plt.subplots(figsize=(12,4))
        ax2.plot(equity_curve.index, equity_curve.values, label='Equity Curve (during period)')
        ax2.set_title('Equity Curve (initial capital + realized + unrealized during period)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity')
        ax2.legend()
        st.pyplot(fig2)

    # PNG & PDF downloads
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.download_button("הורד PNG של הגרף", data=buf, file_name=f"{ticker}_chart.png", mime="image/png")
    buf.seek(0)

    pdf_buf = BytesIO()
    with PdfPages(pdf_buf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        if 'fig2' in locals():
            pdf.savefig(fig2, bbox_inches='tight')
        fig_text = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        text = f"Backtest Summary for {ticker}\n\n"
        for k, v in summary.items():
            text += f"{k}: {v}\n"
        plt.text(0.01, 0.99, text, va='top', wrap=True, fontsize=12)
        pdf.savefig(fig_text, bbox_inches='tight')
        plt.close(fig_text)
    pdf_buf.seek(0)
    st.download_button("הורד PDF (גרף + סיכום)", data=pdf_buf, file_name=f"{ticker}_report.pdf", mime="application/pdf")

    # Compare to Buy & Hold
    if compare_buy_hold:
        st.subheader("השוואה ל-BUY & HOLD")
        try:
            start_price = _to_scalar_safe(df['Close'].iloc[0])
            end_price = _to_scalar_safe(df['Close'].iloc[-1])
        except Exception:
            st.error("לא ניתן להמיר את מחירי הסגירה למחיר סקלרי לצורך השוואת Buy & Hold.")
            start_price = None
            end_price = None

        if start_price is None or end_price is None:
            st.info("אין נתונים להשוואת Buy & Hold.")
        else:
            bh_return = ((end_price - start_price) / start_price) * 100.0 if start_price != 0 else np.nan
            st.write(f"Buy & Hold return for period: {bh_return:.2f}%")

# EOF

