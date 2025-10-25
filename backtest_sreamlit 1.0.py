#backstarterer_streamlit_improved_with_real_indicator.py
# Improved Backtester for Streamlit — expanded warmup + option to use real indicator values/prices
# backtester_streamlit_improved_with_adj_close.py
# Improved Backtester for Streamlit — added option to use ADJ CLOSE for entry/exit
# להרצה: pip install streamlit yfinance pandas numpy matplotlib
# ואז: streamlit run backtester_streamlit_improved_with_real_indicator.py
# ואז: streamlit run backtester_streamlit_improved_with_adj_close.py

import sys
import subprocess
@@ -20,91 +20,16 @@
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO, StringIO
from io import BytesIO
import math

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
st.set_page_config(page_title="Indicator Backtester — Improved (expanded warmup) + AdjClose", layout="wide")

# ------------------------
# Indicator implementations
# ------------------------
def ema(series, period):
    s = ensure_numeric_series(series)
    return s.ewm(span=period, adjust=False).mean()
    return series.ewm(span=period, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    macd_line = ema(df['Close'], fast) - ema(df['Close'], slow)
@@ -113,33 +38,31 @@ def macd(df, fast=12, slow=26, signal=9):
    return macd_line, signal_line, hist

def rsi(series, period=14, method='sma'):
    s = ensure_numeric_series(series)
    delta = s.diff()
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
    rsi_v = 100 - (100 / (1 + rs))
    rsi_v = rsi_v.fillna(50.0)
    return rsi_v
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi

def cci(df, period=20):
    close = ensure_numeric_series(df['Close'])
    high = ensure_numeric_series(df['High'])
    low = ensure_numeric_series(df['Low'])
    tp = (high + low + close) / 3
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_v = (tp - sma_tp) / (0.015 * mad)
    return cci_v
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

# ------------------------
# Warmup helpers (same as before)
# Warmup helpers (expanded/stricter)
# ------------------------
def compute_warmup_bars(indicator_name, indicator_period, rsi_method):
    p = int(max(1, int(indicator_period)))
@@ -158,11 +81,12 @@ def compute_warmup_bars(indicator_name, indicator_period, rsi_method):

def compute_warmup_days(warmup_bars, interval):
    if interval == "1d":
        return int(math.ceil(warmup_bars * 1.8)) + 10
        days = int(math.ceil(warmup_bars * 1.8)) + 10
    else:
        trading_bars_per_day = 6.5
        days = int(math.ceil((warmup_bars / trading_bars_per_day) * 2.0)) + 5
        return max(days, 30)
        days = max(days, 30)
    return days

# ------------------------
# Utilities & Backtest core
@@ -191,6 +115,7 @@ def _to_scalar_safe(x):
        return float(flat[non_nan_idx[0]])
    return float(x)


def _get_indicator_value_at(indicator_series, idx):
    try:
        val = indicator_series.iloc[idx]
@@ -217,6 +142,7 @@ def _get_indicator_value_at(indicator_series, idx):
    except Exception:
        return np.nan


def run_backtest(df, indicator_series, low_thresh, high_thresh,
                 entry_exec='close', exit_exec='close',
                 sizing_mode='fixed', fixed_amount=1000, initial_capital=10000,
@@ -225,7 +151,7 @@ def run_backtest(df, indicator_series, low_thresh, high_thresh,
                 defer_until_next_cross_with_price_higher=False,
                 debug_mode=False):
    """
    Main backtest loop with enhanced debugging.
    Main backtest loop with support for 'adj_close' as a price execution option.
    Returns: trades_df, baseline_summary, equity_curve, open_positions_list, debug_log
    """
    if isinstance(indicator_series, pd.DataFrame):
@@ -305,10 +231,17 @@ def _estimate_periods_per_year(index):

        if ind_val_float < low_thresh_f:
            # decide entry price
            if entry_exec == 'close':
                entry_price_raw = df['Close'].iloc[i]
            else:  # next_open
                entry_price_raw = df['next_open'].iloc[i] if 'next_open' in df.columns else (df['Open'].iloc[i + 1] if (i + 1) < len(df) else None)
            entry_price_raw = None
            try:
                if entry_exec == 'close':
                    entry_price_raw = df['Close'].iloc[i]
                elif entry_exec == 'adj_close':
                    # prefer adjusted close if present, else fallback to Close
                    entry_price_raw = df['Adj Close'].iloc[i] if 'Adj Close' in df.columns else df['Close'].iloc[i]
                else:  # next_open
                    entry_price_raw = df['Open'].iloc[i + 1] if (i + 1) < len(df) else None
            except Exception:
                entry_price_raw = None

            try:
                entry_price = float(entry_price_raw) if entry_price_raw is not None else None
@@ -393,10 +326,16 @@ def _estimate_periods_per_year(index):
        # ----- EXIT LOGIC ----- #
        for pos in positions[:]:
            # default: determine exit_price according to execution
            if exit_exec == 'close':
                exit_price_raw = df['Close'].iloc[i]
            else:
                exit_price_raw = df['next_open'].iloc[i] if 'next_open' in df.columns else (df['Open'].iloc[i + 1] if (i + 1) < len(df) else None)
            exit_price_raw = None
            try:
                if exit_exec == 'close':
                    exit_price_raw = df['Close'].iloc[i]
                elif exit_exec == 'adj_close':
                    exit_price_raw = df['Adj Close'].iloc[i] if 'Adj Close' in df.columns else df['Close'].iloc[i]
                else:
                    exit_price_raw = df['Open'].iloc[i + 1] if (i + 1) < len(df) else None
            except Exception:
                exit_price_raw = None
            try:
                exit_price = float(exit_price_raw) if exit_price_raw is not None else None
            except Exception:
@@ -517,7 +456,7 @@ def _estimate_periods_per_year(index):
    trades_df = pd.DataFrame(trades)

    # ---- Build baseline summary (apply exclude_incomplete as baseline) ----
    open_positions_list = positions  # remaining open positions
    open_positions_list = positions
    open_positions_count = len(open_positions_list)

    if equity_curve.dropna().empty:
@@ -534,7 +473,6 @@ def _estimate_periods_per_year(index):
    total_return_amount_baseline = final_equity_baseline - initial_capital
    total_return_pct_baseline = (total_return_amount_baseline / initial_capital) * 100 if initial_capital != 0 else 0.0

    # compute baseline win rate
    total_trades = len(trades_df)
    wins = 0
    if total_trades > 0:
@@ -553,7 +491,6 @@ def _estimate_periods_per_year(index):
        'open_positions_excluded': int(open_positions_count) if exclude_incomplete else 0
    }

    # If fixed sizing, compute fixed-mode returns as user requested:
    if sizing_mode == 'fixed':
        if total_trades > 0:
            fixed_total_invested = float(fixed_amount) * total_trades
@@ -578,10 +515,11 @@ def _estimate_periods_per_year(index):
    else:
        return trades_df, baseline_summary, equity_curve, open_positions_list, None


# ------------------------
# Streamlit UI
# ------------------------
st.title("Backtester לפי אינדיקטורים — RSI / CCI / MACD (משופר)")
st.title("Backtester לפי אינדיקטורים — RSI / CCI / MACD (משופר, warmup מוגדל, עם AdjClose)")
st.markdown("מלא את הפרטים בצד שמאל ולחץ הרץ.")

with st.sidebar.form(key='params'):
@@ -601,13 +539,14 @@ def _estimate_periods_per_year(index):
    st.markdown("6) תקופת אינדיקטור (לדוגמה RSI/CCI)")
    indicator_period = st.number_input("תקופת אינדיקטור", min_value=1, value=14)

    # RSI warmup selection (default 'wilder')
    rsi_method = st.selectbox("RSI warmup method (אם בחרת RSI)", options=["sma", "wilder"], index=1,
                               format_func=lambda x: "SMA warmup (early values)" if x=="sma" else "Wilder smoothing")

    st.markdown("בחירת ביצוע (execution) למחיר")
    entry_exec = st.selectbox("מחיר כניסה", options=["close", "next_open"])
    exit_exec = st.selectbox("מחיר יציאה", options=["close", "next_open"])
    st.markdown("בחירת ביצוע (execution) למחיר — עכשיו תומך ב-Adj Close")
    entry_exec = st.selectbox("מחיר כניסה", options=["close", "adj_close", "next_open"], index=0,
                              format_func=lambda x: ("Close - מחיר סגירה רגיל" if x=="close" else ("Adj Close - מחיר סגירה מותאם (אם קיים)" if x=="adj_close" else "Next Open - מחיר פתיחה ביום הבא")))
    exit_exec = st.selectbox("מחיר יציאה", options=["close", "adj_close", "next_open"], index=0,
                             format_func=lambda x: ("Close - מחיר סגירה רגיל" if x=="close" else ("Adj Close - מחיר סגירה מותאם (אם קיים)" if x=="adj_close" else "Next Open - מחיר פתיחה ביום הבא")))

    st.markdown("גודל פוזיציה / מודל sizing")
    sizing_mode = st.selectbox("שיטת חישוב תשואה", options=["fixed", "compound"], index=1,
@@ -633,271 +572,103 @@ def _estimate_periods_per_year(index):

    debug_mode = st.checkbox("הצג דיבאג של ערכי אינדיקטור (ראשונים)", value=False)

    # NEW: use real indicator / upload CSV
    use_real_indicators = st.checkbox("השתמש בערכי אינדיקטור אמיתיים מה-Yahoo / העלה CSV (Use real indicator values)", value=False,
                                      help="אם מסומן — המערכת תנסה להשתמש בעמודת אינדיקטור מתוך נתוני yfinance או בקובץ CSV שתעלה. יחד עם זאת גם שערי המניה יהיו מהנתונים שנמשכו.")
    uploaded_indicator_file = None
    if use_real_indicators:
        uploaded_indicator_file = st.file_uploader("העלה קובץ CSV עם עמודת תאריך ועמודת אינדיקטור (אופציונלי)", type=['csv'], help="עמודה אחת צריכה להיות תאריך, השנייה ערך אינדיקטור. אם לא תעלה — המערכת תחפש בעמודות שנמשכו משירות.")

    # post-period options
    check_exits_after = st.checkbox("בדוק סגירות אחרי תאריך הסיום ועד היום", value=True,
                                   help="אם יש פוזיציה פתוחה בסוף התקופה — חפש אם התנאי ליציאה התקיים לאחר תום התקופה ועד היום.")
    include_after_in_summary = st.checkbox("כלול סגירות לאחר התקופה בחישובי הרווחים (אם נמצאו)", value=True)
    include_after_in_summary = st.checkbox("כלול סגירות לאחר התקופה בחישובי הרווחים (אם נמצאו)", value=True,
                                          help="אם מצאנו סגירות אחרי תום התקופה — האם לכלול אותן בחישוב התשואה/מדדים?")

    submitted = st.form_submit_button("הרץ backtest")

def fetch_data(ticker, start_date, end_date, interval):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if df.empty:
        st.error("לא הצלחנו לקבל נתונים — בדוק סימול המניה/טווח וזמינות תדירות.")
    return df

# ------------------------
# Helper: find & normalize price columns
# ------------------------
def find_column_like(df, names):
    """
    return the actual column name in df that matches any name in 'names' (case-insensitive),
    or None if not found.
    """
    low_map = {str(c).lower(): c for c in df.columns}
    for n in names:
        key = n.lower()
        if key in low_map:
            return low_map[key]
    # fallback: try partial contains
    for col in df.columns:
        lc = str(col).lower()
        for n in names:
            if n.lower() in lc:
                return col
    return None

def normalize_price_columns(df):
    """
    Ensure df has standardized columns: 'Open','High','Low','Close','Adj Close','Volume'
    Returns df with those columns (converted to numeric) if possible. Raises ValueError if Close missing.
    """
    df = df.copy()
    # ensure column names are strings
    df.columns = [str(c) for c in df.columns]

    # candidates
    open_col = find_column_like(df, ['open'])
    high_col = find_column_like(df, ['high'])
    low_col = find_column_like(df, ['low'])
    close_col = find_column_like(df, ['close'])
    adj_col = find_column_like(df, ['adj close', 'adjusted close', 'adjclose'])
    vol_col = find_column_like(df, ['volume'])

    # If close not found but adj found -> use adj as close (we'll still keep adj separately)
    if close_col is None and adj_col is not None:
        close_col = adj_col

    # If still no close -> cannot proceed
    if close_col is None:
        raise ValueError("לא נמצאה עמודת 'Close' או 'Adj Close' בנתוני yfinance — לא ניתן להמשיך.")

    # Create standardized columns by coercing numeric
    if open_col is not None:
        df['Open'] = ensure_numeric_series(df[open_col], index=df.index)
    if high_col is not None:
        df['High'] = ensure_numeric_series(df[high_col], index=df.index)
    if low_col is not None:
        df['Low'] = ensure_numeric_series(df[low_col], index=df.index)
    df['Close'] = ensure_numeric_series(df[close_col], index=df.index)
    if adj_col is not None:
        df['Adj Close'] = ensure_numeric_series(df[adj_col], index=df.index)
    if vol_col is not None:
        df['Volume'] = ensure_numeric_series(df[vol_col], index=df.index)

    return df

# ------------------------
# Main execution
# ------------------------
if submitted:
    warmup_bars = compute_warmup_bars(indicator, indicator_period, rsi_method)
    warmup_days = compute_warmup_days(warmup_bars, interval)
    st.info(f"Warmup: מביא כ-{warmup_bars} ברס (בערך {warmup_days} ימים) לפני תאריך ההתחלה כדי 'לחמם' את האינדיקטור — warmup מוגדל לשיפור דיוק.")

    with st.spinner("מושך נתונים מורחבים ומחשב/משתמש באינדיקטורים..."):
        # decide fetch end: if user wants post-period checks, fetch until today; else until end_date+1
    with st.spinner("מושך נתונים מורחבים ומחשב אינדיקטורים..."):
        fetch_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        if check_exits_after:
            fetch_end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
        df_full = fetch_data(ticker, extended_start, fetch_end, interval)

        # validate df_full
        if not isinstance(df_full, pd.DataFrame) or df_full.empty:
            st.error("לא הצלחנו לקבל נתונים תקינים מ-Yahoo לנתונים מלאים. ודא חיבור אינטרנט וסימון נכון של טיקר/טווח.")
        if df_full is None or df_full.empty:
            st.stop()

        # normalize index
        try:
            df_full.index = pd.to_datetime(df_full.index)
        except Exception:
            pass

        # Normalize price columns and coerce numeric
        try:
            df_full = normalize_price_columns(df_full)
        except Exception as e:
            st.error(f"שגיאה במיפוי עמודות המחיר: {e}")
            st.stop()

        # If user requested "use real indicators", do NOT apply AdjFactor to prices (we want raw chart values).
        # Otherwise, optionally apply adjustment factor if available
        if not use_real_indicators:
            # if Adj Close exists and Close exists, compute adj factor and adjust OHLC
            if 'Adj Close' in df_full.columns and df_full['Close'].notna().any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    adjf = (df_full['Adj Close'] / df_full['Close']).replace([np.inf, -np.inf], np.nan)
                adjf = adjf.fillna(method='ffill').fillna(method='bfill').fillna(1.0)
                df_full['AdjFactor'] = adjf
                for c in ['Close','Open','High','Low']:
                    if c in df_full.columns:
                        try:
                            df_full[c] = ensure_numeric_series(df_full[c] * df_full['AdjFactor'], index=df_full.index)
                        except Exception:
                            df_full[c] = ensure_numeric_series(df_full[c], index=df_full.index)
            else:
                df_full['AdjFactor'] = 1.0
                for c in ['Open','High','Low','Close']:
                    if c in df_full.columns:
                        df_full[c] = ensure_numeric_series(df_full[c], index=df_full.index)
        else:
            for c in ['Open','High','Low','Close']:
                if c in df_full.columns:
                    df_full[c] = ensure_numeric_series(df_full[c], index=df_full.index)
        # If user selected adj_close but the data doesn't contain it (common for intraday), fallback and warn
        has_adj = 'Adj Close' in df_full.columns
        if (entry_exec == 'adj_close' or exit_exec == 'adj_close') and not has_adj:
            st.warning("הנתונים שנמשכו לא כוללים את העמודה 'Adj Close' עבור תדירות זו — נשתמש בעמודת Close במקום (לא בוצעו התאמות). אינטראדיי בדרך כלל לא מחזיר 'Adj Close'.")
            # create fallback column so code downstream doesn't break
            df_full['Adj Close'] = df_full['Close']
            has_adj = True

        # compute indicator locally (fallback)
        # compute indicator on full data (including post-period if any)
        if indicator == "RSI":
            ind_computed = rsi(df_full['Close'], period=int(indicator_period), method=rsi_method)
            ind_full = rsi(df_full['Close'], period=int(indicator_period), method=rsi_method)
        elif indicator == "CCI":
            ind_computed = cci(df_full, period=int(indicator_period))
            ind_full = cci(df_full, period=int(indicator_period))
        else:
            macd_line, signal_line, hist = macd(df_full, fast=12, slow=26, signal=9)
            ind_computed = macd_line

        ind_full_series = None
        used_real_source = None

        # If user uploaded CSV, try to use it
        if use_real_indicators and uploaded_indicator_file is not None:
            try:
                file_content = uploaded_indicator_file.read().decode('utf-8')
                csv_df = pd.read_csv(StringIO(file_content))
                # detect date column & indicator column
                col_date = None
                col_ind = None
                for c in csv_df.columns:
                    if 'date' in str(c).lower() or 'time' in str(c).lower() or 'datetime' in str(c).lower():
                        col_date = c
                        break
                for c in csv_df.columns:
                    if c == col_date:
                        continue
                    if indicator.lower() in str(c).lower() or 'indicator' in str(c).lower() or 'value' in str(c).lower():
                        col_ind = c
                        break
                if col_ind is None:
                    for c in csv_df.columns:
                        if c != col_date:
                            try:
                                tmp = pd.to_numeric(csv_df[c], errors='coerce')
                                if tmp.notna().any():
                                    col_ind = c
                                    break
                            except Exception:
                                continue
                if col_date is None:
                    st.warning("לא זוהתה עמודת תאריך בקובץ ה-CSV. יש לוודא שיש עמודת תאריך/זמן.")
                else:
                    csv_df[col_date] = pd.to_datetime(csv_df[col_date])
                    csv_df = csv_df.set_index(col_date).sort_index()
                    if col_ind is None:
                        st.warning("לא זוהתה עמודת אינדיקטור בקובץ ה-CSV; נעשה ניסיון להשתמש בעמודה הראשונה הלא-תאריך.")
                        others = [c for c in csv_df.columns]
                        col_ind = others[0] if others else None
                    if col_ind is not None:
                        ind_full_series = ensure_numeric_series(csv_df[col_ind], index=csv_df.index)
                        # align (try exact reindex, else nearest)
                        aligned = ind_full_series.reindex(df_full.index)
                        nan_ratio = aligned.isna().mean()
                        if nan_ratio > 0.5:
                            try:
                                if interval == "1d":
                                    ind_full_series = ind_full_series.reindex(df_full.index, method='nearest', tolerance=pd.Timedelta(days=1))
                                else:
                                    ind_full_series = ind_full_series.reindex(df_full.index, method='nearest', tolerance=pd.Timedelta(hours=2))
                            except Exception:
                                ind_full_series = aligned
                        else:
                            ind_full_series = aligned
                        used_real_source = "uploaded_csv"
            except Exception as e:
                st.warning(f"שגיאה בקריאת ה-CSV: {e}. נמשיך לנסות מקורות אחרים.")

        # If not from CSV and user asked for real indicators, try to find a column inside df_full
        if ind_full_series is None and use_real_indicators:
            lower_cols = [str(c).lower() for c in df_full.columns]
            found_col = None
            for c in df_full.columns:
                lc = str(c).lower()
                if indicator.lower() == "rsi" and 'rsi' in lc:
                    found_col = c; break
                if indicator.lower() == "cci" and 'cci' in lc:
                    found_col = c; break
                if indicator.lower() == "macd" and 'macd' in lc and ('hist' not in lc):
                    found_col = c; break
            if found_col:
                try:
                    ind_full_series = ensure_numeric_series(df_full[found_col], index=df_full.index)
                    used_real_source = f"df_full_col:{found_col}"
                except Exception:
                    ind_full_series = None
            ind_full = macd_line

        # fallback to computed
        if ind_full_series is None:
            ind_full_series = ensure_numeric_series(ind_computed, index=df_full.index)
            used_real_source = "computed"

        # coerce numeric & align
        # robust align
        try:
            ind_full_series = pd.to_numeric(ind_full_series, errors='coerce')
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
            ind_full_series = ensure_numeric_series(ind_full_series, index=df_full.index)
            ind_full_series = pd.Series([np.nan] * len(df_full), index=df_full.index)

        ind_full_series = pd.to_numeric(ind_full_series, errors='coerce')

        # Trim to requested window for running the actual backtest (entries limited to that window)
        # trim to user-specified window for running the actual backtest (entries limited to that window)
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df = df_full.loc[start_ts:end_ts].copy()
        ind_series = ind_full_series.reindex(df.index).copy()
        ind_series = ind_full_series.loc[df.index].copy()

        if df.empty:
            st.error("לא נותרו ברים בטווח שנבחר אחרי חיתוך ה-warmup — בדוק את תאריכי התחלה/סיום.")
            st.stop()

        # prepare next_open convenience column
        if 'Open' in df.columns:
            df['next_open'] = df['Open'].shift(-1)
        else:
            df['next_open'] = np.nan

        if debug_mode:
            st.subheader("Debug - indicator head and first threshold hits")
            st.write(f"indicator source used: {used_real_source}")
            st.write(f"indicator object type: {type(ind_series)} | length (aligned to df): {len(ind_series)}")
            ind_s = ind_series
            st.write(f"indicator object type: {type(ind_s)} | length (aligned to df): {len(ind_s)}")
            try:
                types_sample = ind_series.head(10).apply(lambda x: type(x).__name__).to_list()
                types_sample = ind_s.head(10).apply(lambda x: type(x).__name__).to_list()
                st.write("Sample types for first 10 indicator values:", types_sample)
            except Exception:
                pass
            ind_df = ind_series.rename('indicator').reset_index()
            ind_df = ind_s.rename('indicator').reset_index()
            if len(ind_df.columns) >= 2:
                ind_df.columns = ['date', 'indicator']
            st.write("First 60 indicator values (aligned to price index):")
@@ -915,7 +686,6 @@ def normalize_price_columns(df):
            else:
                st.write("No value above high_thresh in sample")

        # run backtest over requested period (entries are allowed only within start..end)
        trades_df, baseline_summary, equity_curve, open_positions, debug_log = run_backtest(
            df, ind_series, low_thresh, high_thresh,
            entry_exec=entry_exec, exit_exec=exit_exec,
@@ -932,11 +702,12 @@ def normalize_price_columns(df):
        # POST-PERIOD EXIT SEARCH (strict/BOTH behavior retained)
        # ---------------------------------------------------------------------
        after_trades = []
        remaining_open_positions = list(open_positions)  # copy - we'll remove those closed by post-search
        remaining_open_positions = list(open_positions)

        if check_exits_after and remaining_open_positions:
            last_backtest_idx = df.index[-1]
            future_inds = ind_full_series.loc[ind_full_series.index > last_backtest_idx]
            future_mask = ind_full_series.index > last_backtest_idx
            future_inds = ind_full_series.loc[future_mask]

            try:
                high_thresh_f = float(high_thresh)
@@ -951,18 +722,20 @@ def normalize_price_columns(df):
                    entry_price_val = 0.0

                found = False
                for idx in future_inds.index:
                for j, idx in enumerate(future_inds.index):
                    try:
                        ind_at = float(future_inds.loc[idx]) if not pd.isna(future_inds.loc[idx]) else np.nan
                        ind_at = future_inds.iloc[j]
                    except Exception:
                        ind_at = np.nan

                    # get exit price according to execution
                    # get exit_price according to execution (support adj_close)
                    exit_price = None
                    try:
                        if exit_exec == 'close':
                            exit_price = float(df_full['Close'].loc[idx])
                        else:
                        elif exit_exec == 'adj_close':
                            exit_price = float(df_full['Adj Close'].loc[idx]) if 'Adj Close' in df_full.columns else float(df_full['Close'].loc[idx])
                        else:  # next_open
                            pos_int = df_full.index.get_indexer([idx])[0]
                            if (pos_int + 1) < len(df_full):
                                exit_price = float(df_full['Open'].iloc[pos_int + 1])
@@ -971,8 +744,20 @@ def normalize_price_columns(df):
                    except Exception:
                        exit_price = None

                    cond_ind = (not pd.isna(ind_at)) and (ind_at > high_thresh_f)
                    cond_price = (exit_price is not None) and (exit_price > entry_price_val)
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
                        shares = pos['shares']
@@ -990,7 +775,10 @@ def normalize_price_columns(df):
                        try:
                            invested_val = _to_scalar_safe(pos.get('invested', 0.0))
                        except Exception:
                            invested_val = float(pos.get('invested', 0.0)) if pos.get('invested', 0.0) is not None else 0.0
                            try:
                                invested_val = float(pos.get('invested', 0.0))
                            except Exception:
                                invested_val = 0.0

                        pnl_percent = (pnl_amount / invested_val) * 100 if invested_val != 0 else 0.0

@@ -999,7 +787,7 @@ def normalize_price_columns(df):
                            'entry_indicator': pos.get('entry_indicator', np.nan),
                            'entry_price': pos.get('entry_price', np.nan),
                            'exit_date': idx,
                            'exit_indicator': ind_at,
                            'exit_indicator': float(ind_full_series.loc[idx]) if pd.notna(ind_full_series.loc[idx]) else np.nan,
                            'exit_price': exit_price,
                            'shares': shares,
                            'invested': pos.get('invested', np.nan),
@@ -1022,7 +810,7 @@ def normalize_price_columns(df):
                    debug_log.append(f"POST-NOTFOUND pos entry {pos.get('entry_date')} - no future bar met BOTH conditions")

        # ---------------------------------------------------------------------
        # Build combined rows & summary adjustment
        # Build combined rows & summary adjustment (include after_trades optionally)
        # ---------------------------------------------------------------------
        closed_rows = []
        if not trades_df.empty:
@@ -1031,15 +819,13 @@ def normalize_price_columns(df):
                    entry_dt = pd.to_datetime(row['entry_date'])
                    exit_dt = pd.to_datetime(row['exit_date'])
                    days = (exit_dt - entry_dt).days
                    exit_str = pd.to_datetime(row['exit_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('exit_date')) else ''
                except Exception:
                    days = None
                    exit_str = str(row.get('exit_date',''))
                closed_rows.append({
                    'entry_date': pd.to_datetime(row['entry_date']).strftime('%Y-%m-%d %H:%M'),
                    'entry_indicator': row.get('entry_indicator', np.nan),
                    'entry_price': row.get('entry_price', np.nan),
                    'exit_date': exit_str,
                    'exit_date': pd.to_datetime(row['exit_date']).strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('exit_date')) else '',
                    'exit_indicator': row.get('exit_indicator', np.nan),
                    'exit_price': row.get('exit_price', np.nan),
                    'pnl_percent': row.get('pnl_percent', np.nan),
@@ -1055,7 +841,8 @@ def normalize_price_columns(df):
                exit_dt_s = exit_dt.strftime('%Y-%m-%d %H:%M')
                days = (exit_dt - entry_dt).days
            except Exception:
                exit_dt_s = str(at['exit_date']); days = None
                exit_dt_s = str(at['exit_date'])
                days = None
            after_rows.append({
                'entry_date': pd.to_datetime(at['entry_date']).strftime('%Y-%m-%d %H:%M') if not pd.isna(at['entry_date']) else at['entry_date'],
                'entry_indicator': at.get('entry_indicator', np.nan),
@@ -1076,7 +863,8 @@ def normalize_price_columns(df):
                entry_dt_s = entry_dt.strftime('%Y-%m-%d %H:%M')
                days_open = (last_in_period - entry_dt).days
            except Exception:
                entry_dt_s = str(pos.get('entry_date')); days_open = None
                entry_dt_s = str(pos.get('entry_date'))
                days_open = None
            open_rows.append({
                'entry_date': entry_dt_s,
                'entry_indicator': pos.get('entry_indicator', np.nan),
@@ -1091,7 +879,7 @@ def normalize_price_columns(df):

        combined_rows = closed_rows + after_rows + open_rows

        # SUMMARY ADJUSTMENT
        # SUMMARY ADJUSTMENT (unchanged logic)
        summary = dict(baseline_summary)
        baseline_realized = float(summary.get('total_pnl_amount', 0.0))
        after_realized = sum([float(x.get('pnl_amount', 0.0)) for x in after_trades]) if after_trades else 0.0
@@ -1237,12 +1025,8 @@ def normalize_price_columns(df):
                'max_drawdown': max_drawdown
            })

        # ensure win_rate present
        if summary.get('win_rate_percent') is None:
            try:
                summary['win_rate_percent'] = float(baseline_summary.get('win_rate_percent', np.nan))
            except Exception:
                summary['win_rate_percent'] = None
        if not summary.get('win_rate_percent') and not pd.isna(win_rate):
            summary['win_rate_percent'] = float(win_rate)

    # ---- UI output: summary & trades ----
    st.subheader("תוצאות Backtest")
@@ -1254,9 +1038,8 @@ def normalize_price_columns(df):
        for line in debug_log[:500]:
            st.text(line)

    # ---- Trades table (closed during period, closed after period, open) ----
    st.subheader("טבלת עסקאות (נסגרו / נסגרו אחרי התאריך / פתוחות)")
    if 'combined_rows' not in locals() or len(combined_rows) == 0:
    if len(combined_rows) == 0:
        st.warning("לא נרשמו עסקאות לפי הפרמטרים שנבחרו.")
    else:
        combined_df_display = pd.DataFrame(combined_rows)
@@ -1276,7 +1059,6 @@ def normalize_price_columns(df):
        })
        st.dataframe(combined_df_display)

        # CSV downloads
        if not trades_df.empty:
            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("הורד CSV של עסקאות (נסגרו בתקופת הבדיקה)", data=csv, file_name=f"{ticker}_trades_closed.csv", mime="text/csv")
@@ -1287,8 +1069,7 @@ def normalize_price_columns(df):
            open_csv = pd.DataFrame(open_rows).to_csv(index=False).encode('utf-8')
            st.download_button("הורד CSV של פוזיציות פתוחות", data=open_csv, file_name=f"{ticker}_open_positions.csv", mime="text/csv")

    # Optionally show a detailed table just for open positions (raw fields)
    if 'remaining_open_positions' in locals() and remaining_open_positions:
    if remaining_open_positions:
        st.subheader("פוזיציות שעדיין פתוחות - פרטים")
        raw_open = []
        for pos in remaining_open_positions:
@@ -1309,45 +1090,35 @@ def normalize_price_columns(df):
            pass
        st.dataframe(raw_open_df)

    # Plot price with markers (including after-period exits and open positions)
    st.subheader("גרף מחירים - כניסות/יציאות")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_full.index, df_full['Close'], label=f"{ticker} Close (fetched range)")
    # closed during period
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            try:
                entry_dt = pd.to_datetime(row['entry_date'])
                exit_dt = pd.to_datetime(row['exit_date'])
                ax.scatter(entry_dt, row['entry_price'], marker='^', s=80, color='green')
                ax.scatter(exit_dt, row['exit_price'], marker='v', s=80, color='red')
            except Exception:
                pass
    # after_period closures
    if 'after_trades' in locals():
        for at in after_trades:
            try:
                entry_dt = pd.to_datetime(at['entry_date'])
                exit_dt = pd.to_datetime(at['exit_date'])
                ax.scatter(entry_dt, at['entry_price'], marker='^', s=80, color='green')
                ax.scatter(exit_dt, at['exit_price'], marker='x', s=100, color='blue', label='Exit after period')
            except Exception:
                pass
    # open positions
    if 'remaining_open_positions' in locals():
        for pos in remaining_open_positions:
            try:
                entry_dt = pd.to_datetime(pos.get('entry_date'))
                ax.scatter(entry_dt, pos.get('entry_price'), marker='^', s=120, facecolors='none', edgecolors='orange', linewidths=2, label='Open Position')
            except Exception:
                pass
            entry_dt = pd.to_datetime(row['entry_date'])
            exit_dt = pd.to_datetime(row['exit_date'])
            ax.scatter(entry_dt, row['entry_price'], marker='^', s=80, color='green')
            ax.scatter(exit_dt, row['exit_price'], marker='v', s=80, color='red')
    for at in after_trades:
        try:
            entry_dt = pd.to_datetime(at['entry_date'])
            exit_dt = pd.to_datetime(at['exit_date'])
            ax.scatter(entry_dt, at['entry_price'], marker='^', s=80, color='green')
            ax.scatter(exit_dt, at['exit_price'], marker='x', s=100, color='blue', label='Exit after period')
        except Exception:
            pass
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
@@ -1358,7 +1129,6 @@ def normalize_price_columns(df):
        ax2.legend()
        st.pyplot(fig2)

    # PNG & PDF downloads
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
@@ -1373,18 +1143,14 @@ def normalize_price_columns(df):
        fig_text = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        text = f"Backtest Summary for {ticker}\n\n"
        try:
            for k, v in summary.items():
                text += f"{k}: {v}\n"
        except Exception:
            text += "No summary available\n"
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
@@ -1402,4 +1168,3 @@ def normalize_price_columns(df):
            st.write(f"Buy & Hold return for period: {bh_return:.2f}%")

# EOF
