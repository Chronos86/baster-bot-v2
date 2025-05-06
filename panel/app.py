# panel/app.py

import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import os
import yaml
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# --- Configuration & Setup ---
APP_ROOT = "/app"
CONFIG_PATH = os.path.join(APP_ROOT, "config.yaml")
DB_DIR = os.path.join(APP_ROOT, "database")
# KEYS_DB_PATH is no longer used as keys are from env vars
TRADES_DB_PATH = os.path.join(DB_DIR, "trades.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# --- Database Functions ---
def get_db_connection(db_path):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database at {db_path}: {e}")
        st.error(f"Database connection error: {e}")
        return None

def init_databases():
    """Initializes trades database. API keys are now managed via environment variables."""
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        # API Keys DB (api_keys.db) is no longer initialized here.
        # logger.info(f"API keys database setup skipped as keys are managed via environment variables.")
        
        conn_trades = get_db_connection(TRADES_DB_PATH)
        if conn_trades:
            cursor_trades = conn_trades.cursor()
            cursor_trades.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    order_id TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    pnl REAL DEFAULT 0.0
                )
            """)
            cursor_trades.execute("""
                CREATE TABLE IF NOT EXISTS daily_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    total_pnl REAL DEFAULT 0.0,
                    trades_count INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_pnl_per_trade REAL DEFAULT 0.0,
                    best_indicator TEXT
                )
            """)
            conn_trades.commit()
            conn_trades.close()
            logger.info(f"Trades database initialized at {TRADES_DB_PATH}")
            
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}", exc_info=True)
        st.error(f"Failed to initialize databases: {e}")

# --- Authentication Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        
        stored_username = config.get("panel", {}).get("username", "admin")
        stored_password_plain = config.get("panel", {}).get("password", "password") # Plain password from config
        stored_password_hash = hash_password(stored_password_plain) # Hash it for comparison
        
        if username == stored_username and hash_password(password) == stored_password_hash:
            return True
        else:
            return False
    except FileNotFoundError:
        logger.error(f"Config file not found at {CONFIG_PATH} during login verification.")
        st.error(f"Configuration file missing at expected path: {CONFIG_PATH}. Cannot verify login.")
        return False
    except Exception as e:
        logger.error(f"Error reading config during login: {e}")
        st.error(f"Error reading configuration: {e}")
        return False

# API Key functions (save_api_keys, load_api_keys) are removed as keys are now from env vars.

# --- Reporting Functions ---
def get_trade_history(limit=100):
    conn = get_db_connection(TRADES_DB_PATH)
    if not conn:
        return pd.DataFrame()
    try:
        query = f"SELECT timestamp, symbol, side, price, quantity, order_id, status, pnl FROM trades ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        if not df.empty and "timestamp" in df.columns:
             df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        st.error(f"Error fetching trade history: {e}")
        return pd.DataFrame()
    finally:
        if conn: conn.close()

def get_daily_reports(limit=30):
    conn = get_db_connection(TRADES_DB_PATH)
    if not conn:
        return pd.DataFrame()
    try:
        query = f"SELECT date, total_pnl, trades_count, win_rate, avg_pnl_per_trade, best_indicator FROM daily_reports ORDER BY date DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.error(f"Error fetching daily reports: {e}")
        st.error(f"Error fetching daily reports: {e}")
        return pd.DataFrame()
    finally:
        if conn: conn.close()

# --- Plotting Functions ---
def plot_daily_pnl(reports_df):
    if reports_df.empty or "date" not in reports_df.columns or "total_pnl" not in reports_df.columns:
        return go.Figure()
    reports_df = reports_df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=reports_df["date"],
        y=reports_df["total_pnl"],
        name="Daily PnL",
        marker_color=["green" if pnl >= 0 else "red" for pnl in reports_df["total_pnl"]]
    ))
    fig.update_layout(
        title="Daily Profit and Loss (Estimated)",
        xaxis_title="Date",
        yaxis_title="PnL (USDT)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def plot_cumulative_pnl(reports_df):
    if reports_df.empty or "date" not in reports_df.columns or "total_pnl" not in reports_df.columns:
        return go.Figure()
    reports_df = reports_df.sort_values("date")
    reports_df["cumulative_pnl"] = reports_df["total_pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reports_df["date"],
        y=reports_df["cumulative_pnl"],
        mode="lines+markers",
        name="Cumulative PnL",
        line=dict(color="cyan")
    ))
    fig.update_layout(
        title="Cumulative Profit and Loss (Estimated)",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USDT)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

# --- Streamlit App Layout ---
st.set_page_config(page_title="Trading Bot Panel", layout="wide", initial_sidebar_state="collapsed")
init_databases()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Trading Bot Login")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In")
        if submitted:
            if verify_login(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
else:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "API Configuration", "Trade History", "Settings"])
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title(f"Trading Bot Panel - {page}")

    if page == "Dashboard":
        st.header("Performance Overview")
        reports_df = get_daily_reports()
        if not reports_df.empty:
            latest_report = reports_df.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Daily PnL (Est.)", f"{latest_report['total_pnl']:.2f} USDT", delta=f"{latest_report['total_pnl']:.2f}")
            col2.metric("Latest Win Rate (Est.)", f"{latest_report['win_rate']:.2f}%")
            col3.metric("Latest Indicator", latest_report["best_indicator"].upper())
            st.plotly_chart(plot_daily_pnl(reports_df), use_container_width=True)
            st.plotly_chart(plot_cumulative_pnl(reports_df), use_container_width=True)
            st.subheader("Recent Daily Reports")
            st.dataframe(reports_df, use_container_width=True)
        else:
            st.info("No daily reports found yet. The bot needs to run and generate reports.")

    elif page == "API Configuration":
        st.header("Binance API Key Configuration (via Environment Variables)")
        st.warning("API Keys are now managed via Environment Variables in your Render.com service settings for improved security and persistence.", icon="🔒")
        
        st.markdown("""
        **How to set up your API Keys:**

        1.  Go to your service settings on **Render.com**.
        2.  Navigate to the **Environment** section.
        3.  Add the following two **Environment Variables**:
            *   `BINANCE_API_KEY` : Your Binance API Key
            *   `BINANCE_API_SECRET` : Your Binance API Secret Key
        4.  **Save the changes** in Render.com.
        5.  **Redeploy your service** on Render.com if prompted, or the bot should pick up the new variables on its next restart.

        **Important Notes:**
        *   Ensure your API keys have the necessary permissions on Binance (e.g., `Enable Reading`, `Enable Spot & Margin Trading`).
        *   **NEVER** enable `Enable Withdrawals` for API keys used by bots.
        *   Make sure the keys correspond to the correct environment (Testnet vs. Mainnet) as configured in your `config.yaml` (`testnet: true` or `testnet: false`).
        *   The bot will automatically attempt to load these keys when it starts.
        """)
        
        # Check if environment variables seem to be set (this is just a hint, bot actually uses them)
        env_api_key_present = bool(os.getenv("BINANCE_API_KEY"))
        env_api_secret_present = bool(os.getenv("BINANCE_API_SECRET"))

        if env_api_key_present and env_api_secret_present:
            st.success("Environment variables BINANCE_API_KEY and BINANCE_API_SECRET appear to be set in the environment. The bot will attempt to use them.", icon="✅")
        else:
            st.error("One or both environment variables (BINANCE_API_KEY, BINANCE_API_SECRET) do not seem to be set in this panel's environment. Please ensure they are correctly set in your Render.com service environment settings for the bot to function.", icon="⚠️")

    elif page == "Trade History":
        st.header("Recent Trades")
        trade_history_df = get_trade_history()
        if not trade_history_df.empty:
            st.dataframe(trade_history_df, use_container_width=True)
        else:
            st.info("No trade history found yet.")
            
    elif page == "Settings":
        st.header("Bot Settings")
        st.info("This section is a placeholder for future settings adjustments. Currently, most settings are managed via the `config.yaml` file.")
        try:
            with open(CONFIG_PATH, "r") as f:
                config_content = f.read()
            st.subheader("Current `config.yaml`")
            st.code(config_content, language="yaml")
        except Exception as e:
            st.error(f"Could not read config file at {CONFIG_PATH}: {e}")

