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
# Corrected paths to use absolute paths within the Docker container
APP_ROOT = "/app"
CONFIG_PATH = os.path.join(APP_ROOT, "config.yaml")
DB_DIR = os.path.join(APP_ROOT, "database")
KEYS_DB_PATH = os.path.join(DB_DIR, "api_keys.db")
TRADES_DB_PATH = os.path.join(DB_DIR, "trades.db")

# Configure logging for the panel
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# --- Database Functions ---
def get_db_connection(db_path):
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database at {db_path}: {e}")
        st.error(f"Database connection error: {e}")
        return None

def init_databases():
    """Initializes databases and tables if they don't exist."""
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        # Init Keys DB
        conn_keys = get_db_connection(KEYS_DB_PATH)
        if conn_keys:
            cursor_keys = conn_keys.cursor()
            cursor_keys.execute("CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY, api_key TEXT, api_secret TEXT)")
            conn_keys.commit()
            conn_keys.close()
            logger.info(f"API keys database initialized at {KEYS_DB_PATH}")
        
        # Init Trades DB
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
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    """Verifies login credentials against the config file."""
    try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        
        stored_username = config.get("panel", {}).get("username", "admin")
        stored_password_plain = config.get("panel", {}).get("password", "password")
        stored_password_hash = hash_password(stored_password_plain)
        
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

# --- API Key Functions ---
def save_api_keys(api_key, api_secret):
    """Saves or updates API keys in the database."""
    conn = get_db_connection(KEYS_DB_PATH)
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to handle both initial save and update
        cursor.execute("INSERT OR REPLACE INTO api_keys (id, api_key, api_secret) VALUES (1, ?, ?)", (api_key, api_secret))
        conn.commit()
        logger.info("API keys saved successfully.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error saving API keys: {e}")
        st.error(f"Database error saving keys: {e}")
        return False
    finally:
        if conn: conn.close()

def load_api_keys():
    """Loads API keys from the database."""
    conn = get_db_connection(KEYS_DB_PATH)
    if not conn:
        return None, None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT api_key, api_secret FROM api_keys WHERE id = 1")
        result = cursor.fetchone()
        if result:
            return result["api_key"], result["api_secret"]
        else:
            return None, None
    except sqlite3.Error as e:
        logger.error(f"Database error loading API keys: {e}")
        st.error(f"Database error loading keys: {e}")
        return None, None
    finally:
        if conn: conn.close()

# --- Reporting Functions ---
def get_trade_history(limit=100):
    """Retrieves recent trade history from the database."""
    conn = get_db_connection(TRADES_DB_PATH)
    if not conn:
        return pd.DataFrame()
    try:
        query = f"SELECT timestamp, symbol, side, price, quantity, order_id, status, pnl FROM trades ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        # Convert timestamp to readable format
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
    """Retrieves recent daily reports from the database."""
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
    """Generates a Plotly chart for daily PnL."""
    if reports_df.empty or "date" not in reports_df.columns or "total_pnl" not in reports_df.columns:
        return go.Figure()
    
    # Ensure data is sorted by date for plotting
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
        plot_bgcolor="rgba(0,0,0,0)", # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white") # Adjust font color if needed
    )
    return fig

def plot_cumulative_pnl(reports_df):
    """Generates a Plotly chart for cumulative PnL."""
    if reports_df.empty or "date" not in reports_df.columns or "total_pnl" not in reports_df.columns:
        return go.Figure()
        
    # Ensure data is sorted by date and calculate cumulative PnL
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

# Initialize databases on first run
init_databases()

# --- Login Handling ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Trading Bot Login")
    
    # Simple login form inspired by WordPress (minimalist)
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In")
        
        if submitted:
            if verify_login(username, password):
                st.session_state.logged_in = True
                st.rerun() # Rerun the script to show the main panel
            else:
                st.error("Invalid username or password.")
else:
    # --- Main Panel (after login) ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "API Configuration", "Trade History", "Settings"])
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title(f"Trading Bot Panel - {page}")

    # --- Dashboard Page ---
    if page == "Dashboard":
        st.header("Performance Overview")
        
        reports_df = get_daily_reports()
        
        if not reports_df.empty:
            # Display key metrics from the latest report
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

    # --- API Configuration Page ---
    elif page == "API Configuration":
        st.header("Binance API Keys")
        st.warning("Handle your API keys with extreme care. Ensure they have only the necessary permissions (trading).", icon="⚠️")
        
        current_api_key, current_api_secret = load_api_keys()
        
        with st.form("api_key_form"):
            api_key = st.text_input("API Key", value=current_api_key or "", type="password")
            api_secret = st.text_input("Secret Key", value=current_api_secret or "", type="password")
            submitted = st.form_submit_button("Save API Keys")
            
            if submitted:
                if api_key and api_secret:
                    if save_api_keys(api_key, api_secret):
                        st.success("API keys saved successfully! The bot will attempt to use them on its next check.")
                    else:
                        st.error("Failed to save API keys.")
                else:
                    st.warning("Please provide both API Key and Secret Key.")
                    
        st.info("After saving, the running bot should automatically pick up the keys within a few minutes and attempt to connect to Binance.")

    # --- Trade History Page ---
    elif page == "Trade History":
        st.header("Recent Trades")
        trade_history_df = get_trade_history()
        if not trade_history_df.empty:
            st.dataframe(trade_history_df, use_container_width=True)
        else:
            st.info("No trade history found yet.")
            
    # --- Settings Page (Placeholder) ---
    elif page == "Settings":
        st.header("Bot Settings")
        st.info("This section is a placeholder for future settings adjustments (e.g., risk parameters, strategy toggles). Currently, settings are managed via the `config.yaml` file.")
        
        try:
            with open(CONFIG_PATH, "r") as f:
                config_content = f.read()
            st.subheader("Current `config.yaml`")
            st.code(config_content, language="yaml")
        except Exception as e:
            st.error(f"Could not read config file at {CONFIG_PATH}: {e}")

