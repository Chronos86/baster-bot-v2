# bot/main.py

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
import sqlite3
import schedule
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict, Any # Added for type hinting

# --- Binance & Notifications ---
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from binance import ThreadedWebsocketManager
import apprise

# --- Data Processing & ML ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError # Added for LSTM error handling

# Configure logging first to capture import issues
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Attempt to import TA-Lib with fallback
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib library found and imported.")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib library not found. Using mock indicator functions.")
    # --- Mock TA-Lib Implementation ---
    class MockTALib:
        def EMA(self, series, timeperiod):
            logger.debug(f"MockTALib: EMA({timeperiod}) called")
            if len(series) < timeperiod:
                return np.full(len(series), np.nan)
            return pd.Series(series).rolling(window=timeperiod).mean().to_numpy()

        def RSI(self, series, timeperiod):
            logger.debug(f"MockTALib: RSI({timeperiod}) called")
            if len(series) < timeperiod + 1:
                return np.full(len(series), np.nan)
            return np.full(len(series), 50.0)

        def BBANDS(self, series, timeperiod, nbdevup=2, nbdevdn=2, matype=0):
            logger.debug(f"MockTALib: BBANDS({timeperiod}) called")
            if len(series) < timeperiod:
                nan_array = np.full(len(series), np.nan)
                return nan_array, nan_array, nan_array
            middle = pd.Series(series).rolling(window=timeperiod).mean().to_numpy()
            upper = middle * 1.02
            lower = middle * 0.98
            return upper, middle, lower

    talib = MockTALib()
    # --- End Mock TA-Lib ---

# Attempt to import TensorFlow (optional, requires installation)
try:
    import importlib
    importlib.util.find_spec("tensorflow")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
    logging.info("TensorFlow library found and imported.")
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    logging.warning("TensorFlow library not found. Using mock LSTM implementation. Please install TensorFlow for actual price prediction.")

# --- Local Modules ---
from bot.indicators import evaluate_indicators
from bot.sentiment import get_sentiment

# --- Basic Configuration ---
load_dotenv()

# --- Mock LSTM Implementation (if TensorFlow is not available) ---
if not TF_AVAILABLE:
    class MockLSTMModel:
        def __init__(self, name="MockLSTM"):
            self.name = name
            logger.info(f"Initialized {self.name}")

        def fit(self, X, y, epochs=5, batch_size=32, verbose=0):
            logger.info(f"{self.name}: Mock training with {len(X)} samples...")
            time.sleep(0.1)

        def predict(self, X, verbose=0):
            logger.info(f"{self.name}: Mock predicting...")
            last_price_scaled = X[0, -1, 0]
            prediction = np.array([[last_price_scaled * 1.001]])
            return prediction

        def build_lstm_model(self, input_shape):
            logger.info(f"{self.name}: Mock model build requested with input shape {input_shape}. No actual build needed.")
            return self
# --- End Mock LSTM ---

class ScalpingBot:
    def __init__(self, config_path="config.yaml", db_dir="database"):
        self.config_path = config_path
        self.db_dir = db_dir
        self.config = {}
        self.api_key = None
        self.api_secret = None
        self.client = None
        self.twm = None
        self.ws_kline_key = None
        self.ws_user_key = None
        self.is_websocket_running = False
        self.symbol = "BTCUSDT"
        self.position_size = 0.001
        self.max_orders_per_minute = 90
        self.order_count_minute = 0
        self.last_minute_reset = time.time()
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.max_daily_loss_pct = 0.6
        self.max_daily_loss_usd = 0
        self.daily_pnl = 0
        self.last_daily_reset = datetime.now().date()
        self.stop_trading_today = False
        self.db_conn_trades = None
        # self.db_conn_keys = None # Removed, keys from env vars
        self.apprise = None
        self.prices: List[Tuple[int, float, float]] = []
        self.scaler_price = MinMaxScaler()
        self.scaler_volume = MinMaxScaler()
        self.model = None
        self.lstm_look_back = 10
        self.lstm_train_interval_seconds = 3600
        self.best_indicator = "ema"
        self.sentiment_score = 0.0
        self.last_indicator_update = datetime.min
        self.last_sentiment_update = time.time()
        self.last_lstm_train = time.time()
        self.sentiment_update_interval_seconds = 900
        self.indicator_update_interval_hours = 24
        self.stop_loss_pct = 0.005
        self.take_profit_pct = 0.01

        self.tf_available = TF_AVAILABLE # Store global TF_AVAILABLE status
        self.talib_available = TALIB_AVAILABLE # Store global TALIB_AVAILABLE status

        if not self._load_config(): exit(1)
        if not self._setup_databases(): exit(1) # Sets up trades.db only
        self._setup_notifications()

    def _load_config(self) -> bool:
        try:
            with open(self.config_path, "r") as file:
                self.config = yaml.safe_load(file)
            binance_cfg = self.config.get("binance", {})
            self.symbol = binance_cfg.get("symbol", "BTCUSDT")
            self.position_size = float(binance_cfg.get("position_size", 0.001))
            self.is_testnet = binance_cfg.get("testnet", True)
            risk_cfg = self.config.get("risk", {})
            self.max_daily_loss_pct = float(risk_cfg.get("max_daily_loss", 0.6))
            self.stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.005))
            self.take_profit_pct = float(risk_cfg.get("take_profit_pct", 0.01))
            self.max_daily_loss_usd = self.initial_capital * self.max_daily_loss_pct
            lstm_cfg = self.config.get("lstm", {})
            self.lstm_look_back = int(lstm_cfg.get("look_back", 10))
            self.lstm_train_interval_seconds = int(lstm_cfg.get("train_interval_seconds", 3600))
            update_cfg = self.config.get("update_intervals", {})
            self.sentiment_update_interval_seconds = int(update_cfg.get("sentiment_update_seconds", 900))
            self.indicator_update_interval_hours = int(update_cfg.get("indicator_update_hours", 24))
            logger.info("Configuration loaded successfully.")
            return True
        except FileNotFoundError:
            logger.error(f"CRITICAL: Configuration file not found at {self.config_path}.")
            return False
        except (ValueError, TypeError) as e:
            logger.error(f"CRITICAL: Error parsing configuration values: {e}")
            return False
        except Exception as e:
            logger.error(f"CRITICAL: Unexpected error loading configuration: {e}", exc_info=True)
            return False

    def _setup_databases(self) -> bool:
        """Initializes SQLite database for trades/reports. API keys are now loaded from env vars."""
        try:
            os.makedirs(self.db_dir, exist_ok=True)
            trades_db_path = os.path.join(self.db_dir, "trades.db")
            # API keys database (api_keys.db) setup is skipped as keys are loaded from env vars.
            # logger.info("API keys database (api_keys.db) setup skipped as keys are loaded from env vars.")

            self.db_conn_trades = sqlite3.connect(trades_db_path, check_same_thread=False)
            cursor_trades = self.db_conn_trades.cursor()
            cursor_trades.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL, /* BUY or SELL */
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    order_id TEXT UNIQUE NOT NULL, /* Binance Order ID */
                    status TEXT DEFAULT 'OPEN', /* OPEN, CLOSED, CANCELED */
                    pnl REAL DEFAULT 0.0 /* Realized PnL for this trade when closed */
                )
            """)
            cursor_trades.execute("""
                CREATE TABLE IF NOT EXISTS daily_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL, /* YYYY-MM-DD */
                    total_pnl REAL DEFAULT 0.0,
                    trades_count INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_pnl_per_trade REAL DEFAULT 0.0,
                    best_indicator TEXT
                )
            """)
            self.db_conn_trades.commit()
            logger.info("Trades database initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"CRITICAL: Error setting up trades database in {self.db_dir}: {e}", exc_info=True)
            return False

    def _setup_notifications(self):
        try:
            self.apprise = apprise.Apprise()
            tg_token = self.config.get("telegram", {}).get("bot_token", "YOUR_TELEGRAM_BOT_TOKEN")
            tg_chat_id = self.config.get("telegram", {}).get("chat_id", "YOUR_TELEGRAM_CHAT_ID")
            if tg_token and "YOUR_TELEGRAM" not in tg_token and tg_chat_id and "YOUR_TELEGRAM" not in str(tg_chat_id):
                self.apprise.add(f"tgram://{tg_token}/{tg_chat_id}")
                logger.info("Telegram notifications configured.")
                self.notify("Trading bot starting up...")
            else:
                logger.warning("Telegram token/chat ID not configured or using placeholders. Telegram notifications disabled.")
        except Exception as e:
            logger.error(f"Error setting up Apprise notifications: {e}", exc_info=True)

    def notify(self, message: str):
        logger.info(f"Notification: {message}")
        if self.apprise and self.apprise.servers:
            try:
                self.apprise.notify(body=message)
            except Exception as e:
                logger.error(f"Failed to send notification via Apprise: {e}")

    def _load_api_keys(self) -> bool:
        """Loads API keys from environment variables (BINANCE_API_KEY, BINANCE_API_SECRET)."""
        if self.api_key and self.api_secret: # Already loaded
            return True
            
        logger.info("Attempting to load API keys from environment variables (BINANCE_API_KEY, BINANCE_API_SECRET)...")
        env_api_key = os.getenv("BINANCE_API_KEY")
        env_api_secret = os.getenv("BINANCE_API_SECRET")

        if env_api_key and env_api_secret:
            self.api_key = env_api_key
            self.api_secret = env_api_secret
            logger.info("API keys loaded successfully from environment variables.")
            return True
        else:
            logger.warning("API keys (BINANCE_API_KEY and/or BINANCE_API_SECRET) not found in environment variables.")
            logger.warning("Please ensure BINANCE_API_KEY and BINANCE_API_SECRET are set as environment variables in your Render.com service settings.")
            return False

    def _initialize_client(self) -> bool:
        if not (self.api_key and self.api_secret):
            logger.warning("Cannot initialize Binance client: API keys missing (ensure they are set as env vars).")
            return False
        if self.client:
            return True
        try:
            logger.info(f"Initializing Binance client (Testnet: {self.is_testnet})...")
            self.client = Client(self.api_key, self.api_secret, testnet=self.is_testnet)
            self.client.ping()
            logger.info("Binance client ping successful.")
            account_info = self.client.get_account()
            usdt_balance = 0.0
            for balance in account_info.get("balances", []):
                if balance["asset"] == "USDT":
                    usdt_balance = float(balance["free"])
                    break
            if usdt_balance > 0:
                self.current_capital = usdt_balance
                self.initial_capital = self.current_capital
                self.max_daily_loss_usd = self.initial_capital * self.max_daily_loss_pct
                logger.info(f"Connected to Binance. Initial USDT balance: {self.current_capital:.2f}. Max daily loss set to: {self.max_daily_loss_usd:.2f} USDT")
            else:
                 logger.warning("Could not retrieve USDT balance from Binance account.")
            return True
        except BinanceAPIException as e:
            logger.error(f"Binance API Error during client initialization: {e}. Check API keys, permissions, and network.")
            self.notify(f"CRITICAL: Binance API Error: {e}. Bot cannot connect.")
            self.api_key = None
            self.api_secret = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing Binance client: {e}", exc_info=True)
            return False

    def _build_lstm_model(self):
        if self.model:
            return
        input_shape = (self.lstm_look_back, 2)
        if self.tf_available: # Use instance variable
            try:
                logger.info(f"Building TensorFlow LSTM model with input shape {input_shape}...")
                self.model = Sequential(name="TensorFlowLSTM")
                self.model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
                self.model.add(LSTM(50, return_sequences=False))
                self.model.add(Dense(25))
                self.model.add(Dense(1))
                self.model.compile(optimizer="adam", loss="mean_squared_error")
                logger.info("TensorFlow LSTM model built successfully.")
            except Exception as e:
                logger.error(f"Error building TensorFlow LSTM model: {e}. Falling back to mock LSTM.", exc_info=True)
                self.tf_available = False # Mark TF as unavailable for this session if build fails
                self.model = MockLSTMModel() # Fallback to mock
        else:
            logger.info("TensorFlow not available. Using Mock LSTM model.")
            self.model = MockLSTMModel()
            self.model.build_lstm_model(input_shape) # Call mock build

    def _train_lstm_model(self):
        if not self.tf_available or not self.model or isinstance(self.model, MockLSTMModel):
            logger.info("Skipping LSTM training: TensorFlow not available or using mock model.")
            return
        if not self.prices or len(self.prices) < self.lstm_look_back + 1:
            logger.info("Skipping LSTM training: Not enough price data.")
            return

        try:
            logger.info("Preparing data for LSTM training...")
            df = pd.DataFrame(self.prices, columns=["timestamp", "price", "volume"])
            df_scaled_price = self.scaler_price.fit_transform(df[["price"]])
            df_scaled_volume = self.scaler_volume.fit_transform(df[["volume"]])
            
            X, y = [], []
            for i in range(len(df) - self.lstm_look_back):
                X.append(np.hstack((df_scaled_price[i:i+self.lstm_look_back], df_scaled_volume[i:i+self.lstm_look_back])))
                y.append(df_scaled_price[i+self.lstm_look_back, 0])
            X, y = np.array(X), np.array(y)

            if X.shape[0] == 0:
                logger.warning("No samples generated for LSTM training. Skipping.")
                return

            logger.info(f"Training LSTM model with {X.shape[0]} samples...")
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            self.last_lstm_train = time.time()
            logger.info("LSTM model training complete.")
            self.notify("LSTM model re-trained successfully.")
        except NotFittedError as e:
            logger.error(f"LSTM Scaler NotFittedError during training: {e}. Ensure scalers are fit before transform.")
        except Exception as e:
            logger.error(f"Error during LSTM model training: {e}", exc_info=True)

    def _get_lstm_prediction(self) -> Optional[float]:
        if not self.prices or len(self.prices) < self.lstm_look_back:
            return None
        if not self.model:
            logger.warning("LSTM model not built. Cannot predict.")
            return None
        try:
            df = pd.DataFrame(self.prices, columns=["timestamp", "price", "volume"])
            # Ensure scalers are fitted. If not, fit them with available data.
            if not hasattr(self.scaler_price, "scale_") or not hasattr(self.scaler_volume, "scale_"):
                logger.warning("LSTM scalers not fitted. Fitting with current price data before prediction.")
                self.scaler_price.fit(df[["price"]][-self.lstm_look_back*2:]) # Fit on recent data
                self.scaler_volume.fit(df[["volume"]][-self.lstm_look_back*2:])

            last_sequence_price = self.scaler_price.transform(df[["price"]][-self.lstm_look_back:])
            last_sequence_volume = self.scaler_volume.transform(df[["volume"]][-self.lstm_look_back:])
            last_sequence = np.hstack((last_sequence_price, last_sequence_volume))
            prediction_scaled = self.model.predict(np.array([last_sequence]), verbose=0)
            prediction_actual = self.scaler_price.inverse_transform(prediction_scaled)
            return float(prediction_actual[0,0])
        except NotFittedError as e:
            logger.error(f"LSTM Scaler NotFittedError during prediction: {e}. This shouldn't happen if fitting logic is correct.")
            return None # Or handle by fitting scalers
        except Exception as e:
            logger.error(f"Error getting LSTM prediction: {e}", exc_info=True)
            return None

    def _update_indicators_evaluation(self):
        if not self.prices or len(self.prices) < 60: # Need enough data for evaluation
            logger.info("Not enough price data to evaluate indicators. Skipping update.")
            return
        try:
            df = pd.DataFrame(self.prices, columns=["timestamp", "price", "volume"])
            df.set_index("timestamp", inplace=True)
            # Pass TALIB_AVAILABLE to evaluate_indicators
            self.best_indicator, best_params = evaluate_indicators(df, self.talib_available)
            self.last_indicator_update = datetime.now()
            logger.info(f"Best indicator updated to: {self.best_indicator} with params {best_params}")
            self.notify(f"Indicator strategy updated. Best indicator: {self.best_indicator.upper()}")
        except Exception as e:
            logger.error(f"Error updating indicators evaluation: {e}", exc_info=True)

    def _update_sentiment(self):
        try:
            self.sentiment_score = get_sentiment(self.config)
            self.last_sentiment_update = time.time()
            logger.info(f"Sentiment score updated: {self.sentiment_score:.3f}")
        except Exception as e:
            logger.error(f"Error updating sentiment: {e}", exc_info=True)

    def _check_rate_limits(self) -> bool:
        current_time = time.time()
        if current_time - self.last_minute_reset > 60:
            self.order_count_minute = 0
            self.last_minute_reset = current_time
        if self.order_count_minute >= self.max_orders_per_minute:
            logger.warning("Order rate limit reached for this minute. No new orders will be placed.")
            return False
        return True

    def _check_risk_limit(self):
        today = datetime.now().date()
        if today != self.last_daily_reset:
            logger.info(f"Daily reset. Previous day PnL: {self.daily_pnl:.2f} USDT. Resetting daily PnL and loss limit.")
            self._save_daily_report() # Save report for previous day
            self.daily_pnl = 0
            self.stop_trading_today = False
            self.last_daily_reset = today
            # Update capital and max loss based on current balance if possible
            if self.client:
                try:
                    account_info = self.client.get_account()
                    usdt_balance = 0.0
                    for balance in account_info.get("balances", []):
                        if balance["asset"] == "USDT":
                            usdt_balance = float(balance["free"])
                            break
                    if usdt_balance > 0:
                        self.current_capital = usdt_balance
                        self.max_daily_loss_usd = self.current_capital * self.max_daily_loss_pct
                        logger.info(f"Daily capital updated to: {self.current_capital:.2f} USDT. New max daily loss: {self.max_daily_loss_usd:.2f} USDT")
                except Exception as e:
                    logger.error(f"Error updating capital during daily reset: {e}")
        
        if self.daily_pnl <= -self.max_daily_loss_usd:
            if not self.stop_trading_today:
                logger.critical(f"MAX DAILY LOSS LIMIT REACHED: {self.daily_pnl:.2f} USDT. Stopping trading for today.")
                self.notify(f"CRITICAL: Max daily loss limit reached ({self.daily_pnl:.2f} USDT). Trading stopped for today.")
                self.stop_trading_today = True
        return not self.stop_trading_today

    def _save_trade(self, symbol, side, price, qty, order_id, status="OPEN", pnl=0.0):
        try:
            cursor = self.db_conn_trades.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO trades (timestamp, symbol, side, price, quantity, order_id, status, pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (timestamp, symbol, side, price, qty, order_id, status, pnl))
            self.db_conn_trades.commit()
            logger.info(f"Trade saved: {side} {qty} {symbol} @ {price}, OrderID: {order_id}")
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}", exc_info=True)

    def _update_trade_status(self, order_id, new_status, pnl=None):
        try:
            cursor = self.db_conn_trades.cursor()
            if pnl is not None:
                cursor.execute("UPDATE trades SET status = ?, pnl = ? WHERE order_id = ?", (new_status, pnl, order_id))
            else:
                cursor.execute("UPDATE trades SET status = ? WHERE order_id = ?", (new_status, order_id))
            self.db_conn_trades.commit()
            logger.info(f"Trade status updated for OrderID {order_id} to {new_status}" + (f" with PnL {pnl:.2f}" if pnl is not None else ""))
        except Exception as e:
            logger.error(f"Error updating trade status in database for OrderID {order_id}: {e}", exc_info=True)

    def _save_daily_report(self):
        try:
            # Calculate stats for the report
            date_str = self.last_daily_reset.strftime("%Y-%m-%d")
            cursor = self.db_conn_trades.cursor()
            cursor.execute("SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), AVG(pnl) FROM trades WHERE date(timestamp) = ? AND status = 'CLOSED'", (date_str,))
            result = cursor.fetchone()
            trades_count = result[0] if result and result[0] is not None else 0
            win_count = result[1] if result and result[1] is not None else 0
            avg_pnl = result[2] if result and result[2] is not None else 0.0
            win_rate = (win_count / trades_count * 100) if trades_count > 0 else 0.0

            cursor.execute("INSERT OR REPLACE INTO daily_reports (date, total_pnl, trades_count, win_rate, avg_pnl_per_trade, best_indicator) VALUES (?, ?, ?, ?, ?, ?)",
                           (date_str, self.daily_pnl, trades_count, win_rate, avg_pnl, self.best_indicator))
            self.db_conn_trades.commit()
            logger.info(f"Daily report saved for {date_str}. Total PnL: {self.daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"Error saving daily report: {e}", exc_info=True)

    def _place_order(self, side: str, symbol: str, quantity: float, order_type=ORDER_TYPE_MARKET) -> Optional[Dict]:
        if not self._check_rate_limits(): return None
        if not self._check_risk_limit(): return None
        if not self.client: 
            logger.error("Cannot place order: Binance client not initialized.")
            return None

        try:
            logger.info(f"Attempting to place {side} order for {quantity} {symbol}...")
            if side == "BUY":
                order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            elif side == "SELL":
                order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            else:
                logger.error(f"Invalid order side: {side}")
                return None
            
            logger.info(f"Order placed: {order}")
            self.order_count_minute += 1
            # Save trade immediately with OPEN status
            self._save_trade(symbol, side, float(order["fills"][0]["price"]) if order.get("fills") else 0, quantity, order["orderId"])
            self.notify(f"Order Placed: {side} {quantity} {symbol} @ Market. OrderID: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API Exception placing order: {e}")
            self.notify(f"ERROR: Binance API Exception placing order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error placing order: {e}", exc_info=True)
            return None

    def _handle_kline_message(self, msg):
        try:
            if msg.get("e") == "error":
                logger.error(f"Kline WebSocket Error: {msg.get('m')}")
                # Consider attempting to restart the websocket or client
                self.is_websocket_running = False # Mark as not running to trigger restart
                return

            kline = msg.get("k")
            if kline:
                close_time = int(kline["T"]) # Kline close time (timestamp ms)
                price = float(kline["c"])   # Close price
                volume = float(kline["v"]) # Volume
                is_closed = kline.get("x", False) # Is this kline closed?

                # Add to prices list, keeping a fixed size (e.g., last 1000 candles for indicators)
                if is_closed: # Store only closed klines for consistent indicator calculation
                    self.prices.append((close_time, price, volume))
                    if len(self.prices) > 2000: # Keep a rolling window of prices
                        self.prices.pop(0)
                    # logger.debug(f"Kline closed: {datetime.fromtimestamp(close_time/1000)} P: {price} V: {volume}")

                # --- Trading Logic (example) ---
                if is_closed and len(self.prices) > 60: # Ensure enough data for indicators
                    self._execute_trading_logic(price)

        except Exception as e:
            logger.error(f"Error in kline message handler: {e}", exc_info=True)

    def _handle_user_data_message(self, msg):
        try:
            if msg.get("e") == "error":
                logger.error(f"User Data WebSocket Error: {msg.get('m')}")
                return

            if msg.get("e") == "executionReport": # Order update
                order_id = msg.get("i") # Order ID
                symbol = msg.get("s")
                side = msg.get("S") # BUY or SELL
                order_status = msg.get("X") # NEW, PARTIALLY_FILLED, FILLED, CANCELED, PENDING_CANCEL, REJECTED, EXPIRED
                price = float(msg.get("p")) # Order price
                quantity = float(msg.get("q")) # Order quantity
                commission = float(msg.get("n") or 0) # Commission
                commission_asset = msg.get("N")
                trade_time = datetime.fromtimestamp(int(msg.get("T"))/1000).strftime("%Y-%m-%d %H:%M:%S")
                
                logger.info(f"User Stream: Order Update - ID: {order_id}, Sym: {symbol}, Side: {side}, Status: {order_status}, Px: {price}, Qty: {quantity}")

                if order_status == "FILLED":
                    # Calculate PnL if it's a closing trade (more complex logic needed for partial fills / actual PnL)
                    # For simplicity, assume market orders fill at reported price
                    # This is a simplified PnL calculation, real PnL needs to track open positions
                    # For now, just update status. PnL calculation will be rough.
                    
                    # Find original trade in DB to calculate PnL
                    cursor = self.db_conn_trades.cursor()
                    cursor.execute("SELECT price, side, quantity FROM trades WHERE order_id = ? AND status = 'OPEN'", (str(order_id),))
                    trade_data = cursor.fetchone()
                    
                    pnl_for_this_trade = 0.0
                    if trade_data:
                        open_price = trade_data[0]
                        original_side = trade_data[1]
                        original_quantity = trade_data[2]
                        
                        # This is a very basic PnL, assumes this fill closes the order_id trade
                        if original_side == "BUY": # This fill is a SELL closing a BUY
                            pnl_for_this_trade = (price - open_price) * original_quantity - commission
                        elif original_side == "SELL": # This fill is a BUY closing a SELL
                            pnl_for_this_trade = (open_price - price) * original_quantity - commission
                        
                        self.daily_pnl += pnl_for_this_trade
                        self._update_trade_status(str(order_id), "CLOSED", pnl_for_this_trade)
                        self.notify(f"Trade Filled & Closed: {side} {quantity} {symbol} @ {price}. PnL: {pnl_for_this_trade:.2f} USDT. Daily PnL: {self.daily_pnl:.2f} USDT")
                    else:
                        # If not found as OPEN, it might be a new trade not yet fully processed by _place_order's save
                        # or it's an update for an already closed/canceled trade. For now, just log.
                        logger.info(f"Order {order_id} FILLED, but no matching OPEN trade found in DB for PnL calc, or already processed.")
                        self._update_trade_status(str(order_id), "CLOSED") # Mark as closed anyway

                elif order_status in ["CANCELED", "REJECTED", "EXPIRED"]:
                    self._update_trade_status(str(order_id), order_status)
                    self.notify(f"Order {order_status}: {side} {quantity} {symbol}. OrderID: {order_id}")
            
            elif msg.get("e") == "outboundAccountPosition": # Balance update
                # logger.debug(f"User Stream: Balance Update - {msg}")
                pass # Can be used to update current_capital more frequently
            elif msg.get("e") == "balanceUpdate":
                # logger.debug(f"User Stream: Balance Update (balanceUpdate) - {msg}")
                pass

        except Exception as e:
            logger.error(f"Error in user data message handler: {e}", exc_info=True)

    def _execute_trading_logic(self, current_price: float):
        if not self._check_risk_limit(): return
        if not self.talib_available: # Use self.talib_available
            logger.warning("Trading logic skipped: TA-Lib (or mock) not available.")
            return

        # --- Generate Signals (Example with EMA crossover) ---
        # This is a placeholder for the actual strategy using self.best_indicator
        # For now, let's use a simple EMA crossover as an example if prices are available
        if len(self.prices) < 50: return # Need enough data

        df_prices = pd.DataFrame(self.prices, columns=["timestamp", "price", "volume"])["price"]
        
        # Example: EMA 10 and EMA 20
        ema_short_period = 10
        ema_long_period = 20
        
        ema_short = talib.EMA(df_prices.to_numpy(), timeperiod=ema_short_period)
        ema_long = talib.EMA(df_prices.to_numpy(), timeperiod=ema_long_period)

        if np.isnan(ema_short[-1]) or np.isnan(ema_long[-1]) or np.isnan(ema_short[-2]) or np.isnan(ema_long[-2]):
            return # Not enough data for EMAs yet

        # Sentiment modifier (example: scale signal strength)
        sentiment_modifier = 1.0 + (self.sentiment_score * 0.2) # e.g., +20% strength for strong positive

        # LSTM prediction modifier (example: adjust entry price or confirm signal)
        lstm_pred = self._get_lstm_prediction()
        lstm_signal_confirm = 0 # 0: neutral, 1: confirms buy, -1: confirms sell
        if lstm_pred is not None:
            if lstm_pred > current_price * 1.001: # Predicts >0.1% rise
                lstm_signal_confirm = 1
            elif lstm_pred < current_price * 0.999: # Predicts >0.1% fall
                lstm_signal_confirm = -1

        # --- Decision Logic (Example) ---
        # Buy signal: short EMA crosses above long EMA
        if ema_short[-2] < ema_long[-2] and ema_short[-1] > ema_long[-1]:
            if lstm_signal_confirm >= 0: # LSTM doesn't contradict or confirms
                logger.info(f"BUY SIGNAL: EMA Crossover. EMA_S: {ema_short[-1]:.2f}, EMA_L: {ema_long[-1]:.2f}. Sentiment: {self.sentiment_score:.2f}. LSTM Pred: {lstm_pred if lstm_pred else 'N/A'}")
                self._place_order("BUY", self.symbol, self.position_size * sentiment_modifier)
        
        # Sell signal: short EMA crosses below long EMA
        elif ema_short[-2] > ema_long[-2] and ema_short[-1] < ema_long[-1]:
            if lstm_signal_confirm <= 0: # LSTM doesn't contradict or confirms
                logger.info(f"SELL SIGNAL: EMA Crossover. EMA_S: {ema_short[-1]:.2f}, EMA_L: {ema_long[-1]:.2f}. Sentiment: {self.sentiment_score:.2f}. LSTM Pred: {lstm_pred if lstm_pred else 'N/A'}")
                self._place_order("SELL", self.symbol, self.position_size * sentiment_modifier)

    def _start_websockets(self):
        if not self.client:
            logger.error("Cannot start websockets: Binance client not initialized.")
            return
        if self.is_websocket_running:
            logger.info("Websockets already running.")
            return

        try:
            self.twm = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.api_secret, testnet=self.is_testnet)
            self.twm.start()
            
            # Kline stream (e.g., 1 minute for BTCUSDT)
            kline_stream_name = f"{self.symbol.lower()}@kline_1m"
            self.ws_kline_key = self.twm.start_kline_socket(callback=self._handle_kline_message, symbol=self.symbol, interval=Client.KLINE_INTERVAL_1MINUTE)
            
            # User data stream
            self.ws_user_key = self.twm.start_user_socket(callback=self._handle_user_data_message)
            
            self.is_websocket_running = True
            logger.info(f"Started Kline WebSocket ({kline_stream_name}) and User Data WebSocket.")
            self.notify("WebSockets connected. Bot is now live trading.")
        except Exception as e:
            logger.error(f"Error starting websockets: {e}", exc_info=True)
            self.is_websocket_running = False # Ensure it's marked as not running

    def _stop_websockets(self):
        if self.twm and self.is_websocket_running:
            try:
                if self.ws_kline_key:
                    self.twm.stop_socket(self.ws_kline_key)
                    logger.info("Kline WebSocket stopped.")
                if self.ws_user_key:
                    self.twm.stop_socket(self.ws_user_key)
                    logger.info("User Data WebSocket stopped.")
                self.twm.join(timeout=5) # Wait for threads to finish
                logger.info("ThreadedWebsocketManager stopped.")
            except Exception as e:
                logger.error(f"Error stopping websockets: {e}", exc_info=True)
            finally:
                self.is_websocket_running = False
                self.twm = None
                self.ws_kline_key = None
                self.ws_user_key = None
        else:
            logger.info("WebSocket manager not running or not initialized.")

    def _setup_schedule(self):
        logger.info("Setting up scheduled tasks...")
        # Update indicators evaluation (e.g., daily)
        schedule.every(self.indicator_update_interval_hours).hours.do(self._update_indicators_evaluation)
        logger.info(f"Scheduled job: {schedule.jobs[-1]}")
        
        # Update sentiment (e.g., every 15 minutes)
        schedule.every(self.sentiment_update_interval_seconds).seconds.do(self._update_sentiment)
        logger.info(f"Scheduled job: {schedule.jobs[-1]}")

        # Train LSTM model (e.g., hourly)
        if self.tf_available:
            schedule.every(self.lstm_train_interval_seconds).seconds.do(self._train_lstm_model)
            logger.info(f"Scheduled job: {schedule.jobs[-1]}")
        
        # Daily risk limit check and PnL reset (runs slightly after midnight)
        schedule.every(1).days.at("00:01").do(self._check_risk_limit) # Ensures it runs after date changes
        logger.info(f"Scheduled job: {schedule.jobs[-1]}")

    def _close_db_connections(self):
        """Closes database connections."""
        if self.db_conn_trades:
            try:
                self.db_conn_trades.close()
                logger.info("Trades database connection closed.")
            except Exception as e:
                logger.error(f"Error closing trades database connection: {e}")
        # self.db_conn_keys connection is no longer managed here as it's not used by bot for loading

    def run(self):
        logger.info("Starting bot run loop...")
        self._setup_schedule()
        # Initial calls for updates
        self._update_sentiment()
        self._update_indicators_evaluation()
        self._build_lstm_model() # Build LSTM (real or mock)
        if self.tf_available: self._train_lstm_model() # Initial train if TF available

        try:
            while True:
                if not self._load_api_keys():
                    logger.info("API keys not available. Waiting 60 seconds...")
                    time.sleep(60)
                    continue

                if not self.client or not self.is_websocket_running:
                    if self._initialize_client():
                        self._start_websockets()
                    else:
                        logger.info("Failed to initialize Binance client. Waiting 60 seconds...")
                        time.sleep(60)
                        continue # Retry client initialization
                
                if not self.is_websocket_running and self.client:
                    logger.warning("Websocket is not running, attempting to restart...")
                    self._start_websockets()
                    if not self.is_websocket_running:
                        logger.error("Failed to restart websockets. Waiting 60 seconds...")
                        time.sleep(60)
                        continue

                schedule.run_pending()
                time.sleep(1) # Main loop runs every second

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down bot...")
        except Exception as e:
            logger.critical(f"CRITICAL UNHANDLED EXCEPTION in main loop: {e}", exc_info=True)
            self.notify(f"CRITICAL ERROR: Bot crashed with: {e}. Needs restart!")
        finally:
            logger.info("Initiating bot shutdown...")
            self._stop_websockets()
            self._close_db_connections()
            if schedule.jobs:
                schedule.clear()
                logger.info("Scheduled tasks cleared.")
            self.notify("Trading bot shutting down.")
            logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    bot = ScalpingBot(config_path="config.yaml", db_dir="database")
    bot.run()

