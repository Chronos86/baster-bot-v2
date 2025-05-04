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
            # Simple moving average as mock
            return pd.Series(series).rolling(window=timeperiod).mean().to_numpy()

        def RSI(self, series, timeperiod):
            logger.debug(f"MockTALib: RSI({timeperiod}) called")
            if len(series) < timeperiod + 1:
                return np.full(len(series), np.nan)
            # Mock RSI: return 50 always
            return np.full(len(series), 50.0)

        def BBANDS(self, series, timeperiod, nbdevup=2, nbdevdn=2, matype=0):
            logger.debug(f"MockTALib: BBANDS({timeperiod}) called")
            if len(series) < timeperiod:
                nan_array = np.full(len(series), np.nan)
                return nan_array, nan_array, nan_array
            # Mock BBands: middle = SMA, upper/lower = +/- 2% of middle
            middle = pd.Series(series).rolling(window=timeperiod).mean().to_numpy()
            upper = middle * 1.02
            lower = middle * 0.98
            return upper, middle, lower

    talib = MockTALib() # Assign mock object if real talib failed
    # --- End Mock TA-Lib ---

# Attempt to import TensorFlow (optional, requires installation)
try:
    # Check if TF is installed before importing heavy modules
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
# indicators.py might still be useful for the evaluation logic, but calculations use talib (real or mock)
from bot.indicators import evaluate_indicators
from bot.sentiment import get_sentiment

# --- Basic Configuration ---
# Load .env file if it exists (for local development)
load_dotenv()

# --- Mock LSTM Implementation (if TensorFlow is not available) ---
if not TF_AVAILABLE:
    class MockLSTMModel:
        def __init__(self, name="MockLSTM"):
            self.name = name
            logger.info(f"Initialized {self.name}")

        def fit(self, X, y, epochs=5, batch_size=32, verbose=0):
            logger.info(f"{self.name}: Mock training with {len(X)} samples...")
            time.sleep(0.1) # Simulate training time

        def predict(self, X, verbose=0):
            logger.info(f"{self.name}: Mock predicting...")
            # Simple mock: predict a slight increase from the last price in the sequence
            last_price_scaled = X[0, -1, 0] # Assuming price is the first feature
            prediction = np.array([[last_price_scaled * 1.001]]) # Predict 0.1% increase
            return prediction

        def build_lstm_model(self, input_shape):
            logger.info(f"{self.name}: Mock model build requested with input shape {input_shape}. No actual build needed.")
            return self # Return self for chaining
# --- End Mock LSTM ---

class ScalpingBot:
    """A scalping bot for Binance using technical indicators, sentiment, and optional LSTM prediction."""

    def __init__(self, config_path="config.yaml", db_dir="database"):
        self.config_path = config_path
        self.db_dir = db_dir
        self.config = {}
        self.api_key = None
        self.api_secret = None
        self.client = None
        self.twm = None
        self.ws_kline_key = None
        self.ws_user_key = None # Added for user stream
        self.is_websocket_running = False
        self.symbol = "BTCUSDT"
        self.position_size = 0.001
        self.max_orders_per_minute = 90 # Safety margin below 100
        self.order_count_minute = 0
        self.last_minute_reset = time.time()
        self.initial_capital = 10000 # Default, updated from Binance if possible
        self.current_capital = self.initial_capital
        self.max_daily_loss_pct = 0.6
        self.max_daily_loss_usd = 0
        self.daily_pnl = 0
        self.last_daily_reset = datetime.now().date()
        self.stop_trading_today = False
        self.db_conn_trades = None
        self.db_conn_keys = None
        self.apprise = None
        self.prices: List[Tuple[int, float, float]] = [] # Stores tuples of (timestamp_ms, price, volume)
        self.scaler_price = MinMaxScaler()
        self.scaler_volume = MinMaxScaler()
        self.model = None
        self.lstm_look_back = 10
        self.lstm_train_interval_seconds = 3600
        self.best_indicator = "ema" # Default
        self.sentiment_score = 0.0
        self.last_indicator_update = datetime.min # Ensure update on first run
        self.last_sentiment_update = time.time()
        self.last_lstm_train = time.time()
        self.sentiment_update_interval_seconds = 900
        self.indicator_update_interval_hours = 24
        self.stop_loss_pct = 0.005
        self.take_profit_pct = 0.01

        # --- Initialization Steps ---
        if not self._load_config(): exit(1)
        if not self._setup_databases(): exit(1)
        self._setup_notifications()
        # Client/Model/Websocket are initialized in the run loop after keys are confirmed

    def _load_config(self) -> bool:
        """Loads configuration from the YAML file."""
        try:
            with open(self.config_path, "r") as file:
                self.config = yaml.safe_load(file)

            # Binance settings
            binance_cfg = self.config.get("binance", {})
            self.symbol = binance_cfg.get("symbol", "BTCUSDT")
            self.position_size = float(binance_cfg.get("position_size", 0.001))
            self.is_testnet = binance_cfg.get("testnet", True)

            # Risk settings
            risk_cfg = self.config.get("risk", {})
            self.max_daily_loss_pct = float(risk_cfg.get("max_daily_loss", 0.6))
            self.stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.005))
            self.take_profit_pct = float(risk_cfg.get("take_profit_pct", 0.01))
            self.max_daily_loss_usd = self.initial_capital * self.max_daily_loss_pct # Initial calculation

            # LSTM settings
            lstm_cfg = self.config.get("lstm", {})
            self.lstm_look_back = int(lstm_cfg.get("look_back", 10))
            self.lstm_train_interval_seconds = int(lstm_cfg.get("train_interval_seconds", 3600))

            # Update intervals
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
        """Initializes SQLite databases for API keys and trades/reports."""
        try:
            os.makedirs(self.db_dir, exist_ok=True)
            keys_db_path = os.path.join(self.db_dir, "api_keys.db")
            trades_db_path = os.path.join(self.db_dir, "trades.db")

            # Connect and create API keys table
            self.db_conn_keys = sqlite3.connect(keys_db_path, check_same_thread=False)
            cursor_keys = self.db_conn_keys.cursor()
            cursor_keys.execute("CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY, api_key TEXT, api_secret TEXT)")
            self.db_conn_keys.commit()

            # Connect and create trades/reports tables
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
            logger.info("Databases initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"CRITICAL: Error setting up databases in {self.db_dir}: {e}", exc_info=True)
            return False

    def _setup_notifications(self):
        """Configures Apprise for Telegram notifications."""
        try:
            self.apprise = apprise.Apprise()
            tg_token = self.config.get("telegram", {}).get("bot_token", "YOUR_TELEGRAM_BOT_TOKEN")
            tg_chat_id = self.config.get("telegram", {}).get("chat_id", "YOUR_TELEGRAM_CHAT_ID")

            # Check if placeholders are still present
            if tg_token and "YOUR_TELEGRAM" not in tg_token and tg_chat_id and "YOUR_TELEGRAM" not in str(tg_chat_id):
                self.apprise.add(f"tgram://{tg_token}/{tg_chat_id}")
                logger.info("Telegram notifications configured.")
                self.notify("Trading bot starting up...")
            else:
                logger.warning("Telegram token/chat ID not configured or using placeholders. Telegram notifications disabled.")
        except Exception as e:
            logger.error(f"Error setting up Apprise notifications: {e}", exc_info=True)

    def notify(self, message: str):
        """Sends a notification if Apprise is configured."""
        logger.info(f"Notification: {message}") # Log all notifications
        if self.apprise and self.apprise.servers:
            try:
                # Corrected: Use standard string formatting, avoid unnecessary backslashes
                self.apprise.notify(body=message)
            except Exception as e:
                logger.error(f"Failed to send notification via Apprise: {e}")

    def _load_api_keys(self) -> bool:
        """Loads API keys from the database."""
        if self.api_key and self.api_secret: # Already loaded
            return True
        try:
            cursor = self.db_conn_keys.cursor()
            cursor.execute("SELECT api_key, api_secret FROM api_keys WHERE id = 1")
            result = cursor.fetchone()
            if result and result[0] and result[1]:
                self.api_key = result[0]
                self.api_secret = result[1]
                logger.info("API keys loaded successfully from database.")
                return True
            else:
                logger.warning("API keys not found in database. Please add them via the panel.")
                return False
        except Exception as e:
            logger.error(f"Error loading API keys from database: {e}", exc_info=True)
            return False

    def _initialize_client(self) -> bool:
        """Initializes the Binance client if keys are available."""
        if not (self.api_key and self.api_secret):
            logger.warning("Cannot initialize Binance client: API keys missing.")
            return False
        if self.client: # Already initialized
            return True

        try:
            logger.info(f"Initializing Binance client (Testnet: {self.is_testnet})...")
            self.client = Client(self.api_key, self.api_secret, testnet=self.is_testnet)
            self.client.ping() # Test connection
            logger.info("Binance client ping successful.")

            # Fetch account info and update capital
            account_info = self.client.get_account()
            usdt_balance = 0.0
            for balance in account_info.get("balances", []):
                if balance["asset"] == "USDT":
                    usdt_balance = float(balance["free"])
                    break

            if usdt_balance > 0:
                self.current_capital = usdt_balance
                # Reset initial capital based on current balance at startup
                self.initial_capital = self.current_capital
                self.max_daily_loss_usd = self.initial_capital * self.max_daily_loss_pct
                logger.info(f"Connected to Binance. Initial USDT balance: {self.current_capital:.2f}. Max daily loss set to: {self.max_daily_loss_usd:.2f} USDT")
            else:
                 logger.warning("Could not retrieve USDT balance from Binance account.")
                 # Keep default initial capital and max loss

            return True
        except BinanceAPIException as e:
            logger.error(f"Binance API Error during client initialization: {e}. Check API keys, permissions, and network.")
            self.notify(f"CRITICAL: Binance API Error: {e}. Bot cannot connect.")
            self.api_key = None # Invalidate keys on connection failure
            self.api_secret = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing Binance client: {e}", exc_info=True)
            return False

    def _build_lstm_model(self):
        """Builds the LSTM model (real or mock)."""
        if self.model: # Already built
            return

        input_shape = (self.lstm_look_back, 2) # 2 features: price, volume

        if TF_AVAILABLE:
            try:
                logger.info(f"Building TensorFlow LSTM model with input shape {input_shape}...")
                self.model = Sequential(name="TensorFlowLSTM")
                self.model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
                self.model.add(LSTM(50, return_sequences=False))
                # Corrected line 314 - Using double quotes for strings
                self.model.add(Dense(25, activation="relu"))
                self.model.add(Dense(1))
                # Corrected line 316 - Using double quotes for strings
                self.model.compile(optimizer="adam", loss="mse")
                logger.info("TensorFlow LSTM model built successfully.")
            except Exception as e:
                logger.error(f"Error building TensorFlow LSTM model: {e}", exc_info=True)
                self.notify("Error building LSTM model. Check logs.")
                self.model = None # Ensure model is None if build fails
                TF_AVAILABLE = False # Disable TF if build fails
                self._build_lstm_model() # Fallback to mock model
        else:
            logger.info("Building Mock LSTM model...")
            self.model = MockLSTMModel().build_lstm_model(input_shape)
            logger.info("Mock LSTM model built.")

    def _train_lstm_model(self):
        """Trains the LSTM model (real or mock) if enough data is available."""
        if not self.model or len(self.prices) < self.lstm_look_back + 1:
            logger.debug("Not enough data or no model to train LSTM.")
            return

        logger.info("Preparing data for LSTM training...")
        # Extract prices and volumes, ensuring they are numpy arrays
        timestamps = np.array([p[0] for p in self.prices])
        prices_arr = np.array([p[1] for p in self.prices]).reshape(-1, 1)
        volumes_arr = np.array([p[2] for p in self.prices]).reshape(-1, 1)

        # Scale data
        try:
            scaled_prices = self.scaler_price.fit_transform(prices_arr)
            scaled_volumes = self.scaler_volume.fit_transform(volumes_arr)
        except ValueError as e:
            logger.error(f"Error scaling data for LSTM (ValueError): {e}. Skipping training.", exc_info=True)
            return
        except Exception as e:
            logger.error(f"Unexpected error scaling data for LSTM: {e}. Skipping training.", exc_info=True)
            return

        # Create sequences
        X, y = [], []
        for i in range(self.lstm_look_back, len(scaled_prices)):
            X.append(np.hstack((scaled_prices[i-self.lstm_look_back:i], scaled_volumes[i-self.lstm_look_back:i])))
            y.append(scaled_prices[i])
        X, y = np.array(X), np.array(y)

        if X.shape[0] == 0:
            logger.warning("No sequences generated for LSTM training. Skipping.")
            return

        logger.info(f"Training LSTM model with {X.shape[0]} sequences...")
        try:
            self.model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            self.last_lstm_train = time.time()
            logger.info("LSTM model training completed.")
        except Exception as e:
            logger.error(f"Error during LSTM model training: {e}", exc_info=True)
            self.notify("Error during LSTM training. Check logs.")

    def _predict_price_lstm(self) -> Optional[float]:
        """Predicts the next price using the LSTM model (real or mock)."""
        if not self.model or len(self.prices) < self.lstm_look_back:
            logger.debug("Not enough data or no model for LSTM prediction.")
            return None

        logger.debug("Preparing data for LSTM prediction...")
        # Extract last sequence
        last_sequence_prices = np.array([p[1] for p in self.prices[-self.lstm_look_back:]]).reshape(-1, 1)
        last_sequence_volumes = np.array([p[2] for p in self.prices[-self.lstm_look_back:]]).reshape(-1, 1)

        # Scale data using the *already fitted* scalers
        try:
            scaled_prices = self.scaler_price.transform(last_sequence_prices)
            scaled_volumes = self.scaler_volume.transform(last_sequence_volumes)
        except NotFittedError:
            logger.warning("Scalers not fitted yet. Skipping LSTM prediction.")
            return None
        except ValueError as e:
            logger.error(f"Error scaling data for LSTM prediction (ValueError): {e}. Skipping prediction.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error scaling data for LSTM prediction: {e}. Skipping prediction.", exc_info=True)
            return None

        X_pred = np.array([np.hstack((scaled_prices, scaled_volumes))])

        logger.debug("Predicting next price with LSTM...")
        try:
            predicted_scaled = self.model.predict(X_pred, verbose=0)
            # Inverse transform the prediction
            predicted_price = self.scaler_price.inverse_transform(predicted_scaled)[0][0]
            logger.debug(f"LSTM Predicted Price: {predicted_price:.2f}")
            return float(predicted_price)
        except NotFittedError: # Should not happen if transform worked, but for safety
            logger.warning("Price scaler not fitted for inverse transform. Skipping LSTM prediction.")
            return None
        except Exception as e:
            logger.error(f"Error during LSTM prediction or inverse transform: {e}", exc_info=True)
            return None

    def _get_historical_data(self, interval="1m", limit=10080) -> Optional[pd.DataFrame]:
        """Fetches historical klines from Binance."""
        if not self.client:
            logger.error("Cannot fetch historical data: Binance client not initialized.")
            return None
        try:
            logger.info(f"Fetching historical data: {self.symbol}, interval={interval}, limit={limit}")
            # Calculate start time string for Binance API
            start_str = (datetime.utcnow() - timedelta(minutes=limit)).strftime("%Y-%m-%d %H:%M:%S")

            klines = self.client.get_historical_klines(self.symbol, interval, f"{limit} minutes ago UTC")
            if not klines:
                logger.warning("No historical klines received from Binance.")
                return None

            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            logger.info(f"Fetched {len(df)} historical klines.")
            return df
        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching historical data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data: {e}", exc_info=True)
            return None

    def _check_order_limit(self) -> bool:
        """Checks if the bot is within the orders-per-minute limit."""
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            logger.debug(f"Resetting minute order count (was {self.order_count_minute}).")
            self.order_count_minute = 0
            self.last_minute_reset = current_time
        return self.order_count_minute < self.max_orders_per_minute

    def _check_risk_limit(self, potential_loss: float) -> bool:
        """Checks if a potential loss exceeds the daily limit."""
        # Check if day has changed
        current_date = datetime.now().date()
        if current_date > self.last_daily_reset:
            logger.info(f"New day detected ({current_date}). Resetting daily PnL and trading status.")
            self._generate_daily_report() # Generate report for the previous day
            self.daily_pnl = 0
            self.stop_trading_today = False
            self.last_daily_reset = current_date
            # Recalculate max loss based on potentially updated capital
            self.max_daily_loss_usd = self.current_capital * self.max_daily_loss_pct
            logger.info(f"Max daily loss for {current_date} set to: {self.max_daily_loss_usd:.2f} USDT")

        if self.stop_trading_today:
            logger.warning("Trading stopped for today due to reaching max daily loss.")
            return False

        # Check potential loss against remaining allowance
        # Note: daily_pnl is negative for losses
        if self.daily_pnl - potential_loss < -self.max_daily_loss_usd:
            logger.critical(f"Potential loss ({potential_loss:.2f}) exceeds remaining daily limit ({self.max_daily_loss_usd + self.daily_pnl:.2f}). Max loss: {self.max_daily_loss_usd:.2f}, Current daily PnL: {self.daily_pnl:.2f}")
            self.notify(f"CRITICAL: Max daily loss limit reached ({self.max_daily_loss_usd:.2f} USDT). Stopping trading for today.")
            self.stop_trading_today = True
            # Consider canceling open orders here if necessary
            # self._cancel_all_open_orders()
            return False
        return True

    def _start_websocket(self) -> bool:
        """Starts the Binance Kline and User Data websockets."""
        if not self.client:
            logger.error("Cannot start websocket: Binance client not initialized.")
            return False
        if self.is_websocket_running:
            logger.warning("Websocket is already running.")
            return True

        try:
            self.twm = ThreadedWebsocketManager(self.api_key, self.api_secret, testnet=self.is_testnet)
            self.twm.start()

            # --- Kline WebSocket --- (Handles price updates)
            kline_stream_name = f"{self.symbol.lower()}@kline_1m"
            self.ws_kline_key = self.twm.start_symbol_miniticker_socket(self._handle_kline_message, symbol=self.symbol)
            # self.ws_kline_key = self.twm.start_kline_socket(self._handle_kline_message, symbol=self.symbol, interval=KLINE_INTERVAL_1MINUTE)
            if self.ws_kline_key:
                logger.info(f"Started Kline WebSocket for {self.symbol}.")
            else:
                logger.error("Failed to start Kline WebSocket.")
                self.twm.stop()
                return False

            # --- User Data WebSocket --- (Handles order updates, balance changes)
            self.ws_user_key = self.twm.start_user_socket(self._handle_user_message)
            if self.ws_user_key:
                logger.info("Started User Data WebSocket.")
            else:
                logger.error("Failed to start User Data WebSocket. Order status updates will be missed!")
                # Continue running with only kline data, but log a warning

            self.is_websocket_running = True
            logger.info("Websocket manager started.")
            # self.twm.join() # Don't join here, let it run in background thread
            return True

        except Exception as e:
            logger.error(f"Error starting WebSocket manager: {e}", exc_info=True)
            if self.twm:
                try: self.twm.stop() # Attempt cleanup
                except: pass
            self.is_websocket_running = False
            return False

    def _stop_websocket(self):
        """Stops the Binance websockets gracefully."""
        if self.twm and self.is_websocket_running:
            logger.info("Stopping WebSocket manager...")
            try:
                if self.ws_kline_key:
                    self.twm.stop_socket(self.ws_kline_key)
                    self.ws_kline_key = None
                if self.ws_user_key:
                    self.twm.stop_socket(self.ws_user_key)
                    self.ws_user_key = None
                self.twm.stop()
                self.is_websocket_running = False
                logger.info("WebSocket manager stopped.")
            except Exception as e:
                logger.error(f"Error stopping WebSocket manager: {e}", exc_info=True)
        else:
            logger.info("WebSocket manager not running or not initialized.")

    def _handle_kline_message(self, msg: Dict[str, Any]):
        """Handles incoming Kline/Ticker messages from the WebSocket."""
        # Using miniticker stream ('c', 'v', 'E')
        if msg and 'stream' in msg and 'data' in msg:
            data = msg['data']
            if data.get('e') == '24hrMiniTicker':
                try:
                    timestamp_ms = int(data['E'])
                    price = float(data['c']) # Last price
                    volume = float(data['v']) # Total traded base asset volume

                    # Append data: (timestamp_ms, price, volume)
                    self.prices.append((timestamp_ms, price, volume))

                    # Keep price list from growing indefinitely
                    max_prices = 2000 # Keep roughly last ~33 hours of 1m data + buffer
                    if len(self.prices) > max_prices:
                        self.prices = self.prices[-max_prices:]

                    # --- Trigger Actions Based on New Price --- 
                    # 1. Train LSTM periodically
                    if TF_AVAILABLE and (time.time() - self.last_lstm_train > self.lstm_train_interval_seconds):
                        self._train_lstm_model()

                    # 2. Update sentiment periodically
                    if time.time() - self.last_sentiment_update > self.sentiment_update_interval_seconds:
                        self._update_sentiment()

                    # 3. Make trading decision
                    self._make_decision(price, volume, timestamp_ms)

                except (KeyError, ValueError, TypeError) as e:
                    logger.error(f"Error processing kline/ticker message: {e}. Message: {msg}")
                except Exception as e:
                    logger.error(f"Unexpected error in _handle_kline_message: {e}. Message: {msg}", exc_info=True)
        else:
            logger.warning(f"Received unexpected message format on kline stream: {msg}")

    def _handle_user_message(self, msg: Dict[str, Any]):
        """Handles incoming User Data messages from the WebSocket."""
        if msg and 'e' in msg: # Check if 'e' (event type) exists
            event_type = msg['e']
            try:
                if event_type == 'executionReport':
                    order_id = msg.get('c') # Use client order ID if available, else Binance order ID 'i'
                    if not order_id:
                        order_id = str(msg.get('i'))
                    order_status = msg.get('X')
                    side = msg.get('S')
                    price = float(msg.get('p')) # Order price
                    quantity = float(msg.get('q')) # Order quantity
                    filled_qty = float(msg.get('z')) # Cumulative filled quantity
                    last_filled_qty = float(msg.get('l')) # Last filled quantity
                    last_filled_price = float(msg.get('L')) # Last filled price
                    commission = float(msg.get('n') or 0.0) # Commission amount
                    commission_asset = msg.get('N') # Commission asset
                    trade_time = int(msg.get('T')) # Trade time (timestamp ms)
                    pnl = float(msg.get('rp', 0.0)) # Realized profit of the trade (for closing trades)

                    logger.info(f"User Event: Order Update - ID: {order_id}, Status: {order_status}, Side: {side}, Filled: {filled_qty}/{quantity} @ {last_filled_price}, PnL: {pnl}")

                    # --- Update Trade Record in DB --- 
                    if order_status in ['FILLED', 'PARTIALLY_FILLED'] and last_filled_qty > 0:
                        # If it's a closing trade (opposite side of an open one), calculate PnL
                        # This requires tracking open positions, which is complex. Using 'rp' if available.
                        self._update_trade_status(order_id, order_status, realized_pnl=pnl)
                        if pnl != 0:
                            self.daily_pnl += pnl
                            self.current_capital += pnl # Update capital based on realized PnL
                            logger.info(f"Realized PnL: {pnl:.2f}. Current Daily PnL: {self.daily_pnl:.2f}. Updated Capital: {self.current_capital:.2f}")

                    elif order_status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                        self._update_trade_status(order_id, order_status)

                elif event_type == 'outboundAccountPosition': # Balance update
                    # This provides updates on asset balances
                    balances = msg.get('B', [])
                    for balance in balances:
                        if balance['a'] == 'USDT':
                            new_capital = float(balance['f']) # Free balance
                            if abs(new_capital - self.current_capital) > 0.01: # Log if changed significantly
                                logger.info(f"User Event: Balance Update - USDT Free: {new_capital:.2f} (Previous: {self.current_capital:.2f})")
                                self.current_capital = new_capital
                                # Recalculate max loss based on new capital
                                self.max_daily_loss_usd = self.current_capital * self.max_daily_loss_pct
                            break
                # Add handling for other user events if needed (e.g., 'balanceUpdate')

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Error processing user message ({event_type}): {e}. Message: {msg}")
            except Exception as e:
                logger.error(f"Unexpected error in _handle_user_message ({event_type}): {e}. Message: {msg}", exc_info=True)
        else:
            logger.warning(f"Received unexpected message format on user stream: {msg}")

    def _update_trade_status(self, order_id: str, status: str, realized_pnl: float = 0.0):
        """Updates the status and PnL of a trade in the database."""
        try:
            cursor = self.db_conn_trades.cursor()
            cursor.execute(
                "UPDATE trades SET status = ?, pnl = pnl + ? WHERE order_id = ?",
                (status, realized_pnl, order_id)
            )
            if cursor.rowcount > 0:
                self.db_conn_trades.commit()
                logger.info(f"Updated trade {order_id} status to {status} with PnL {realized_pnl:.2f} in DB.")
            else:
                logger.warning(f"Could not find trade with order_id {order_id} to update status.")
        except Exception as e:
            logger.error(f"Error updating trade status for order {order_id} in DB: {e}", exc_info=True)

    def _update_sentiment(self):
        """Fetches and updates the sentiment score."""
        logger.info("Updating sentiment score...")
        try:
            # Pass the whole config, get_sentiment will extract relevant parts
            new_sentiment = get_sentiment(self.config)
            if new_sentiment is not None:
                self.sentiment_score = new_sentiment
                logger.info(f"Sentiment score updated: {self.sentiment_score:.2f}")
            else:
                logger.warning("Failed to update sentiment score.")
            self.last_sentiment_update = time.time()
        except Exception as e:
            logger.error(f"Error during sentiment update: {e}", exc_info=True)

    def _update_indicators_evaluation(self):
        """Fetches historical data and evaluates indicators to find the best one."""
        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available, cannot evaluate indicators. Using default: {self.best_indicator}")
            return

        logger.info("Updating best indicator evaluation...")
        # Fetch enough data for evaluation (e.g., 1 week = 10080 minutes)
        df = self._get_historical_data(interval="1m", limit=10080)
        if df is None or df.empty:
            logger.error("Failed to get historical data for indicator evaluation.")
            return

        try:
            new_best_indicator = evaluate_indicators(df)
            if new_best_indicator:
                if new_best_indicator != self.best_indicator:
                    logger.info(f"Best indicator changed from '{self.best_indicator}' to '{new_best_indicator}'.")
                    self.notify(f"Indicator Strategy Update: Switched to '{new_best_indicator}'.")
                    self.best_indicator = new_best_indicator
                else:
                    logger.info(f"Best indicator remains '{self.best_indicator}'.")
            else:
                logger.warning("Indicator evaluation did not return a best indicator. Keeping current: '{self.best_indicator}'.")
            self.last_indicator_update = datetime.now()
        except Exception as e:
            logger.error(f"Error during indicator evaluation: {e}", exc_info=True)
            self.notify("Error evaluating indicators. Check logs.")

    def _make_decision(self, current_price: float, volume: float, timestamp_ms: int):
        """Analyzes data and decides whether to place a trade."""
        # --- Pre-Trade Checks ---
        if self.stop_trading_today:
            return # Daily loss limit reached

        if not self._check_order_limit():
            logger.warning("Order limit reached for this minute. Skipping decision.")
            return

        if len(self.prices) < 21: # Need enough data for longest indicator (e.g., EMA 21)
            logger.debug("Not enough price data yet to make decision.")
            return

        # --- Calculate Indicators ---
        # Use only the necessary recent data
        recent_prices = np.array([p[1] for p in self.prices[-50:]]) # Get last 50 prices
        recent_volumes = np.array([p[2] for p in self.prices[-50:]]) # Get last 50 volumes

        try:
            ema9 = talib.EMA(recent_prices, timeperiod=9)[-1]
            ema21 = talib.EMA(recent_prices, timeperiod=21)[-1]
            rsi = talib.RSI(recent_prices, timeperiod=14)[-1]
            upper_bb, middle_bb, lower_bb = talib.BBANDS(recent_prices, timeperiod=20)
            upper_bb, lower_bb = upper_bb[-1], lower_bb[-1]
            # VWAP calculation (approximation over recent period)
            vwap = np.sum(recent_prices * recent_volumes) / np.sum(recent_volumes) if np.sum(recent_volumes) > 0 else current_price

            # Check for NaN values (can happen at the start)
            if any(np.isnan([ema9, ema21, rsi, upper_bb, lower_bb, vwap])):
                logger.debug("Indicator calculation resulted in NaN. Skipping decision.")
                return

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return

        # --- Get LSTM Prediction (Optional) ---
        predicted_price = self._predict_price_lstm()
        # If LSTM prediction fails or is disabled, use a neutral prediction (current price)
        if predicted_price is None:
            predicted_price = current_price
            logger.debug("Using current price as neutral prediction (LSTM failed or disabled).")

        # --- Calculate Stop Loss and Take Profit ---
        buy_stop_loss = current_price * (1 - self.stop_loss_pct)
        buy_take_profit = current_price * (1 + self.take_profit_pct)
        sell_stop_loss = current_price * (1 + self.stop_loss_pct)
        sell_take_profit = current_price * (1 - self.take_profit_pct)

        # --- Risk Check for potential BUY ---
        potential_buy_loss = self.position_size * (current_price - buy_stop_loss)
        can_buy = self._check_risk_limit(potential_buy_loss)

        # --- Risk Check for potential SELL (assuming shorting or closing a long) ---
        # For simplicity, assume symmetrical risk for sell signal (though actual shorting risk differs)
        potential_sell_loss = self.position_size * (sell_stop_loss - current_price)
        can_sell = self._check_risk_limit(potential_sell_loss)

        # --- Define Buy/Sell Signals based on Best Indicator ---
        should_buy = False
        should_sell = False

        # Sentiment modifier: scales the expected price move slightly
        # Positive sentiment boosts buy signals, negative sentiment boosts sell signals
        sentiment_modifier = 1 + (self.sentiment_score * 0.05) # Max +/- 5% influence

        logger.debug(f"Decision Data: Price={current_price:.2f}, EMA9={ema9:.2f}, EMA21={ema21:.2f}, RSI={rsi:.1f}, BB={lower_bb:.2f}-{upper_bb:.2f}, VWAP={vwap:.2f}, LSTM_Pred={predicted_price:.2f}, Sentiment={self.sentiment_score:.2f}, BestInd='{self.best_indicator}'")

        # Apply strategy based on the best indicator found by evaluation
        if self.best_indicator == "ema":
            if ema9 > ema21 and predicted_price > current_price * sentiment_modifier:
                should_buy = True
            elif ema9 < ema21 and predicted_price < current_price / sentiment_modifier:
                should_sell = True
        elif self.best_indicator == "rsi":
            if rsi < 35 and predicted_price > current_price * sentiment_modifier: # Adjusted threshold
                should_buy = True
            elif rsi > 65 and predicted_price < current_price / sentiment_modifier: # Adjusted threshold
                should_sell = True
        elif self.best_indicator == "bb":
            if current_price <= lower_bb and predicted_price > current_price * sentiment_modifier:
                should_buy = True
            elif current_price >= upper_bb and predicted_price < current_price / sentiment_modifier:
                should_sell = True
        elif self.best_indicator == "vwap":
            if current_price < vwap and predicted_price > current_price * sentiment_modifier:
                should_buy = True
            elif current_price > vwap and predicted_price < current_price / sentiment_modifier:
                should_sell = True
        else: # Default to EMA if best_indicator is somehow invalid
             if ema9 > ema21 and predicted_price > current_price * sentiment_modifier:
                should_buy = True
             elif ema9 < ema21 and predicted_price < current_price / sentiment_modifier:
                should_sell = True

        # --- Execute Trade --- 
        # Basic logic: only enter if not already in a position (requires position tracking)
        # For simplicity here, we just execute if signal is present and risk allows
        # WARNING: This simple version doesn't prevent multiple simultaneous positions!
        if should_buy and can_buy:
            logger.info("BUY signal triggered.")
            self._execute_trade(SIDE_BUY, current_price, buy_stop_loss, buy_take_profit)
        elif should_sell and can_sell:
            logger.info("SELL signal triggered.")
            self._execute_trade(SIDE_SELL, current_price, sell_stop_loss, sell_take_profit)

    def _execute_trade(self, side: str, price: float, stop_loss: float, take_profit: float):
        """Executes a market order on Binance and logs it."""
        if not self.client:
            logger.error("Cannot execute trade: Binance client not initialized.")
            return

        order_side = SIDE_BUY if side == "BUY" else SIDE_SELL
        # Use client order ID for better tracking
        client_order_id = f"manus_{self.symbol}_{int(time.time() * 1000)}"

        try:
            logger.info(f"Attempting to place {side} MARKET order for {self.position_size} {self.symbol} (ID: {client_order_id})...")
            # Using MARKET order for simplicity to ensure execution
            # LIMIT orders might be better for scalping but add complexity (checking fills)
            order = self.client.create_order(
                symbol=self.symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.position_size,
                newClientOrderId=client_order_id
                # For MARKET orders, price is not specified
            )
            self.order_count_minute += 1
            logger.info(f"MARKET Order placed successfully: {order}")

            # --- Log Order to Database --- 
            # We log the order intent here. Status/PnL updated by websocket executionReport.
            # Use the actual execution price if available in response, else the intended price (approx)
            executed_price = price # Placeholder, ideally from order response or websocket
            if order and 'fills' in order and len(order['fills']) > 0:
                 # Average execution price from fills
                 executed_price = sum(float(f['price']) * float(f['qty']) for f in order['fills']) / sum(float(f['qty']) for f in order['fills'])
                 logger.info(f"Order executed at average price: {executed_price:.2f}")
            elif order and 'price' in order and float(order['price']) > 0:
                 executed_price = float(order['price'])

            binance_order_id = str(order.get('orderId', 'N/A'))

            cursor = self.db_conn_trades.cursor()
            cursor.execute(
                "INSERT INTO trades (timestamp, symbol, side, price, quantity, order_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), self.symbol, side, executed_price, self.position_size, binance_order_id, order.get('status', 'SUBMITTED'))
            )
            self.db_conn_trades.commit()
            logger.info(f"Logged {side} order {binance_order_id} to database.")

            self.notify(f"Trade Executed: {side} {self.position_size} {self.symbol} @ ~{executed_price:.2f} (Order ID: {binance_order_id})")

            # --- OCO Order for Stop-Loss and Take-Profit --- 
            # Place OCO immediately after market order confirmation
            # Note: OCO requires precise quantity and opposite side
            oco_side = SIDE_SELL if order_side == SIDE_BUY else SIDE_BUY
            oco_stop_price = round(stop_loss, 2) # Adjust precision as needed for the symbol
            oco_limit_price = round(take_profit, 2) # Adjust precision
            # Stop price must be below market for SELL OCO, above for BUY OCO
            # Limit price must be above market for SELL OCO, below for BUY OCO
            # Ensure correct price ordering for OCO
            if oco_side == SIDE_SELL:
                oco_stop_limit_price = oco_stop_price # Price at which the stop-limit order triggers
                if oco_stop_price >= executed_price:
                    oco_stop_price = round(executed_price * (1 - self.stop_loss_pct * 1.1), 2) # Adjust slightly lower
                    oco_stop_limit_price = oco_stop_price
                    logger.warning(f"Adjusted OCO SELL stop price lower: {oco_stop_price}")
                if oco_limit_price <= executed_price:
                     oco_limit_price = round(executed_price * (1 + self.take_profit_pct * 1.1), 2) # Adjust slightly higher
                     logger.warning(f"Adjusted OCO SELL limit price higher: {oco_limit_price}")
            else: # BUY OCO (closing a short)
                oco_stop_limit_price = oco_stop_price
                if oco_stop_price <= executed_price:
                    oco_stop_price = round(executed_price * (1 + self.stop_loss_pct * 1.1), 2) # Adjust slightly higher
                    oco_stop_limit_price = oco_stop_price
                    logger.warning(f"Adjusted OCO BUY stop price higher: {oco_stop_price}")
                if oco_limit_price >= executed_price:
                     oco_limit_price = round(executed_price * (1 - self.take_profit_pct * 1.1), 2) # Adjust slightly lower
                     logger.warning(f"Adjusted OCO BUY limit price lower: {oco_limit_price}")

            try:
                logger.info(f"Placing OCO {oco_side} order: Qty={self.position_size}, Limit={oco_limit_price}, Stop={oco_stop_price}, StopLimit={oco_stop_limit_price}")
                oco_order = self.client.create_oco_order(
                    symbol=self.symbol,
                    side=oco_side,
                    quantity=self.position_size,
                    price=str(oco_limit_price), # Take Profit Limit Price
                    stopPrice=str(oco_stop_price), # Stop Loss Trigger Price
                    stopLimitPrice=str(oco_stop_limit_price), # Stop Loss Limit Price (can be same as stopPrice or slightly worse)
                    stopLimitTimeInForce=TIME_IN_FORCE_GTC
                )
                logger.info(f"OCO Order placed successfully: {oco_order}")
                # We don't typically log OCO orders separately in the 'trades' table unless needed
            except BinanceAPIException as e:
                logger.error(f"Failed to place OCO order for {side} trade {binance_order_id}: {e}")
                self.notify(f"CRITICAL: Failed to place OCO order for {binance_order_id}. Manual intervention may be needed.")
            except Exception as e:
                 logger.error(f"Unexpected error placing OCO order: {e}", exc_info=True)

        except BinanceAPIException as e:
            logger.error(f"Binance API Error executing {side} order: {e}")
            self.notify(f"ERROR: Failed to execute {side} order: {e}")
            # Log failed attempt?
        except Exception as e:
            logger.error(f"Unexpected error executing {side} order: {e}", exc_info=True)
            self.notify(f"ERROR: Unexpected error executing {side} order. Check logs.")

    def _generate_daily_report(self):
        """Generates and saves a daily performance report."""
        report_date = self.last_daily_reset # Report for the day that just ended
        report_date_str = report_date.strftime("%Y-%m-%d")
        logger.info(f"Generating daily report for {report_date_str}...")

        try:
            cursor = self.db_conn_trades.cursor()
            # Select trades closed on the report date
            # Assuming status 'FILLED' for the closing part of the trade means closed
            cursor.execute(
                "SELECT pnl FROM trades WHERE status = 'FILLED' AND date(timestamp) = ? AND pnl != 0",
                (report_date_str,)
            )
            closed_trades_pnl = cursor.fetchall()

            total_pnl = sum(trade[0] for trade in closed_trades_pnl)
            trades_count = len(closed_trades_pnl)
            win_trades = len([t for t in closed_trades_pnl if t[0] > 0])
            win_rate = (win_trades / trades_count * 100) if trades_count > 0 else 0
            avg_pnl_per_trade = total_pnl / trades_count if trades_count > 0 else 0

            # Get the best indicator used during that day (simplistic: last used)
            # A better approach would be to log indicator changes with timestamps
            current_best_indicator = self.best_indicator

            # Insert or update the report in the database
            cursor.execute("""
                INSERT INTO daily_reports (date, total_pnl, trades_count, win_rate, avg_pnl_per_trade, best_indicator)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_pnl = excluded.total_pnl,
                    trades_count = excluded.trades_count,
                    win_rate = excluded.win_rate,
                    avg_pnl_per_trade = excluded.avg_pnl_per_trade,
                    best_indicator = excluded.best_indicator
            """, (report_date_str, total_pnl, trades_count, win_rate, avg_pnl_per_trade, current_best_indicator))
            self.db_conn_trades.commit()

            logger.info(f"Daily report for {report_date_str} generated and saved.")

            # Notify with the report summary
            report_summary = (
                f"--- Daily Report {report_date_str} ---\n"
                f"Total PnL: {total_pnl:.2f} USDT\n"
                f"Trades Count: {trades_count}\n"
                f"Win Rate: {win_rate:.2f}%\n"
                f"Avg PnL/Trade: {avg_pnl_per_trade:.2f} USDT\n"
                f"Ending Capital: {self.current_capital:.2f} USDT\n"
                f"Best Indicator Used: {current_best_indicator}"
            )
            self.notify(report_summary)

        except Exception as e:
            logger.error(f"Error generating daily report for {report_date_str}: {e}", exc_info=True)
            self.notify(f"Error generating daily report for {report_date_str}. Check logs.")

    def _setup_schedule(self):
        """Sets up the scheduled tasks for the bot."""
        logger.info("Setting up scheduled tasks...")
        # Schedule indicator evaluation (e.g., daily at a specific time)
        schedule.every(self.indicator_update_interval_hours).hours.do(self._update_indicators_evaluation)
        # Schedule daily report generation (e.g., just after midnight)
        schedule.every().day.at("00:01").do(self._check_risk_limit) # This also triggers report generation

        # Log scheduled jobs
        for job in schedule.get_jobs():
            logger.info(f"Scheduled job: {job}")

    def run(self):
        """Main execution loop for the bot."""
        logger.info("Starting bot run loop...")

        while True:
            # --- Initial Setup / Reconnect Logic --- 
            if not (self.api_key and self.api_secret):
                if not self._load_api_keys():
                    logger.info("API keys not available. Waiting 60 seconds...")
                    time.sleep(60)
                    continue # Retry loading keys

            if not self.client:
                if not self._initialize_client():
                    logger.info("Failed to initialize Binance client. Waiting 60 seconds...")
                    time.sleep(60)
                    continue # Retry initialization

            if not self.model:
                self._build_lstm_model() # Build LSTM (real or mock)

            if not self.is_websocket_running:
                if not self._start_websocket():
                    logger.info("Failed to start WebSocket. Waiting 60 seconds...")
                    time.sleep(60)
                    continue # Retry starting websocket

            # --- Run Scheduled Tasks --- 
            schedule.run_pending()

            # --- Periodic Checks (if not handled by schedule/websocket) ---
            # Example: Force indicator update on first run or if too old
            if (datetime.now() - self.last_indicator_update).total_seconds() > self.indicator_update_interval_hours * 3600 * 1.1: # Add buffer
                 logger.info("Indicator evaluation seems overdue. Triggering update.")
                 self._update_indicators_evaluation()

            # --- Main Loop Sleep --- 
            # Websocket runs in background threads, main loop handles schedule and checks
            time.sleep(10) # Check schedule every 10 seconds

    def stop(self):
        """Stops the bot gracefully."""
        logger.info("Initiating bot shutdown...")
        self._stop_websocket()
        if self.db_conn_trades:
            self.db_conn_trades.close()
            logger.info("Trades database connection closed.")
        if self.db_conn_keys:
            self.db_conn_keys.close()
            logger.info("API keys database connection closed.")
        schedule.clear()
        logger.info("Scheduled tasks cleared.")
        self.notify("Trading bot shutting down.")
        logger.info("Bot shutdown complete.")

# --- Main Execution --- 
if __name__ == "__main__":
    bot = None
    try:
        bot = ScalpingBot()
        bot._setup_schedule() # Setup schedule before starting the loop
        bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping bot...")
    except Exception as e:
        logger.critical(f"CRITICAL UNHANDLED EXCEPTION in main loop: {e}", exc_info=True)
        if bot and bot.apprise:
             bot.notify(f"CRITICAL ERROR: Bot crashed unexpectedly: {e}. Requires restart.")
    finally:
        if bot:
            bot.stop()

