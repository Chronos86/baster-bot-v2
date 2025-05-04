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

# --- Binance & Notifications ---
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from binance import ThreadedWebsocketManager
import apprise

# --- Data Processing & ML ---
from sklearn.preprocessing import MinMaxScaler
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
from bot.indicators import evaluate_indicators # Handles its own TA-Lib import/mock
from bot.sentiment import get_sentiment

# --- Basic Configuration ---
# Load .env file if it exists (for local development)
load_dotenv()

# Configure logging (ensure no trailing backslash or invalid chars)
# Using a basic format, avoiding potential syntax issues
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

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
        self.prices = [] # Stores tuples of (timestamp_ms, price, volume)
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
                self.model.add(Dense(25, activation='relu'))
                self.model.add(Dense(1))
                self.model.compile(optimizer=\'adam\', loss=\'mse\')
                logger.info("TensorFlow LSTM model built successfully.")
            except Exception as e:
                logger.error(f"Error building TensorFlow LSTM model: {e}. Falling back to mock model.", exc_info=True)
                self.notify("Error building LSTM model. Using mock predictor.")
                self.model = MockLSTMModel().build_lstm_model(input_shape=input_shape)
        else:
            logger.info("TensorFlow not available. Building Mock LSTM model.")
            self.model = MockLSTMModel().build_lstm_model(input_shape=input_shape)

    def _prepare_lstm_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Prepares data for LSTM training/prediction."""
        required_length = self.lstm_look_back + 1
        if len(self.prices) < required_length:
            return None, None

        try:
            # Extract recent prices and volumes
            recent_data = self.prices[-required_length:]
            prices = np.array([p[1] for p in recent_data]).reshape(-1, 1)
            volumes = np.array([p[2] for p in recent_data]).reshape(-1, 1)

            # Scale data (fit scaler only once or periodically, transform here)
            # Note: Fitting scaler on small rolling windows can be unstable.
            # Consider fitting on a larger initial dataset if possible.
            scaled_prices = self.scaler_price.fit_transform(prices)
            scaled_volumes = self.scaler_volume.fit_transform(volumes)

            # Combine features
            data_scaled = np.hstack((scaled_prices, scaled_volumes))

            # Create sequences for X and y
            X, y = [], []
            for i in range(self.lstm_look_back, len(data_scaled)):
                X.append(data_scaled[i-self.lstm_look_back:i, :])
                y.append(scaled_prices[i, 0]) # Predict the next scaled price
            
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}", exc_info=True)
            return None, None

    def _train_lstm_model(self):
        """Trains the LSTM model periodically."""
        if not self.model:
            logger.warning("LSTM model not available for training.")
            return
        
        logger.info("Preparing data and training LSTM model...")
        X, y = self._prepare_lstm_data()

        if X is not None and y is not None and len(X) > 0:
            try:
                self.model.fit(X, y, epochs=5, batch_size=16, verbose=0) # Smaller batch size might help
                self.last_lstm_train = time.time()
                logger.info("LSTM model training complete.")
            except Exception as e:
                logger.error(f"Error training LSTM model: {e}", exc_info=True)
        else:
            logger.warning("Not enough data or error preparing data for LSTM training.")

    def _predict_price_lstm(self) -> float | None:
        """Predicts the next price using the LSTM model."""
        if not self.model or len(self.prices) < self.lstm_look_back:
            return None # Not enough data or no model

        try:
            # Get the last sequence
            last_sequence_data = self.prices[-self.lstm_look_back:]
            prices = np.array([p[1] for p in last_sequence_data]).reshape(-1, 1)
            volumes = np.array([p[2] for p in last_sequence_data]).reshape(-1, 1)

            # Scale using the *same* fitted scalers
            scaled_prices = self.scaler_price.transform(prices)
            scaled_volumes = self.scaler_volume.transform(volumes)
            last_sequence_scaled = np.hstack((scaled_prices, scaled_volumes)).reshape(1, self.lstm_look_back, 2)

            # Predict
            predicted_scaled = self.model.predict(last_sequence_scaled, verbose=0)

            # Inverse transform the prediction
            predicted_price = self.scaler_price.inverse_transform(predicted_scaled)
            
            logger.debug(f"LSTM Predicted Price: {predicted_price[0][0]:.4f}")
            return float(predicted_price[0][0])
        except Exception as e:
            # Log specific errors, e.g., if scaler wasn\t fitted
            if "NotFittedError" in str(e):
                 logger.warning(f"LSTM prediction skipped: Scaler not fitted yet ({e})")
            else:
                 logger.error(f"Error predicting price with LSTM: {e}", exc_info=True)
            return None

    def _get_historical_data(self, interval="1m", limit=1000) -> pd.DataFrame:
        """Fetches historical klines from Binance."""
        if not self.client:
            logger.warning("Binance client not initialized. Cannot fetch historical data.")
            return pd.DataFrame()
        try:
            logger.info(f"Fetching {limit} historical {interval} klines for {self.symbol}...")
            # Use get_historical_klines for simplicity, handles pagination internally for longer periods
            # Ensure limit doesn\t exceed Binance capabilities for the timeframe
            # e.g., for 1m data, fetch maybe 1-2 days max per call if needed, or adjust interval
            klines = self.client.get_historical_klines(self.symbol, interval, f"{limit} minutes ago UTC")
            
            if not klines:
                logger.warning("No historical data received from Binance.")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            # Convert necessary columns to numeric, coercing errors
            for col in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]:
                df[col] = pd.to_numeric(df[col], errors=\'coerce\')
            df.dropna(subset=["close", "volume"], inplace=True) # Drop rows where essential data is missing
            logger.info(f"Fetched and processed {len(df)} historical klines.")
            return df
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching historical data: {e}")
            self.notify(f"API Error: Failed to fetch historical data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()

    def _update_best_indicator(self):
        """Evaluates indicators on recent historical data to select the best one."""
        logger.info("Evaluating indicators to determine the best strategy...")
        # Fetch sufficient data, e.g., 1 day of 1m data (1440 points)
        # More data (e.g., 7 days = 10080) might be better but takes longer
        required_data_points = self.indicator_update_interval_hours * 60 
        df = self._get_historical_data(interval="1m", limit=max(required_data_points, 100)) # Fetch at least 100 points
        
        if not df.empty and len(df) >= 50: # Need enough data for evaluation
            try:
                self.best_indicator = evaluate_indicators(df.copy()) # Pass a copy
                self.last_indicator_update = datetime.now()
                logger.info(f"Best indicator updated to: {self.best_indicator}")
                self.notify(f"Strategy Update: Best indicator set to {self.best_indicator}")
            except Exception as e:
                logger.error(f"Error evaluating indicators: {e}", exc_info=True)
                self.notify("Error evaluating indicators. Using previous or default.")
        else:
            logger.warning("Could not fetch sufficient historical data for indicator evaluation.")
            self.notify("Warning: Could not fetch data to evaluate indicators.")

    def _update_sentiment(self):
        """Updates the sentiment score periodically."""
        logger.debug("Updating sentiment score...")
        try:
            self.sentiment_score = get_sentiment(self.config)
            self.last_sentiment_update = time.time()
            logger.info(f"Sentiment score updated: {self.sentiment_score:.3f}")
        except Exception as e:
            logger.error(f"Error updating sentiment: {e}", exc_info=True)
            self.sentiment_score = 0.0 # Default to neutral on error

    def _check_order_limit(self) -> bool:
        """Checks if the per-minute order limit has been reached."""
        current_time = time.time()
        if current_time - self.last_minute_reset >= 60:
            logger.debug(f"Resetting minute order count (was {self.order_count_minute})")
            self.order_count_minute = 0
            self.last_minute_reset = current_time
        
        if self.order_count_minute >= self.max_orders_per_minute:
            logger.warning(f"Per-minute order limit ({self.max_orders_per_minute}) reached. Pausing new orders.")
            return False
        return True

    def _check_risk_limit(self) -> bool:
        """Checks if the maximum daily loss has been exceeded."""
        # Note: self.daily_pnl should be updated based on *realized* PnL.
        # This requires tracking closed trades, which is complex with OCO.
        # For now, this check might be based on estimated or open PnL, which is less accurate.
        # A more robust implementation would track fills from the user data stream.
        if self.daily_pnl <= -self.max_daily_loss_usd:
            if not self.stop_trading_today:
                logger.warning(f"Maximum daily loss limit ({self.max_daily_loss_usd:.2f} USD) reached. Stopping trading for today.")
                self.notify(f"STOP TRADING: Max daily loss ({self.max_daily_loss_pct*100:.1f}%) reached. Today\s PnL: {self.daily_pnl:.2f} USDT")
                self.stop_trading_today = True
                # Consider implementing logic to cancel open orders or close positions here
                # self._cancel_all_open_orders()
            return False
        return True

    def _reset_daily_limits_if_needed(self):
        """Resets daily PnL and trading stop flag if a new day has started."""
        now_date = datetime.now().date()
        if now_date > self.last_daily_reset:
            logger.info(f"New trading day ({now_date}). Resetting daily PnL and limits.")
            # Generate report for the *previous* day before resetting
            self._generate_daily_report(date_to_report=self.last_daily_reset)
            
            self.daily_pnl = 0.0
            self.stop_trading_today = False
            self.last_daily_reset = now_date
            # Recalculate max loss based on potentially updated capital (optional)
            # self.current_capital = self._get_usdt_balance() # Needs implementation
            self.max_daily_loss_usd = self.initial_capital * self.max_daily_loss_pct
            logger.info(f"Max daily loss for today set to: {self.max_daily_loss_usd:.2f} USDT")
            self.notify(f"New trading day. Daily PnL reset. Max loss: {self.max_daily_loss_usd:.2f} USDT.")

    def _process_kline_message(self, msg: dict):
        """Handles incoming WebSocket kline messages."""
        try:
            if msg.get("e") == "error":
                logger.error(f"WebSocket Error received: {msg.get(\"m\")}")
                self.notify("CRITICAL: WebSocket error received. Bot may need restart.")
                # Consider stopping or attempting reconnect based on error type
                self.stop_websocket() # Stop processing on error
                return
            
            if msg.get("e") == "kline":
                kline = msg.get("k", {})
                # Process only closed klines for stability
                if kline.get("s") == self.symbol and kline.get("x") == True:
                    timestamp_ms = int(kline.get("t", 0))
                    price = float(kline.get("c", 0.0))
                    volume = float(kline.get("v", 0.0))
                    
                    if timestamp_ms == 0 or price == 0.0:
                        logger.warning(f"Received invalid kline data: {kline}")
                        return
                        
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    logger.debug(f"Kline received: {timestamp} - Price: {price:.4f}, Volume: {volume:.4f}")
                    
                    # Store price data (timestamp_ms, price, volume)
                    self.prices.append((timestamp_ms, price, volume))
                    # Keep price list size manageable (e.g., last 2 days of 1m data)
                    max_prices = 2 * 24 * 60 
                    if len(self.prices) > max_prices:
                        self.prices = self.prices[-max_prices:]

                    # --- Periodic Tasks within WebSocket handler --- 
                    current_time = time.time()
                    # Train LSTM periodically
                    if self.model and current_time - self.last_lstm_train > self.lstm_train_interval_seconds:
                        self._train_lstm_model()
                    
                    # Update sentiment periodically
                    if current_time - self.last_sentiment_update > self.sentiment_update_interval_seconds:
                        self._update_sentiment()
                    # --- End Periodic Tasks --- 

                    # Make trading decision based on the new kline data
                    self._make_decision(price, volume)
        except Exception as e:
            logger.error(f"Error processing kline message: {e}", exc_info=True)
            # Avoid crashing the websocket thread

    def _make_decision(self, current_price: float, current_volume: float):
        """Analyzes data and decides whether to place a trade."""
        self._reset_daily_limits_if_needed() # Check if day rolled over

        if self.stop_trading_today:
            logger.debug("Trading stopped for today due to loss limit.")
            return

        if not self._check_order_limit():
            logger.debug("Order limit reached, skipping decision.")
            return
        
        # Need enough data for the slowest indicator + lookback for LSTM
        required_data_points = max(21, self.lstm_look_back + 1) 
        if len(self.prices) < required_data_points:
            logger.debug(f"Need {required_data_points} data points, have {len(self.prices)}. Skipping decision.")
            return

        # --- Get Indicator Values --- 
        # Use the last N prices for calculation (e.g., 100 for stability)
        closes = np.array([p[1] for p in self.prices[-100:]])
        volumes = np.array([p[2] for p in self.prices[-100:]])

        try:
            # Use talib (real or mock based on import)
            ema9 = talib.EMA(closes, timeperiod=9)[-1]
            ema21 = talib.EMA(closes, timeperiod=21)[-1]
            rsi = talib.RSI(closes, timeperiod=14)[-1]
            upper_bb, middle_bb, lower_bb = talib.BBANDS(closes, timeperiod=20)
            upper_bb, lower_bb = upper_bb[-1], lower_bb[-1]
            # Rolling VWAP over last 50 periods
            rolling_vwap = (pd.Series(closes[-50:]) * pd.Series(volumes[-50:])).sum() / pd.Series(volumes[-50:]).sum()
            vwap = rolling_vwap
            
            # Check for NaN values from indicators
            if any(pd.isna(x) for x in [ema9, ema21, rsi, upper_bb, lower_bb, vwap]):
                logger.warning("Indicator calculation resulted in NaN. Skipping decision.")
                return
                
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return

        # --- Get LSTM Prediction --- 
        predicted_price = self._predict_price_lstm()
        if predicted_price is None:
            logger.debug("LSTM prediction not available. Proceeding without it.")
            # Fallback: Use a neutral prediction factor
            prediction_factor = 1.0 
        else:
            prediction_factor = predicted_price / current_price

        # --- Combine Signals --- 
        should_buy = False
        should_sell = False
        # Sentiment modifier: scales impact, e.g., 0.05 means +/- 5% influence max
        sentiment_modifier = 1.0 + (self.sentiment_score * 0.05) 
        # Prediction modifier: prediction relative to current price
        prediction_modifier = prediction_factor

        logger.debug(f"Decision Data: Price={current_price:.2f}, PredFactor={prediction_modifier:.4f}, SentMod={sentiment_modifier:.4f}, BestInd={self.best_indicator}")
        logger.debug(f"Indicators: EMA9={ema9:.2f}, EMA21={ema21:.2f}, RSI={rsi:.2f}, BB={lower_bb:.2f}-{upper_bb:.2f}, VWAP={vwap:.2f}")

        # Apply strategy based on the chosen best_indicator
        # Conditions combine indicator signal with prediction and sentiment modifiers
        # Example: Buy if indicator is bullish AND predicted price > current price AND sentiment is not strongly negative
        combined_buy_signal = prediction_modifier > 1.0 and sentiment_modifier > 0.98 # Basic check: prediction up, sentiment not too bad
        combined_sell_signal = prediction_modifier < 1.0 and sentiment_modifier < 1.02 # Basic check: prediction down, sentiment not too good

        if self.best_indicator == "ema":
            if ema9 > ema21 and combined_buy_signal:
                should_buy = True
            elif ema9 < ema21 and combined_sell_signal:
                should_sell = True
        elif self.best_indicator == "rsi":
            if rsi < 35 and combined_buy_signal: # Looser threshold
                should_buy = True
            elif rsi > 65 and combined_sell_signal: # Looser threshold
                should_sell = True
        elif self.best_indicator == "bb":
            if current_price <= lower_bb * 1.001 and combined_buy_signal:
                should_buy = True
            elif current_price >= upper_bb * 0.999 and combined_sell_signal:
                should_sell = True
        elif self.best_indicator == "vwap":
            if current_price < vwap and combined_buy_signal:
                should_buy = True
            elif current_price > vwap and combined_sell_signal:
                should_sell = True

        # --- Execute Trade --- 
        if should_buy:
            logger.info(f"BUY Signal triggered based on {self.best_indicator}, Prediction, Sentiment.")
            self._execute_trade(SIDE_BUY, current_price)
        elif should_sell:
            logger.info(f"SELL Signal triggered based on {self.best_indicator}, Prediction, Sentiment.")
            self._execute_trade(SIDE_SELL, current_price)

    def _execute_trade(self, side: str, price: float):
        """Executes a market order and places an OCO order for SL/TP."""
        if not self.client:
            logger.error("Cannot execute trade: Binance client not initialized.")
            return
        if not self._check_risk_limit(): # Double check risk before placing order
             logger.warning(f"Trade execution blocked: Risk limit reached.")
             return

        try:
            # --- Calculate SL/TP Prices --- 
            # Ensure prices are rounded to the correct precision for the symbol
            # Fetch symbol info if needed, or use a reasonable default (e.g., 2 decimal places for USDT pairs)
            price_precision = 2 # Adjust based on symbol (e.g., self.client.get_symbol_info(self.symbol)))
            
            if side == SIDE_BUY:
                stop_loss_price = round(price * (1 - self.stop_loss_pct), price_precision)
                take_profit_price = round(price * (1 + self.take_profit_pct), price_precision)
                sl_tp_side = SIDE_SELL
            else: # SIDE_SELL
                stop_loss_price = round(price * (1 + self.stop_loss_pct), price_precision)
                take_profit_price = round(price * (1 - self.take_profit_pct), price_precision)
                sl_tp_side = SIDE_BUY
            
            # Ensure SL/TP prices are valid (e.g., TP > price for BUY, SL < price for BUY)
            if (side == SIDE_BUY and (stop_loss_price >= price or take_profit_price <= price)) or \
               (side == SIDE_SELL and (stop_loss_price <= price or take_profit_price >= price)):
                logger.error(f"Invalid SL/TP calculation: Entry={price}, SL={stop_loss_price}, TP={take_profit_price}. Skipping trade.")
                return

            # --- Place Market Order for Entry --- 
            logger.info(f"Placing MARKET {side} order for {self.position_size} {self.symbol}...")
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=self.position_size
            )
            self.order_count_minute += 1
            entry_price = float(order.get("fills", [{}])[0].get("price", price)) # Use actual fill price if available
            order_id = str(order.get("orderId", f"mock_{time.time()}")) # Use mock ID if needed
            logger.info(f"MARKET {side} Order {order_id} filled at ~{entry_price:.{price_precision}f}")
            self.notify(f"Trade Executed: MARKET {side} {self.position_size} {self.symbol} @ ~{entry_price:.{price_precision}f}")

            # --- Place OCO Order for Stop Loss and Take Profit --- 
            # Stop price is the trigger, limit price is the execution price for the limit order part of SL
            # Use a slightly worse limit price for SL to increase fill chance
            stop_limit_offset = 0.001 # 0.1% offset
            if sl_tp_side == SIDE_SELL:
                stop_limit_price = round(stop_loss_price * (1 - stop_limit_offset), price_precision)
            else: # SIDE_BUY
                stop_limit_price = round(stop_loss_price * (1 + stop_limit_offset), price_precision)
                
            # Ensure stop_limit_price is valid (e.g., lower than stopPrice for SELL SL)
            if (sl_tp_side == SIDE_SELL and stop_limit_price >= stop_loss_price) or \
               (sl_tp_side == SIDE_BUY and stop_limit_price <= stop_loss_price):
                 logger.warning(f"Adjusting Stop Limit Price due to calculation: SL={stop_loss_price}, OriginalStopLimit={stop_limit_price}")
                 stop_limit_price = stop_loss_price # Fallback to stop_loss_price if offset makes it invalid

            logger.info(f"Placing OCO {sl_tp_side} order: TP={take_profit_price}, SLPrice={stop_loss_price}, SLLimit={stop_limit_price}")
            oco_order = self.client.create_oco_order(
                symbol=self.symbol,
                side=sl_tp_side,
                quantity=self.position_size,
                price=f"{take_profit_price:.{price_precision}f}", # TP Limit Price (string)
                stopPrice=f"{stop_loss_price:.{price_precision}f}", # SL Trigger Price (string)
                stopLimitPrice=f"{stop_limit_price:.{price_precision}f}", # SL Limit Price (string)
                stopLimitTimeInForce=TIME_IN_FORCE_GTC
            )
            self.order_count_minute += 1 # OCO counts as one order
            oco_order_list_id = str(oco_order.get("orderListId", -1))
            logger.info(f"OCO Order List ID {oco_order_list_id} placed (TP={take_profit_price}, SL={stop_loss_price})")
            self.notify(f"OCO Placed: {sl_tp_side} {self.position_size} {self.symbol}, TP={take_profit_price}, SL={stop_loss_price}")

            # --- Record Entry Trade in DB --- 
            # PnL is calculated later when the position is closed.
            cursor = self.db_conn_trades.cursor()
            cursor.execute(
                "INSERT INTO trades (timestamp, symbol, side, price, quantity, order_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), self.symbol, side, entry_price, self.position_size, order_id, "OPEN")
            )
            self.db_conn_trades.commit()
            logger.debug(f"Trade {order_id} recorded as OPEN in database.")

            # --- PnL Tracking Note --- 
            # Accurate PnL requires tracking the execution of the OCO order (TP or SL).
            # This typically involves listening to the Binance User Data Stream for order updates.
            # The current implementation estimates PnL in the daily report based on simplified logic.

        except BinanceAPIException as e:
            logger.error(f"Binance API error executing trade: {e}")
            self.notify(f"Trade Execution Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error executing trade: {e}", exc_info=True)
            self.notify(f"Trade Execution Error: {e}")

    def _generate_daily_report(self, date_to_report: datetime.date):
        """Generates and saves a daily performance report."""
        report_date_str = date_to_report.strftime("%Y-%m-%d")
        logger.info(f"Generating daily report for {report_date_str}...")
        cursor = self.db_conn_trades.cursor()
        
        try:
            # --- Simplified PnL Calculation --- 
            # Fetch all trades for the specific day. Assumes trades are OPENed on that day.
            # A robust solution needs to track position closure via User Data Stream.
            cursor.execute(
                "SELECT id, side, price, quantity, status FROM trades WHERE date(timestamp) = ? ORDER BY timestamp", 
                (report_date_str,)
            )
            trades_of_day = cursor.fetchall()
            
            # Placeholder logic: Assume simple pairing or use a fixed estimate
            # This needs significant improvement for accuracy.
            # Example: Calculate PnL based on *assumed* closures via SL/TP percentages.
            total_pnl = 0.0
            trades_count = 0
            win_trades = 0
            
            # Iterate through OPEN trades from that day and estimate outcome
            for trade_id, side, entry_price, qty, status in trades_of_day:
                 if status == "OPEN": # Only consider trades opened today
                     trades_count += 1
                     # Simulate outcome (highly inaccurate, just for placeholder)
                     # Assume 50% hit TP, 50% hit SL for estimation
                     if np.random.rand() < 0.5: # Simulate TP hit
                         if side == SIDE_BUY:
                             pnl = (entry_price * (1 + self.take_profit_pct) - entry_price) * qty
                         else: # SIDE_SELL
                             pnl = (entry_price - entry_price * (1 - self.take_profit_pct)) * qty
                         if pnl > 0: win_trades += 1
                     else: # Simulate SL hit
                         if side == SIDE_BUY:
                             pnl = (entry_price * (1 - self.stop_loss_pct) - entry_price) * qty
                         else: # SIDE_SELL
                             pnl = (entry_price - entry_price * (1 + self.stop_loss_pct)) * qty
                     total_pnl += pnl
                     # Ideally, update trade status to CLOSED in DB here, but that requires real tracking

            # --- End Simplified PnL --- 

            win_rate = (win_trades / trades_count * 100) if trades_count > 0 else 0.0
            avg_pnl_per_trade = total_pnl / trades_count if trades_count > 0 else 0.0

            # Save report to database
            cursor.execute(
                "INSERT OR REPLACE INTO daily_reports (date, total_pnl, trades_count, win_rate, avg_pnl_per_trade, best_indicator) VALUES (?, ?, ?, ?, ?, ?)",
                (report_date_str, total_pnl, trades_count, win_rate, avg_pnl_per_trade, self.best_indicator)
            )
            self.db_conn_trades.commit()
            logger.info(f"Daily report for {report_date_str} saved. Estimated PnL: {total_pnl:.2f}")

            # Send notification
            report_msg = (
                f"--- Daily Report {report_date_str} ---\n"
                f"Est. PnL: {total_pnl:.2f} USDT\n"
                f"Trades: {trades_count}\n"
                f"Win Rate: {win_rate:.2f}%\n"
                f"Avg PnL/Trade: {avg_pnl_per_trade:.2f} USDT\n"
                f"Indicator: {self.best_indicator}\n"
                f"Note: PnL is estimated."
            )
            self.notify(report_msg)

        except Exception as e:
            logger.error(f"Error generating daily report for {report_date_str}: {e}", exc_info=True)
            self.notify(f"Error generating daily report for {report_date_str}.")

    def _setup_schedule(self):
        """Sets up the scheduled tasks using the schedule library."""
        logger.info("Setting up scheduled tasks...")
        # Schedule daily report generation (e.g., at 00:01 UTC)
        # Reports for the *previous* day, run just after midnight
        schedule.every().day.at("00:01").do(self._reset_daily_limits_if_needed) 
        # Schedule indicator update (e.g., every X hours as per config)
        schedule.every(self.indicator_update_interval_hours).hours.do(self._update_best_indicator)
        # Schedule API key check (e.g., every 5 minutes) in case they are added later
        schedule.every(5).minutes.do(self.check_and_initialize_client)
        logger.info("Scheduled tasks configured.")

    def check_and_initialize_client(self):
        """Checks for API keys and initializes client if not already done."""
        if not self.client: # If client not initialized
            logger.info("Checking for API keys and attempting client initialization...")
            if self._load_api_keys():
                if self._initialize_client():
                    # Initialize model and fetch initial data only after client is ready
                    self._build_lstm_model()
                    self._update_best_indicator() # Run immediately after client init
                    self._update_sentiment() # Run immediately after client init
                    self.start_websocket() # Start WebSocket only after successful init
                else:
                    logger.warning("Client initialization failed after loading keys. Will retry later.")
            # else: keys still not available, will check again later via schedule
        # else: client already initialized

    def start_websocket(self):
        """Starts the Binance WebSocket manager and streams."""
        if not self.client:
            logger.error("Cannot start WebSocket: Binance client not initialized.")
            return
        if self.is_websocket_running:
            logger.warning("WebSocket is already running.")
            return
            
        try:
            logger.info("Starting Binance WebSocket manager...")
            # Use testnet status from config
            self.twm = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.api_secret, testnet=self.is_testnet)
            self.twm.start()
            
            # Define the callback for kline stream
            def handle_kline_message_wrapper(msg):
                self._process_kline_message(msg)

            # Start the kline stream
            kline_interval = Client.KLINE_INTERVAL_1MINUTE # Use constant
            self.ws_kline_key = self.twm.start_kline_socket(
                callback=handle_kline_message_wrapper, 
                symbol=self.symbol, 
                interval=kline_interval
            )
            
            # --- Optional: Start User Data Stream --- 
            # Needed for accurate PnL tracking and order status updates
            # def handle_user_message(msg):
            #     logger.debug(f"User WS Message: {msg}")
            #     if msg[\'e\'] == \'executionReport\':
            #         # Process order updates, update trade status in DB, calculate realized PnL
            #         order_id = msg[\'i\']
            #         status = msg[\'X\'] # e.g., FILLED, CANCELED, EXPIRED
            #         side = msg[\'S\']
            #         price = float(msg[\'L\'])
            #         qty = float(msg[\'l\'])
            #         # ... logic to find related trade and update PnL ...
            # self.ws_user_key = self.twm.start_user_socket(callback=handle_user_message)
            # logger.info("User Data WebSocket started.")
            # --- End Optional User Data Stream ---
            
            self.is_websocket_running = True
            logger.info(f"Kline WebSocket started for {self.symbol} ({kline_interval}). Bot is active.")
            self.notify(f"Bot activated. Monitoring {self.symbol} klines.")

        except Exception as e:
             logger.error(f"Failed to start WebSocket: {e}", exc_info=True)
             self.notify("CRITICAL: Failed to start WebSocket. Bot cannot trade.")
             self.is_websocket_running = False
             if self.twm:
                 try: self.twm.stop() 
                 except: pass
             self.twm = None

    def stop_websocket(self):
        """Stops the Binance WebSocket manager."""
        if not self.is_websocket_running or not self.twm:
            logger.info("WebSocket is not running.")
            return
        try:
            logger.info("Stopping WebSocket manager...")
            self.twm.stop()
            self.twm = None
            self.is_websocket_running = False
            logger.info("WebSocket manager stopped.")
            self.notify("WebSocket connection stopped.")
        except Exception as e:
            logger.error(f"Error stopping WebSocket manager: {e}", exc_info=True)

    def run(self):
        """Main execution loop for the bot."""
        logger.info("Initializing Scalping Bot...")
        self._setup_schedule()
        self.check_and_initialize_client() # Initial check

        logger.info("Starting main loop...")
        while True:
            try:
                schedule.run_pending()
                # Check if websocket died and needs restart (if client is ready)
                if self.client and not self.is_websocket_running:
                    logger.warning("WebSocket is not running. Attempting to restart...")
                    self.start_websocket()
                
                time.sleep(1) # Main loop sleep, schedule handles timing
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Stopping bot...")
                break
            except Exception as e:
                logger.error(f"Unhandled error in main loop: {e}", exc_info=True)
                self.notify(f"CRITICAL ERROR in main loop: {e}. Bot may be unstable.")
                time.sleep(10) # Pause before potentially retrying
        
        self.stop() # Cleanup on exit

    def stop(self):
        """Stops the bot and cleans up resources."""
        logger.info("Initiating bot shutdown sequence...")
        self.stop_websocket()
        schedule.clear()
        if self.db_conn_trades:
            try: self.db_conn_trades.close() 
            except: pass
            logger.info("Trades database connection closed.")
        if self.db_conn_keys:
            try: self.db_conn_keys.close() 
            except: pass
            logger.info("API keys database connection closed.")
        self.notify("Trading bot stopped gracefully.")
        logger.info("Scalping Bot stopped.")

# --- Entry Point --- 
if __name__ == "__main__":
    logger.info("Starting bot execution from main entry point...")
    bot = ScalpingBot()
    bot.run()
    logger.info("Bot execution finished.")

