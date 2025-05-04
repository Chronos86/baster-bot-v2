# bot/indicators.py

import pandas as pd
import numpy as np
import logging

# Attempt to import the actual TA-Lib
try:
    import talib
    logging.info("Successfully imported TA-Lib library.")
except ImportError:
    logging.warning("TA-Lib library not found. Using mock implementations for indicators. Please install TA-Lib for accurate calculations.")
    # --- Mock TA-Lib Implementation (if import fails) ---
    class MockTALib:
        def EMA(self, closes, timeperiod):
            return pd.Series(closes).rolling(window=timeperiod, min_periods=timeperiod).mean().to_numpy()
        def RSI(self, closes, timeperiod):
            delta = pd.Series(closes).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod, min_periods=timeperiod).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod, min_periods=timeperiod).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Fill initial NaNs with 50 (neutral)
            rsi = rsi.fillna(50)
            return rsi.to_numpy()
        def BBANDS(self, closes, timeperiod):
            sma = pd.Series(closes).rolling(window=timeperiod, min_periods=timeperiod).mean()
            std = pd.Series(closes).rolling(window=timeperiod, min_periods=timeperiod).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            return upper.to_numpy(), sma.to_numpy(), lower.to_numpy()

    talib = MockTALib()
    # --- End Mock TA-Lib ---

def evaluate_indicators(df: pd.DataFrame) -> str:
    """Evaluates different technical indicators on historical data to find the best performing one.

    Args:
        df: Pandas DataFrame with historical kline data, must include 'close' and 'volume' columns.

    Returns:
        The name of the best performing indicator ('ema', 'rsi', 'bb', 'vwap').
    """
    results = {}
    initial_capital = 10000
    commission = 0.001 # Assume 0.1% commission per trade (applied on sell)

    if df.empty or len(df) < 50: # Need sufficient data for indicators
        logging.warning("Insufficient data for indicator evaluation. Returning default 'ema'.")
        return "ema"

    # Ensure correct data types
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])

    # --- EMA Crossover Strategy --- 
    try:
        df["ema9"] = talib.EMA(df["close"], timeperiod=9)
        df["ema21"] = talib.EMA(df["close"], timeperiod=21)
        capital = initial_capital
        position = 0 # 0: no position, >0: long position size
        trades = []
        for i in range(1, len(df)):
            # Check for NaN values before comparing
            if pd.isna(df["ema9"].iloc[i]) or pd.isna(df["ema21"].iloc[i]) or pd.isna(df["ema9"].iloc[i-1]) or pd.isna(df["ema21"].iloc[i-1]):
                continue
            # Buy signal: EMA9 crosses above EMA21
            if df["ema9"].iloc[i] > df["ema21"].iloc[i] and df["ema9"].iloc[i-1] <= df["ema21"].iloc[i-1] and position == 0:
                buy_price = df["close"].iloc[i]
                position = capital / buy_price # Invest all capital
                capital = 0
                trades.append(("BUY", buy_price))
            # Sell signal: EMA9 crosses below EMA21
            elif df["ema9"].iloc[i] < df["ema21"].iloc[i] and df["ema9"].iloc[i-1] >= df["ema21"].iloc[i-1] and position > 0:
                sell_price = df["close"].iloc[i]
                capital = position * sell_price * (1 - commission) # Apply commission on sell
                trades.append(("SELL", sell_price))
                position = 0
        # If still holding position at the end, close it at the last price
        if position > 0:
            capital = position * df["close"].iloc[-1] * (1 - commission)
        final_value_ema = capital
        win_trades_ema = len([t for i, t in enumerate(trades) if t[0] == "SELL" and i > 0 and t[1] > trades[i-1][1]])
        total_sell_trades_ema = len([t for t in trades if t[0] == "SELL"])
        results["ema"] = {"profit": final_value_ema - initial_capital, "win_rate": win_trades_ema / total_sell_trades_ema if total_sell_trades_ema else 0}
    except Exception as e:
        logging.error(f"Error evaluating EMA strategy: {e}")
        results["ema"] = {"profit": -np.inf, "win_rate": 0} # Penalize on error

    # --- RSI Strategy --- 
    try:
        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        capital = initial_capital
        position = 0
        trades = []
        for i in range(1, len(df)):
            if pd.isna(df["rsi"].iloc[i]):
                continue
            # Buy signal: RSI crosses below 30
            if df["rsi"].iloc[i] < 30 and df["rsi"].iloc[i-1] >= 30 and position == 0:
                buy_price = df["close"].iloc[i]
                position = capital / buy_price
                capital = 0
                trades.append(("BUY", buy_price))
            # Sell signal: RSI crosses above 70
            elif df["rsi"].iloc[i] > 70 and df["rsi"].iloc[i-1] <= 70 and position > 0:
                sell_price = df["close"].iloc[i]
                capital = position * sell_price * (1 - commission)
                trades.append(("SELL", sell_price))
                position = 0
        if position > 0:
            capital = position * df["close"].iloc[-1] * (1 - commission)
        final_value_rsi = capital
        win_trades_rsi = len([t for i, t in enumerate(trades) if t[0] == "SELL" and i > 0 and t[1] > trades[i-1][1]])
        total_sell_trades_rsi = len([t for t in trades if t[0] == "SELL"])
        results["rsi"] = {"profit": final_value_rsi - initial_capital, "win_rate": win_trades_rsi / total_sell_trades_rsi if total_sell_trades_rsi else 0}
    except Exception as e:
        logging.error(f"Error evaluating RSI strategy: {e}")
        results["rsi"] = {"profit": -np.inf, "win_rate": 0}

    # --- Bollinger Bands Strategy --- 
    try:
        df["upper_bb"], df["middle_bb"], df["lower_bb"] = talib.BBANDS(df["close"], timeperiod=20)
        capital = initial_capital
        position = 0
        trades = []
        for i in range(1, len(df)):
            if pd.isna(df["lower_bb"].iloc[i]) or pd.isna(df["upper_bb"].iloc[i]) or pd.isna(df["close"].iloc[i]):
                continue
            # Buy signal: Price touches or crosses below lower band
            if df["close"].iloc[i] <= df["lower_bb"].iloc[i] and position == 0:
                buy_price = df["close"].iloc[i]
                position = capital / buy_price
                capital = 0
                trades.append(("BUY", buy_price))
            # Sell signal: Price touches or crosses above upper band
            elif df["close"].iloc[i] >= df["upper_bb"].iloc[i] and position > 0:
                sell_price = df["close"].iloc[i]
                capital = position * sell_price * (1 - commission)
                trades.append(("SELL", sell_price))
                position = 0
        if position > 0:
            capital = position * df["close"].iloc[-1] * (1 - commission)
        final_value_bb = capital
        win_trades_bb = len([t for i, t in enumerate(trades) if t[0] == "SELL" and i > 0 and t[1] > trades[i-1][1]])
        total_sell_trades_bb = len([t for t in trades if t[0] == "SELL"])
        results["bb"] = {"profit": final_value_bb - initial_capital, "win_rate": win_trades_bb / total_sell_trades_bb if total_sell_trades_bb else 0}
    except Exception as e:
        logging.error(f"Error evaluating Bollinger Bands strategy: {e}")
        results["bb"] = {"profit": -np.inf, "win_rate": 0}

    # --- VWAP Strategy --- 
    try:
        # Calculate VWAP (cumulative)
        df["cum_vol"] = df["volume"].cumsum()
        df["cum_vol_price"] = (df["close"] * df["volume"]).cumsum()
        df["vwap"] = df["cum_vol_price"] / df["cum_vol"]
        capital = initial_capital
        position = 0
        trades = []
        for i in range(1, len(df)):
            if pd.isna(df["vwap"].iloc[i]) or pd.isna(df["close"].iloc[i]):
                continue
            # Buy signal: Price crosses below VWAP
            if df["close"].iloc[i] < df["vwap"].iloc[i] and df["close"].iloc[i-1] >= df["vwap"].iloc[i-1] and position == 0:
                buy_price = df["close"].iloc[i]
                position = capital / buy_price
                capital = 0
                trades.append(("BUY", buy_price))
            # Sell signal: Price crosses above VWAP
            elif df["close"].iloc[i] > df["vwap"].iloc[i] and df["close"].iloc[i-1] <= df["vwap"].iloc[i-1] and position > 0:
                sell_price = df["close"].iloc[i]
                capital = position * sell_price * (1 - commission)
                trades.append(("SELL", sell_price))
                position = 0
        if position > 0:
            capital = position * df["close"].iloc[-1] * (1 - commission)
        final_value_vwap = capital
        win_trades_vwap = len([t for i, t in enumerate(trades) if t[0] == "SELL" and i > 0 and t[1] > trades[i-1][1]])
        total_sell_trades_vwap = len([t for t in trades if t[0] == "SELL"])
        results["vwap"] = {"profit": final_value_vwap - initial_capital, "win_rate": win_trades_vwap / total_sell_trades_vwap if total_sell_trades_vwap else 0}
    except Exception as e:
        logging.error(f"Error evaluating VWAP strategy: {e}")
        results["vwap"] = {"profit": -np.inf, "win_rate": 0}

    # --- Determine Best Indicator --- 
    # Use a scoring system (e.g., profit * win_rate), handling potential zero/negative profits/rates
    best_indicator = "ema" # Default
    max_score = -np.inf

    logging.info(f"Indicator Evaluation Results: {results}")

    for indicator, metrics in results.items():
        # Ensure metrics are valid numbers
        profit = metrics.get("profit", -np.inf)
        win_rate = metrics.get("win_rate", 0)

        if not isinstance(profit, (int, float)) or not isinstance(win_rate, (int, float)):
            continue

        # Simple scoring: prioritize profit, use win_rate as tie-breaker or multiplier
        # Add a small constant to win_rate to avoid issues with 0 win rate
        # Give higher weight to profit
        score = profit * (win_rate + 0.1) # Example scoring

        if score > max_score:
            max_score = score
            best_indicator = indicator

    logging.info(f"Selected Best Indicator based on evaluation: {best_indicator}")
    return best_indicator

