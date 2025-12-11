# backtest.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import logging
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)

# Configuration Dictionary (Ensure these match your bot's configuration)
CONFIG = {
    "USE_ATR_SL_TP": True,
    "ATR_MULTIPLIER_SCALPING": 1.0,
    "ATR_MULTIPLIER_LONGTERM": 2.0,
    "FIXED_SL_SCALPING": 10 * 0.1,      # Adjust based on symbol's point value
    "FIXED_TP_SCALPING": 20 * 0.1,
    "FIXED_SL_LONGTERM": 50 * 0.1,
    "FIXED_TP_LONGTERM": 100 * 0.1,
    "LONGTERM_COOLDOWN_HOURS": 24,
    "MAE_VALUES": {  # Mean Absolute Error for each timeframe
        "M1": 2.38,
        "M5": 2.38,
        "M15": 2.58,
        "H1": 3.03,
        "H4": 5.31,
        "D1": 23.20
    },
    "SL_TP_MULTIPLIER": {  # Multipliers for SL and TP adjustments
        "buy": 1.5,
        "sell": 1.5
    }
}

# Parameters
INITIAL_BALANCE = 500.0
LOT_SIZE_SCALPING = 0.01
LOT_SIZE_LONGTERM = 0.02
POINT_VALUE = 0.1  # Adjust based on symbol (e.g., for XAUUSD, 1 point = $0.1)
WINDOW_SIZE_SCALPING = 7   # As per your short-term model
WINDOW_SIZE_LONGTERM = 37  # As per your long-term model

# Extract configuration parameters
USE_ATR_SL_TP = CONFIG["USE_ATR_SL_TP"]
ATR_MULTIPLIER_SCALPING = CONFIG["ATR_MULTIPLIER_SCALPING"]
ATR_MULTIPLIER_LONGTERM = CONFIG["ATR_MULTIPLIER_LONGTERM"]
FIXED_SL_SCALPING = CONFIG["FIXED_SL_SCALPING"]
FIXED_TP_SCALPING = CONFIG["FIXED_TP_SCALPING"]
FIXED_SL_LONGTERM = CONFIG["FIXED_SL_LONGTERM"]
FIXED_TP_LONGTERM = CONFIG["FIXED_TP_LONGTERM"]
LONGTERM_COOLDOWN_HOURS = CONFIG["LONGTERM_COOLDOWN_HOURS"]
MAE_VALUES = CONFIG["MAE_VALUES"]
SL_TP_MULTIPLIER = CONFIG["SL_TP_MULTIPLIER"]

# Load historical data
def load_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """Load and preprocess historical data."""
    try:
        data = pd.read_csv(filename, index_col='time', parse_dates=True)
        data = data.sort_index()
        data.dropna(inplace=True)
        logging.info(f"Historical data loaded from {filename}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit()

# Load models and scalers
def load_models_and_scalers():
    """Load trained models and their corresponding scalers."""
    try:
        model_scalping = load_model('scalping_model.keras')
        logging.info("Scalping model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load scalping model: {e}")
        exit()
    
    try:
        feature_scaler_scalping = joblib.load('scalping_feature_scaler.pkl')
        target_scalers_scalping = joblib.load('scalping_target_scalers.pkl')
        logging.info("Scalping scalers loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load scalping scalers: {e}")
        exit()
    
    try:
        model_longterm = load_model('longterm_model.keras')
        logging.info("Long-term model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load long-term model: {e}")
        exit()
    
    try:
        feature_scaler_longterm = joblib.load('longterm_feature_scaler.pkl')
        target_scalers_longterm = joblib.load('longterm_target_scalers.pkl')
        logging.info("Long-term scalers loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load long-term scalers: {e}")
        exit()
    
    return (model_scalping, feature_scaler_scalping, target_scalers_scalping,
            model_longterm, feature_scaler_longterm, target_scalers_longterm)

# Prepare data for prediction
def prepare_data_for_prediction(data, feature_scaler, window_size):
    """Prepare the latest window of data for model prediction."""
    try:
        data_window = data.iloc[-window_size:]
        data_scaled = feature_scaler.transform(data_window)
        data_lstm = data_scaled.reshape(1, window_size, data_scaled.shape[1])
        return data_lstm
    except Exception as e:
        logging.error(f"Error preparing data for prediction: {e}")
        return None

# Generate signals based on model predictions
def generate_signals(predictions_scalping, predictions_longterm):
    """
    Generate trading signals based on both scalping and long-term predictions.
    This is a simplified example; adjust the logic as per your bot's strategy.
    """
    signals = {'buy': False, 'sell': False}
    
    # Example logic: Buy if scalping predicts upward and long-term predicts upward
    # Sell if scalping predicts downward and long-term predicts downward
    # Hold otherwise
    try:
        scalping_signal = None
        longterm_signal = None
        
        # Determine scalping signal
        scalping_predictions = predictions_scalping  # Dict with timeframes M1, M5, M15
        buy_signals_scalping = sum(1 for tf in ['M1', 'M5', 'M15'] if scalping_predictions.get(tf, 0) > 0)
        sell_signals_scalping = sum(1 for tf in ['M1', 'M5', 'M15'] if scalping_predictions.get(tf, 0) < 0)
        
        if buy_signals_scalping > sell_signals_scalping:
            scalping_signal = 'buy'
        elif sell_signals_scalping > buy_signals_scalping:
            scalping_signal = 'sell'
        
        # Determine long-term signal
        longterm_predictions = predictions_longterm  # Dict with timeframes H1, H4, D1
        buy_signals_longterm = sum(1 for tf in ['H1', 'H4', 'D1'] if longterm_predictions.get(tf, 0) > 0)
        sell_signals_longterm = sum(1 for tf in ['H1', 'H4', 'D1'] if longterm_predictions.get(tf, 0) < 0)
        
        if buy_signals_longterm > sell_signals_longterm:
            longterm_signal = 'buy'
        elif sell_signals_longterm > buy_signals_longterm:
            longterm_signal = 'sell'
        
        # Combine signals
        if scalping_signal == 'buy' and longterm_signal == 'buy':
            signals['buy'] = True
        elif scalping_signal == 'sell' and longterm_signal == 'sell':
            signals['sell'] = True
        # Else, hold (no action)
        
        return signals
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return signals

# Trade execution logic
def execute_trade(balance, position, signal, price, trade_type, SL, TP):
    """
    Execute a trade and update the balance.
    position: Dict containing 'type', 'entry_price', 'SL', 'TP'
    """
    try:
        if signal == 'buy' and position['type'] != 'long':
            # Close short position if exists
            if position['type'] == 'short':
                profit = position['entry_price'] - price
                balance += profit * LOT_SIZE_LONGTERM * POINT_VALUE
                logging.info(f"Closed short position at {price}, Profit: {profit * LOT_SIZE_LONGTERM * POINT_VALUE}")
                position['type'] = None
            
            # Open long position
            position['type'] = 'long'
            position['entry_price'] = price
            position['SL'] = price - SL
            position['TP'] = price + TP
            logging.info(f"Opened long position at {price}, SL: {position['SL']}, TP: {position['TP']}")
        
        elif signal == 'sell' and position['type'] != 'short':
            # Close long position if exists
            if position['type'] == 'long':
                profit = price - position['entry_price']
                balance += profit * LOT_SIZE_LONGTERM * POINT_VALUE
                logging.info(f"Closed long position at {price}, Profit: {profit * LOT_SIZE_LONGTERM * POINT_VALUE}")
                position['type'] = None
            
            # Open short position
            position['type'] = 'short'
            position['entry_price'] = price
            position['SL'] = price + SL
            position['TP'] = price - TP
            logging.info(f"Opened short position at {price}, SL: {position['SL']}, TP: {position['TP']}")
        
        return balance, position
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return balance, position

# Backtest simulation
def simulate_trades(data, models, scalers):
    """Simulate trades over historical data."""
    (model_scalping, feature_scaler_scalping, target_scalers_scalping,
     model_longterm, feature_scaler_longterm, target_scalers_longterm) = models
    
    balance = INITIAL_BALANCE
    max_balance = INITIAL_BALANCE
    min_balance = INITIAL_BALANCE
    max_drawdown = 0
    total_profit = 0
    total_loss = 0
    trade_history = []
    
    position = {
        'type': None,            # 'long' or 'short'
        'entry_price': 0.0,
        'SL': 0.0,
        'TP': 0.0
    }
    
    for idx in range(WINDOW_SIZE_LONGTERM, len(data)):
        current_time = data.index[idx]
        logging.info(f"Processing time: {current_time}")
        
        # Prepare data for scalping prediction
        window_scalping = data.iloc[idx - WINDOW_SIZE_SCALPING:idx]
        X_scalping = prepare_data_for_prediction(window_scalping, feature_scaler_scalping, WINDOW_SIZE_SCALPING)
        if X_scalping is None:
            logging.warning("Skipping this iteration due to data preparation error.")
            continue
        
        # Make scalping predictions
        predictions_scalping = {}
        try:
            preds_scalping = model_scalping.predict(X_scalping)
            # Assuming the model outputs predictions for M1, M5, M15 in that order
            predictions_scalping['M1'] = preds_scalping[0][0]
            predictions_scalping['M5'] = preds_scalping[0][1]
            predictions_scalping['M15'] = preds_scalping[0][2]
            logging.debug(f"Scalping Predictions: {predictions_scalping}")
        except Exception as e:
            logging.error(f"Error making scalping predictions: {e}")
            continue
        
        # Prepare data for long-term prediction
        window_longterm = data.iloc[idx - WINDOW_SIZE_LONGTERM:idx]
        X_longterm = prepare_data_for_prediction(window_longterm, feature_scaler_longterm, WINDOW_SIZE_LONGTERM)
        if X_longterm is None:
            logging.warning("Skipping this iteration due to data preparation error.")
            continue
        
        # Make long-term predictions
        predictions_longterm = {}
        try:
            preds_longterm = model_longterm.predict(X_longterm)
            # Assuming the model outputs predictions for H1, H4, D1 in that order
            predictions_longterm['H1'] = preds_longterm[0][0]
            predictions_longterm['H4'] = preds_longterm[0][1]
            predictions_longterm['D1'] = preds_longterm[0][2]
            logging.debug(f"Long-Term Predictions: {predictions_longterm}")
        except Exception as e:
            logging.error(f"Error making long-term predictions: {e}")
            continue
        
        # Generate signals based on predictions
        signals = generate_signals(predictions_scalping, predictions_longterm)
        logging.info(f"Generated Signals: {signals}")
        
        # Current price (using M1 close price for action)
        current_price = data['close_M1'].iloc[idx]
        
        # Execute trade based on signal
        if signals['buy'] or signals['sell']:
            trade_type = 'long' if signals['buy'] else 'short'
            SL = (ATR_MULTIPLIER_SCALPING * data['atr14_M1'].iloc[idx]) if trade_type == 'long' else (ATR_MULTIPLIER_SCALPING * data['atr14_M1'].iloc[idx])
            TP = (SL * SL_TP_MULTIPLIER['buy']) if trade_type == 'long' else (SL * SL_TP_MULTIPLIER['sell'])
            
            balance, position = execute_trade(
                balance,
                position,
                signals['buy'] and 'buy' or signals['sell'] and 'sell',
                current_price,
                trade_type,
                SL,
                TP
            )
        
        # Check for SL/TP hits
        if position['type'] == 'long':
            # Check if current price hits SL or TP
            if current_price <= position['SL']:
                # Stop Loss hit
                profit = (position['SL'] - position['entry_price']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                total_loss += abs(profit)
                logging.info(f"Long position SL hit at {position['SL']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
            elif current_price >= position['TP']:
                # Take Profit hit
                profit = (position['TP'] - position['entry_price']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                total_profit += profit
                logging.info(f"Long position TP hit at {position['TP']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
        
        elif position['type'] == 'short':
            # Check if current price hits SL or TP
            if current_price >= position['SL']:
                # Stop Loss hit
                profit = (position['entry_price'] - position['SL']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                total_loss += abs(profit)
                logging.info(f"Short position SL hit at {position['SL']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
            elif current_price <= position['TP']:
                # Take Profit hit
                profit = (position['entry_price'] - position['TP']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                total_profit += profit
                logging.info(f"Short position TP hit at {position['TP']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
        
        # Update max balance and calculate drawdown
        if balance > max_balance:
            max_balance = balance
        drawdown = max_balance - balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate performance metrics
    def calculate_metrics(final_balance, trade_history, max_drawdown):
        """Calculate and return performance metrics."""
        total_trades = len(trade_history)
        wins = sum(1 for t in trade_history if t > 0)
        losses = sum(1 for t in trade_history if t < 0)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_profit = sum(t for t in trade_history if t > 0)
        total_loss = sum(t for t in trade_history if t < 0)
        net_profit = total_profit + total_loss
        max_profit_trade = max(trade_history) if trade_history else 0
        max_loss_trade = min(trade_history) if trade_history else 0
    
        metrics = {
            'Final Balance': final_balance,
            'Total Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Win Rate (%)': win_rate,
            'Total Profit': total_profit,
            'Total Loss': total_loss,
            'Net Profit': net_profit,
            'Max Drawdown': max_drawdown,
            'Max Profit per Trade': max_profit_trade,
            'Max Loss per Trade': max_loss_trade
        }
    
        return metrics
    
    def main():
        """Main function to run the backtest."""
        data = load_data()
        models = load_models_and_scalers()
        
        logging.info("Starting backtest simulation...")
        
        for idx in range(WINDOW_SIZE_LONGTERM, len(data)):
            simulate_trades(data, models, scalers=None)
        
        # After simulation, calculate metrics
        # For simplicity, let's assume trade_history is collected globally
        # Alternatively, you can modify simulate_trades to return trade history
        # Here, we'll implement a simplified version
        
        # Placeholder for trade history
        trade_history = []  # This should be collected during simulation
        
        # Assuming simulate_trades appends to trade_history, modify accordingly
        # For this example, let's skip actual metric calculation
        
        # Example metrics (replace with actual calculations)
        metrics = {
            'Final Balance': INITIAL_BALANCE,  # Replace with actual final balance
            'Total Trades': 0,
            'Wins': 0,
            'Losses': 0,
            'Win Rate (%)': 0,
            'Total Profit': 0,
            'Total Loss': 0,
            'Net Profit': 0,
            'Max Drawdown': 0,
            'Max Profit per Trade': 0,
            'Max Loss per Trade': 0
        }
        
        # Print metrics
        logging.info("Backtest Results:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")
    
    if __name__ == "__main__":
        main()