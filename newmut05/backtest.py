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
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    "USE_ATR_SL_TP": True,
    "ATR_MULTIPLIER_SCALPING": 1.0,
    "ATR_MULTIPLIER_LONGTERM": 2.0,
    "FIXED_SL_SCALPING": 10 * 0.1,
    "FIXED_TP_SCALPING": 20 * 0.1,
    "FIXED_SL_LONGTERM": 50 * 0.1,
    "FIXED_TP_LONGTERM": 100 * 0.1,
    "LONGTERM_COOLDOWN_HOURS": 24,
    "MAE_VALUES": {
        "M1": 2.38,
        "M5": 2.38,
        "M15": 2.58,
        "H1": 3.03,
        "H4": 5.31,
        "D1": 23.20
    },
    "SL_TP_MULTIPLIER": {
        "buy": 1.5,
        "sell": 1.5
    }
}

INITIAL_BALANCE = 500.0
LOT_SIZE_SCALPING = 0.01
LOT_SIZE_LONGTERM = 0.02
POINT_VALUE = 0.1
WINDOW_SIZE_SCALPING = 7
WINDOW_SIZE_LONGTERM = 37
SLIPPAGE_FACTOR = 0.1  # Slippage as a fraction of ATR

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

def load_models_and_scalers():
    """Load trained models and their corresponding scalers."""
    try:
        model_scalping = load_model('scalping_model.keras')
        feature_scaler_scalping = joblib.load('scalping_feature_scaler.pkl')
        target_scalers_scalping = joblib.load('scalping_target_scalers.pkl')
        model_longterm = load_model('longterm_model.keras')
        feature_scaler_longterm = joblib.load('longterm_feature_scaler.pkl')
        target_scalers_longterm = joblib.load('longterm_target_scalers.pkl')
        logging.info("Models and scalers loaded successfully.")
        return (model_scalping, feature_scaler_scalping, target_scalers_scalping,
                model_longterm, feature_scaler_longterm, target_scalers_longterm)
    except Exception as e:
        logging.error(f"Failed to load models or scalers: {e}")
        exit()

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

def generate_signals(predictions_scalping, predictions_longterm):
    """Generate trading signals based on predictions."""
    signals = {'buy': False, 'sell': False}
    try:
        scalping_signal = None
        longterm_signal = None
        buy_signals_scalping = sum(1 for tf in ['M1', 'M5', 'M15'] if predictions_scalping.get(tf, 0) > 0)
        sell_signals_scalping = sum(1 for tf in ['M1', 'M5', 'M15'] if predictions_scalping.get(tf, 0) < 0)
        
        if buy_signals_scalping > sell_signals_scalping:
            scalping_signal = 'buy'
        elif sell_signals_scalping > buy_signals_scalping:
            scalping_signal = 'sell'
        
        buy_signals_longterm = sum(1 for tf in ['H1', 'H4', 'D1'] if predictions_longterm.get(tf, 0) > 0)
        sell_signals_longterm = sum(1 for tf in ['H1', 'H4', 'D1'] if predictions_longterm.get(tf, 0) < 0)
        
        if buy_signals_longterm > sell_signals_longterm:
            longterm_signal = 'buy'
        elif sell_signals_longterm > buy_signals_longterm:
            longterm_signal = 'sell'
        
        if scalping_signal == 'buy' and longterm_signal == 'buy':
            signals['buy'] = True
        elif scalping_signal == 'sell' and longterm_signal == 'sell':
            signals['sell'] = True
        
        return signals
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return signals

def execute_trade(balance, position, signal, price, trade_type, SL, TP, atr):
    """Execute a trade with slippage simulation."""
    try:
        slippage = SLIPPAGE_FACTOR * atr
        price += slippage if signal == 'buy' else -slippage
        
        if signal == 'buy' and position['type'] != 'long':
            if position['type'] == 'short':
                profit = position['entry_price'] - price
                balance += profit * LOT_SIZE_LONGTERM * POINT_VALUE
                logging.info(f"Closed short position at {price}, Profit: {profit * LOT_SIZE_LONGTERM * POINT_VALUE}")
                position['type'] = None
            
            position['type'] = 'long'
            position['entry_price'] = price
            position['SL'] = price - SL
            position['TP'] = price + TP
            logging.info(f"Opened long position at {price}, SL: {position['SL']}, TP: {position['TP']}")
        
        elif signal == 'sell' and position['type'] != 'short':
            if position['type'] == 'long':
                profit = price - position['entry_price']
                balance += profit * LOT_SIZE_LONGTERM * POINT_VALUE
                logging.info(f"Closed long position at {price}, Profit: {profit * LOT_SIZE_LONGTERM * POINT_VALUE}")
                position['type'] = None
            
            position['type'] = 'short'
            position['entry_price'] = price
            position['SL'] = price + SL
            position['TP'] = price - TP
            logging.info(f"Opened short position at {price}, SL: {position['SL']}, TP: {position['TP']}")
        
        return balance, position
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return balance, position

def calculate_metrics(final_balance, trade_history, max_drawdown, returns):
    """Calculate advanced performance metrics."""
    total_trades = len(trade_history)
    wins = sum(1 for t in trade_history if t > 0)
    losses = sum(1 for t in trade_history if t < 0)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_profit = sum(t for t in trade_history if t > 0)
    total_loss = sum(t for t in trade_history if t < 0)
    net_profit = total_profit + total_loss
    max_profit_trade = max(trade_history) if trade_history else 0
    max_loss_trade = min(trade_history) if trade_history else 0
    
    # Calculate Sharpe and Sortino Ratios
    returns = np.array(returns)
    mean_return = np.mean(returns) * 252  # Annualized
    std_return = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 1
    sortino_ratio = mean_return / downside_std if downside_std != 0 else 0
    
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
        'Max Loss per Trade': max_loss_trade,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio
    }
    
    return metrics

def simulate_trades(data, models):
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
    returns = []
    
    position = {
        'type': None,
        'entry_price': 0.0,
        'SL': 0.0,
        'TP': 0.0
    }
    
    for idx in range(WINDOW_SIZE_LONGTERM, len(data)):
        current_time = data.index[idx]
        logging.info(f"Processing time: {current_time}")
        
        window_scalping = data.iloc[idx - WINDOW_SIZE_SCALPING:idx]
        X_scalping = prepare_data_for_prediction(window_scalping, feature_scaler_scalping, WINDOW_SIZE_SCALPING)
        if X_scalping is None:
            logging.warning("Skipping this iteration due to data preparation error.")
            continue
        
        predictions_scalping = {}
        try:
            preds_scalping = model_scalping.predict(X_scalping)
            predictions_scalping['M1'] = preds_scalping[0][0]
            predictions_scalping['M5'] = preds_scalping[0][1]
            predictions_scalping['M15'] = preds_scalping[0][2]
            logging.debug(f"Scalping Predictions: {predictions_scalping}")
        except Exception as e:
            logging.error(f"Error making scalping predictions: {e}")
            continue
        
        window_longterm = data.iloc[idx - WINDOW_SIZE_LONGTERM:idx]
        X_longterm = prepare_data_for_prediction(window_longterm, feature_scaler_longterm, WINDOW_SIZE_LONGTERM)
        if X_longterm is None:
            logging.warning("Skipping this iteration due to data preparation error.")
            continue
        
        predictions_longterm = {}
        try:
            preds_longterm = model_longterm.predict(X_longterm)
            predictions_longterm['H1'] = preds_longterm[0][0]
            predictions_longterm['H4'] = preds_longterm[0][1]
            predictions_longterm['D1'] = preds_longterm[0][2]
            logging.debug(f"Long-Term Predictions: {predictions_longterm}")
        except Exception as e:
            logging.error(f"Error making long-term predictions: {e}")
            continue
        
        signals = generate_signals(predictions_scalping, predictions_longterm)
        logging.info(f"Generated Signals: {signals}")
        
        current_price = data['close_M1'].iloc[idx]
        atr = data['atr14_M1'].iloc[idx]
        
        if signals['buy'] or signals['sell']:
            trade_type = 'long' if signals['buy'] else 'short'
            SL = (ATR_MULTIPLIER_SCALPING * atr) if trade_type == 'long' else (ATR_MULTIPLIER_SCALPING * atr)
            TP = (SL * SL_TP_MULTIPLIER['buy']) if trade_type == 'long' else (SL * SL_TP_MULTIPLIER['sell'])
            
            balance, position = execute_trade(
                balance,
                position,
                signals['buy'] and 'buy' or signals['sell'] and 'sell',
                current_price,
                trade_type,
                SL,
                TP,
                atr
            )
        
        if position['type'] == 'long':
            if current_price <= position['SL']:
                profit = (position['SL'] - position['entry_price']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                returns.append(profit / balance)
                total_loss += abs(profit) if profit < 0 else 0
                logging.info(f"Long position SL hit at {position['SL']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
            elif current_price >= position['TP']:
                profit = (position['TP'] - position['entry_price']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                returns.append(profit / balance)
                total_profit += profit if profit > 0 else 0
                logging.info(f"Long position TP hit at {position['TP']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
        
        elif position['type'] == 'short':
            if current_price >= position['SL']:
                profit = (position['entry_price'] - position['SL']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                returns.append(profit / balance)
                total_loss += abs(profit) if profit < 0 else 0
                logging.info(f"Short position SL hit at {position['SL']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
            elif current_price <= position['TP']:
                profit = (position['entry_price'] - position['TP']) * LOT_SIZE_LONGTERM * POINT_VALUE
                balance += profit
                trade_history.append(profit)
                returns.append(profit / balance)
                total_profit += profit if profit > 0 else 0
                logging.info(f"Short position TP hit at {position['TP']}, Profit: {profit}")
                position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}
        
        if balance > max_balance:
            max_balance = balance
        drawdown = max_balance - balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return balance, trade_history, max_drawdown, returns

def main():
    """Main function to run the backtest."""
    data = load_data()
    models = load_models_and_scalers()
    
    logging.info("Starting backtest simulation...")
    final_balance, trade_history, max_drawdown, returns = simulate_trades(data, models)
    
    metrics = calculate_metrics(final_balance, trade_history, max_drawdown, returns)
    
    logging.info("Backtest Results:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

if __name__ == "__main__":
    main()