import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import time
import logging
import sys
from decimal import Decimal, ROUND_HALF_UP
import random

# Define filling mode bitmask constants
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2
SYMBOL_FILLING_RETURN = 4

# Order filling types
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout)
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
        "M1": 1.3907,
        "M5": 1.4307,
        "M15": 1.5205,
        "H1": 2.6945,
        "H4": 3.0554,
        "D1": 5.4452
    },
    "SL_TP_MULTIPLIER": {
        "buy": 1.5,
        "sell": 1.5
    },
    "TRAILING_STOP_FACTOR": 0.5  # Trailing stop as fraction of ATR
}

SYMBOL = "XAUUSD"
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1
}

WINDOW_SIZE_SCALPING = 7
WINDOW_SIZE_LONGTERM = 37
LOT_SIZE_SCALPING = 0.03
LOT_SIZE_LONGTERM = 0.02
MAGIC_NUMBER_SCALPING = 123456
MAGIC_NUMBER_LONGTERM = 654321
MAX_SLIPPAGE = 10
INTERVAL = 5  # Reduced frequency of checks

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
TRAILING_STOP_FACTOR = CONFIG["TRAILING_STOP_FACTOR"]

last_longterm_trade_time = None

def load_models():
    """Load trained models, scalers, and the action classifier."""
    try:
        model_scalping = load_model('scalping_model.keras')
        feature_scaler_scalping = joblib.load('scalping_feature_scaler.pkl')
        target_scalers_scalping = joblib.load('scalping_target_scalers.pkl')
        model_longterm = load_model('longterm_model.keras')
        feature_scaler_longterm = joblib.load('longterm_feature_scaler.pkl')
        target_scalers_longterm = joblib.load('longterm_target_scalers.pkl')
        clf, action_features = joblib.load('action_classifier.pkl')
        logging.info("All models and scalers loaded successfully.")
        return (model_scalping, feature_scaler_scalping, target_scalers_scalping,
                model_longterm, feature_scaler_longterm, target_scalers_longterm,
                clf, action_features)
    except Exception as e:
        logging.error(f"Failed to load models or scalers: {e}")
        sys.exit()

def initialize_mt5_connection(max_retries=5, initial_delay=5):
    """Initialize MetaTrader5 connection with reconnection logic."""
    for attempt in range(max_retries):
        if mt5.initialize():
            account_info = mt5.account_info()
            if account_info is not None and account_info.trade_allowed:
                logging.info(f"Trading is allowed for account {account_info.login}")
                return True
            else:
                logging.error(f"Trading is not allowed for account {account_info.login}")
                mt5.shutdown()
                sys.exit()
        else:
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 0.1)
            logging.warning(f"MT5 initialization failed. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    
    logging.error("Max retries reached. Failed to initialize MT5.")
    mt5.shutdown()
    sys.exit()

def collect_data():
    """Collect recent historical data for all timeframes."""
    num_bars = 200
    data_frames = {}
    for tf_name, tf in TIMEFRAMES.items():
        logging.info(f"Collecting data for timeframe: {tf_name}")
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, num_bars)
        if rates is None or len(rates) == 0:
            logging.warning(f"No data for {tf_name}")
            continue
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('time', inplace=True)

        if 'tick_volume' in data.columns:
            data['volume'] = data['tick_volume']
        else:
            logging.error(f"No volume data available for timeframe {tf_name}.")
            continue

        try:
            bb = ta.bbands(data['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                bb.columns = [
                    f"bollinger_lower_{tf_name}",
                    f"bollinger_middle_{tf_name}",
                    f"bollinger_upper_{tf_name}",
                    f"bollinger_bandwidth_{tf_name}",
                    f"bollinger_pctB_{tf_name}"
                ]
                data = pd.concat([data, bb], axis=1)

            data[f"rsi14_{tf_name}"] = ta.rsi(data['close'], length=14)
            stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
            if stoch is not None and not stoch.empty:
                stoch.columns = [
                    f"STOCHk_14_3_3_{tf_name}",
                    f"STOCHd_14_3_3_{tf_name}"
                ]
                data = pd.concat([data, stoch], axis=1)

            data[f"cci20_{tf_name}"] = ta.cci(data['high'], data['low'], data['close'], length=20)
            adx = ta.adx(data['high'], data['low'], data['close'], length=14)
            if adx is not None and 'ADX_14' in adx.columns:
                data[f"adx14_{tf_name}"] = adx['ADX_14']

            data[f"williams_r_{tf_name}"] = ta.willr(data['high'], data['low'], data['close'], length=14)
            data[f"obv_{tf_name}"] = ta.obv(data['close'], data['volume'])
            data[f"momentum_{tf_name}"] = ta.mom(data['close'], length=10)

            macd = ta.macd(data['close'])
            if macd is not None and not macd.empty:
                macd.columns = [
                    f"MACD_12_26_9_{tf_name}",
                    f"MACDs_12_26_9_{tf_name}",
                    f"MACDh_12_26_9_{tf_name}"
                ]
                data = pd.concat([data, macd], axis=1)

            data[f"atr14_{tf_name}"] = ta.atr(data['high'], data['low'], data['close'], length=14)

            data[f"ema5_{tf_name}"] = ta.ema(data['close'], length=5)
            data[f"ema10_{tf_name}"] = ta.ema(data['close'], length=10)
            data[f"sma5_{tf_name}"] = ta.sma(data['close'], length=5)
            data[f"sma10_{tf_name}"] = ta.sma(data['close'], length=10)

            base_columns = ['open', 'high', 'low', 'close', 'volume']
            renamed_columns = {col: f"{col}_{tf_name}" for col in base_columns}
            data.rename(columns=renamed_columns, inplace=True)

            additional_columns = ['tick_volume', 'spread', 'real_volume']
            for col in additional_columns:
                if col in data.columns:
                    data.rename(columns={col: f"{col}_{tf_name}"}, inplace=True)

            data.dropna(inplace=True)
            logging.debug(f"Columns for timeframe {tf_name}: {data.columns.tolist()}")
            data_frames[tf_name] = data
            logging.info(f"Indicators calculated and renamed for timeframe {tf_name}.")
        except Exception as e:
            logging.error(f"Error calculating indicators for timeframe {tf_name}: {e}")
            continue

    if not data_frames:
        logging.error("No data frames to merge.")
        return None

    try:
        data_merged = pd.concat(data_frames.values(), axis=1)
        data_merged.sort_index(inplace=True)
        data_merged.ffill(inplace=True)
        data_merged.dropna(inplace=True)
        logging.info("Data from all timeframes merged successfully.")
        logging.debug(f"Columns in merged data: {data_merged.columns.tolist()}")
        return data_merged
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return None

def prepare_data_for_prediction(data_subset, feature_scaler, window_size, model_type='scalping'):
    """Prepare data window for prediction."""
    try:
        data_window = data_subset.iloc[-window_size:]
        if len(data_window) < window_size:
            logging.warning("Not enough data for prediction.")
            return None
        X_scaled = feature_scaler.transform(data_window)
        X_lstm = X_scaled.reshape(1, window_size, X_scaled.shape[1])
        return X_lstm
    except Exception as e:
        logging.error(f"Error preparing data for prediction: {e}")
        return None

def make_prediction(model, X_lstm, target_scalers, model_type='scalping'):
    """Make predictions using the trained model."""
    try:
        prediction_scaled = model.predict(X_lstm)
        logging.debug(f"{model_type.capitalize()} Scaled Prediction: {prediction_scaled}")
        predictions = {}
        timeframes = ['M1', 'M5', 'M15'] if model_type == 'scalping' else ['H1', 'H4', 'D1']
        
        for i, tf in enumerate(timeframes):
            scaler_key = f"close_{tf}"
            if scaler_key not in target_scalers:
                logging.error(f"Scaler for {tf} not found.")
                continue
            scaler = target_scalers[scaler_key]
            pred = scaler.inverse_transform(prediction_scaled[:, i].reshape(-1, 1))
            predictions[tf] = pred[0][0]
            logging.info(f"Predicted next close price for {tf}: {predictions[tf]}")
        return predictions
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None

def close_opposite_positions(signal, magic_number):
    """Close positions opposite to the current signal."""
    try:
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            logging.error(f"Failed to get positions: {mt5.last_error()}")
            return False
        if len(positions) == 0:
            logging.info("No open positions to close.")
            return True
        opposite_type = mt5.ORDER_TYPE_SELL if signal == 'buy' else mt5.ORDER_TYPE_BUY
        for pos in positions:
            if pos.type == opposite_type and pos.magic == magic_number:
                logging.info(f"Closing opposite position with ticket {pos.ticket}")
                close_req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(SYMBOL).bid if pos.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask,
                    "deviation": MAX_SLIPPAGE,
                    "magic": magic_number,
                    "comment": "Closing opposite position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": ORDER_FILLING_FOK
                }
                result = mt5.order_send(close_req)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Failed to close position {pos.ticket}, retcode: {result.retcode}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Error closing opposite positions: {e}")
        return False

def update_trailing_stop(position, current_price, atr, digits):
    """Update trailing stop based on ATR."""
    trailing_distance = TRAILING_STOP_FACTOR * atr
    if position['type'] == 'long':
        new_sl = current_price - trailing_distance
        if new_sl > position['SL'] and new_sl < current_price:
            position['SL'] = float(Decimal(str(new_sl)).quantize(Decimal(f'1.{"0" * digits}'), rounding=ROUND_HALF_UP))
            logging.info(f"Updated trailing stop for long position to {position['SL']}")
    elif position['type'] == 'short':
        new_sl = current_price + trailing_distance
        if new_sl < position['SL'] and new_sl > current_price:
            position['SL'] = float(Decimal(str(new_sl)).quantize(Decimal(f'1.{"0" * digits}'), rounding=ROUND_HALF_UP))
            logging.info(f"Updated trailing stop for short position to {position['SL']}")
    return position

def execute_trade(signal, predicted_price, latest_close_price, data_merged, trade_timeframe, magic_number, lot_size, atr_multiplier):
    """Execute a trade with volatility-based SL/TP and trailing stop."""
    try:
        if not close_opposite_positions(signal, magic_number):
            logging.error("Failed to close opposite positions.")
            return

        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            logging.error(f"{SYMBOL} not found.")
            return
        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                logging.error(f"Failed to select {SYMBOL}")
                return

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logging.error("Failed to get tick data.")
            return

        point = symbol_info.point
        digits = symbol_info.digits
        stop_level = symbol_info.trade_stops_level if symbol_info.trade_stops_level else 25
        min_stop_distance = stop_level * point

        if not (symbol_info.volume_min <= lot_size <= symbol_info.volume_max):
            logging.error(f"Lot size {lot_size} is out of allowed range for {SYMBOL}")
            return

        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to retrieve account information.")
            return

        if signal == 'buy':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        required_margin = mt5.order_calc_margin(order_type, SYMBOL, lot_size, price)
        if required_margin is None or account_info.margin_free < required_margin:
            logging.error("Not enough free margin to execute the trade.")
            return

        atr_value = data_merged[f'atr14_{trade_timeframe}'].iloc[-1]
        logging.info(f"ATR ({trade_timeframe}): {atr_value} ({atr_value / point} points)")

        mae = MAE_VALUES.get(trade_timeframe, None)
        if mae is None:
            logging.error(f"No MAE value defined for timeframe {trade_timeframe}.")
            return

        sl_tp_multiplier = SL_TP_MULTIPLIER.get(signal, 1.0)

        if USE_ATR_SL_TP:
            volatility_factor = 1.0 + (atr_value / latest_close_price)  # Adjust SL/TP based on volatility
            if signal == 'buy':
                sl = price - (atr_multiplier * atr_value * volatility_factor)
                tp = predicted_price + (sl_tp_multiplier * atr_multiplier * atr_value * volatility_factor)
                if tp <= price or sl >= price:
                    logging.warning("Invalid TP or SL for buy order.")
                    return
            else:
                sl = price + (atr_multiplier * atr_value * volatility_factor)
                tp = predicted_price - (sl_tp_multiplier * atr_multiplier * atr_value * volatility_factor)
                if tp >= price or sl <= price:
                    logging.warning("Invalid TP or SL for sell order.")
                    return
        else:
            if signal == 'buy':
                sl = price - FIXED_SL_SCALPING - (sl_tp_multiplier * mae)
                tp = price + FIXED_TP_SCALPING + (sl_tp_multiplier * mae)
                if tp <= price or sl >= price:
                    logging.warning("Invalid TP or SL for buy order.")
                    return
            else:
                sl = price + FIXED_SL_SCALPING + (sl_tp_multiplier * mae)
                tp = price - FIXED_TP_SCALPING - (sl_tp_multiplier * mae)
                if tp >= price or sl <= price:
                    logging.warning("Invalid TP or SL for sell order.")
                    return

        sl_distance_points = abs(price - sl) / point
        tp_distance_points = abs(tp - price) / point

        if sl_distance_points < stop_level:
            sl = price - (stop_level * point) if order_type == mt5.ORDER_TYPE_BUY else price + (stop_level * point)
            logging.warning(f"Adjusted SL to meet minimum stop level: {sl}")

        if tp_distance_points < stop_level:
            tp = price + (stop_level * point) if order_type == mt5.ORDER_TYPE_BUY else price - (stop_level * point)
            logging.warning(f"Adjusted TP to meet minimum stop level: {tp}")

        sl = float(Decimal(str(sl)).quantize(Decimal(f'1.{"0" * digits}'), rounding=ROUND_HALF_UP))
        tp = float(Decimal(str(tp)).quantize(Decimal(f'1.{"0" * digits}'), rounding=ROUND_HALF_UP))
        price = float(Decimal(str(price)).quantize(Decimal(f'1.{"0" * digits}'), rounding=ROUND_HALF_UP))

        if sl <= 0 or tp <= 0:
            logging.error("Invalid SL or TP after adjustments.")
            return

        filling_mode = symbol_info.filling_mode
        if filling_mode & SYMBOL_FILLING_FOK:
            filling_type = ORDER_FILLING_FOK
        elif filling_mode & SYMBOL_FILLING_IOC:
            filling_type = ORDER_FILLING_IOC
        elif filling_mode & SYMBOL_FILLING_RETURN:
            filling_type = ORDER_FILLING_RETURN
        else:
            logging.error("No acceptable filling type found for symbol.")
            return

        logging.info(f"Determined filling type: {filling_type}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": MAX_SLIPPAGE,
            "magic": magic_number,
            "comment": f"Python script open - {trade_timeframe}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        logging.info(f"Order Request: {request}")

        try:
            result = mt5.order_send(request)
            if result is None:
                logging.error("Order send failed: mt5.order_send() returned None")
                error_code, error_description = mt5.last_error()
                logging.error(f"Error code: {error_code}, description: {error_description}")
                return
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order send failed, retcode={result.retcode}")
                logging.error(f"Result: {result}")
            else:
                logging.info(f"Order executed successfully, ticket={result.order}")
        except Exception as e:
            logging.error(f"Exception occurred while sending order: {e}")
            return
    except Exception as e:
        logging.error(f"Exception in execute_trade: {e}")
        return

def main():
    """Main function to run the trading bot."""
    logging.info("Starting the trading bot.")
    (model_scalping, feature_scaling, target_scaling,
     model_longterm, feature_longterm, target_longterm,
     clf, action_features) = load_models()

    logging.info("All models and scalers loaded successfully.")

    initialize_mt5_connection()
    logging.info("MetaTrader5 connection initialized.")

    open_position = {'type': None, 'entry_price': 0.0, 'SL': 0.0, 'TP': 0.0}

    while True:
        try:
            logging.info("\n--- New Iteration ---")
            logging.info(f"Time: {datetime.now()}")

            data_merged = collect_data()
            if data_merged is None:
                logging.warning("Data collection failed. Waiting before next iteration.")
                time.sleep(INTERVAL)
                continue

            scalping_columns = list(feature_scaling.feature_names_in_)
            longterm_columns = list(feature_longterm.feature_names_in_)

            missing_scalping = set(scalping_columns) - set(data_merged.columns)
            if missing_scalping:
                logging.error(f"Scalping missing features: {missing_scalping}")
                time.sleep(INTERVAL)
                continue

            missing_longterm = set(longterm_columns) - set(data_merged.columns)
            if missing_longterm:
                logging.error(f"Long-term missing features: {missing_longterm}")
                time.sleep(INTERVAL)
                continue

            data_for_scalping = data_merged[scalping_columns]
            data_for_longterm = data_merged[longterm_columns]

            X_new_scalping = prepare_data_for_prediction(
                data_for_scalping,
                feature_scaling,
                WINDOW_SIZE_SCALPING,
                model_type='scalping'
            )
            if X_new_scalping is None:
                logging.warning("Scalping data preparation failed. Waiting before next iteration.")
                time.sleep(INTERVAL)
                continue

            predictions_scalping = make_prediction(
                model_scalping,
                X_new_scalping,
                target_scaling,
                model_type='scalping'
            )
            if predictions_scalping is None:
                logging.warning("Scalping prediction failed. Waiting before next iteration.")
                time.sleep(INTERVAL)
                continue

            X_new_longterm = prepare_data_for_prediction(
                data_for_longterm,
                feature_longterm,
                WINDOW_SIZE_LONGTERM,
                model_type='longterm'
            )
            if X_new_longterm is None:
                logging.warning("Long-term data preparation failed. Waiting before next iteration.")
                time.sleep(INTERVAL)
                continue

            predictions_longterm = make_prediction(
                model_longterm,
                X_new_longterm,
                target_longterm,
                model_type='longterm'
            )
            if predictions_longterm is None:
                logging.warning("Long-term prediction failed. Waiting before next iteration.")
                time.sleep(INTERVAL)
                continue

            current_data = data_merged.iloc[-1].copy()
            current_data['pred_close_M1'] = predictions_scalping['M1']
            current_data['pred_close_M5'] = predictions_scalping['M5']
            current_data['pred_close_M15'] = predictions_scalping['M15']

            if not all(feat in current_data.index for feat in action_features):
                logging.error("Some action features are missing from current_data.")
                time.sleep(INTERVAL)
                continue

            X_action = current_data[action_features].values.reshape(1, -1)
            action = clf.predict(X_action)[0]
            if action == 1:
                signal = 'buy'
            elif action == -1:
                signal = 'sell'
            else:
                logging.warning("Received an undefined action. Skipping trade.")
                time.sleep(INTERVAL)
                continue

            logging.info(f"Classifier decided action: {action} -> signal: {signal}")

            current_price = data_merged['close_M1'].iloc[-1]
            atr = data_merged['atr14_M1'].iloc[-1]

            # Update trailing stop for open position
            if open_position['type'] is not None:
                open_position = update_trailing_stop(open_position, current_price, atr, mt5.symbol_info(SYMBOL).digits)

            if signal in ['buy', 'sell']:
                magic_number = MAGIC_NUMBER_SCALPING
                lot_size = LOT_SIZE_SCALPING
                atr_multiplier = ATR reconnaissant_MULTIPLIER_SCALPING
                trade_timeframe = 'M1'
                execute_trade(
                    signal=signal,
                    predicted_price=predictions_scalping['M1'],
                    latest_close_price=current_price,
                    data_merged=data_merged,
                    trade_timeframe=trade_timeframe,
                    magic_number=magic_number,
                    lot_size=lot_size,
                    atr_multiplier=atr_multiplier
                )

            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            logging.info("Stopping the bot.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()