# collect.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
import sys
from functools import reduce

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("collect.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def initialize_mt5():
    """Initialize MetaTrader5 connection."""
    if not mt5.initialize():
        logging.error("Failed to initialize MT5.")
        mt5.shutdown()
        sys.exit()

def shutdown_mt5():
    """Shutdown MetaTrader5 connection."""
    mt5.shutdown()

def get_timeframes():
    """Define the timeframes to collect data for."""
    return {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }

def collect_data(symbol, timeframes, start_date, end_date, max_attempts=50):
    """
    Collect historical data for the specified symbol and timeframes.
    
    Attempts to retrieve data starting from `start_date` to `end_date`.
    If no data is found, it moves the start date forward by 30 days and tries again, 
    up to `max_attempts`.
    """
    data_frames = {}
    for tf_name, tf in timeframes.items():
        logging.info(f"Collecting data for timeframe: {tf_name}")
        from_date = start_date
        attempts = 0
        data_retrieved = False

        while not data_retrieved and attempts < max_attempts:
            rates = mt5.copy_rates_range(symbol, tf, from_date, end_date)

            if rates is None or len(rates) == 0:
                logging.warning(f"No data from {from_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
                                f"for timeframe {tf_name}. Attempt {attempts + 1}/{max_attempts}.")
                from_date += timedelta(days=30)
                attempts += 1
            else:
                data_retrieved = True
                logging.info(f"Data retrieved for {tf_name} from {from_date.strftime('%Y-%m-%d')} "
                             f"to {end_date.strftime('%Y-%m-%d')}.")
        
        if not data_retrieved:
            logging.error(f"Max attempts reached for {tf_name}. No data retrieved. Skipping this timeframe.")
            continue  # Skip this timeframe if no data

        # Process the retrieved data
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.drop_duplicates(subset='time', inplace=True)
        data.dropna(inplace=True)

        # Feature Engineering: Calculate Technical Indicators
        try:
            # Use 'tick_volume' as 'volume'
            data['volume'] = data['tick_volume']

            # Simple Moving Averages
            data['sma5'] = ta.sma(data['close'], length=5)
            data['sma10'] = ta.sma(data['close'], length=10)
            
            # Exponential Moving Averages
            data['ema5'] = ta.ema(data['close'], length=5)
            data['ema10'] = ta.ema(data['close'], length=10)
            
            # Bollinger Bands
            bb = ta.bbands(data['close'], length=20, std=2)
            if bb is not None:
                data = pd.concat([data, bb], axis=1)
            
            # RSI
            data['rsi14'] = ta.rsi(data['close'], length=14)
            
            # Stochastic Oscillator
            stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
            if stoch is not None:
                data = pd.concat([data, stoch], axis=1)
            
            # CCI
            data['cci20'] = ta.cci(data['high'], data['low'], data['close'], length=20)
            
            # ADX
            adx = ta.adx(data['high'], data['low'], data['close'], length=14)
            if adx is not None and 'ADX_14' in adx.columns:
                data['adx14'] = adx['ADX_14']
            
            # Williams %R
            data['williams_r'] = ta.willr(data['high'], data['low'], data['close'], length=14)
            
            # On-Balance Volume
            data['obv'] = ta.obv(data['close'], data['volume'])
            
            # Momentum
            data['momentum'] = ta.mom(data['close'], length=10)
            
            # MACD
            macd = ta.macd(data['close'])
            if macd is not None:
                data = pd.concat([data, macd], axis=1)
            
            # ATR
            data['atr14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            
            # Drop any NaN values after calculations
            data.dropna(inplace=True)
            
            # Set time as index
            data.set_index('time', inplace=True)
            
            # Rename columns to include timeframe suffix
            data.columns = [f"{col}_{tf_name}" for col in data.columns]
            
            # Store the DataFrame
            data_frames[tf_name] = data
            logging.info(f"Indicators calculated for timeframe {tf_name}.")
        
        except Exception as e:
            logging.error(f"Error calculating indicators for timeframe {tf_name}: {e}")
            continue

    return data_frames

def merge_data(data_frames):
    """Merge data from different timeframes into a single DataFrame."""
    if not data_frames:
        logging.error("No data frames to merge.")
        return None
    try:
        dfs = list(data_frames.values())
        data_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
        data_merged.sort_index(inplace=True)
        data_merged.ffill(inplace=True)
        data_merged.dropna(inplace=True)
        logging.info("Data from all timeframes merged successfully.")
        return data_merged
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return None

def save_data(data, filename='XAUUSD_data_multiple_timeframes.csv'):
    """Save the merged data to a CSV file."""
    try:
        data.to_csv(filename)
        logging.info(f"Data saved to '{filename}'.")
    except Exception as e:
        logging.error(f"Error saving data to '{filename}': {e}")

def main():
    """Main function to execute data collection and feature engineering."""
    symbol = "XAUUSD"
    timeframes = get_timeframes()
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()
    filename = 'XAUUSD_data_multiple_timeframes.csv'

    initialize_mt5()
    data_frames = collect_data(symbol, timeframes, start_date, end_date, max_attempts=50)
    shutdown_mt5()

    if not data_frames:
        logging.error("No data was collected for any timeframe. Exiting script.")
        sys.exit()

    data_merged = merge_data(data_frames)
    if data_merged is not None:
        save_data(data_merged, filename)
        logging.info("Data collection and indicator calculation complete.")
    else:
        logging.error("Data merging failed. Exiting script.")

if __name__ == "__main__":
    main()
