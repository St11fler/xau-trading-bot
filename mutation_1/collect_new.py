# collect_new.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
import sys
from functools import reduce

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("collect.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def initialize_mt5():
    """
    Инициализира връзка с MetaTrader 5.
    """
    if not mt5.initialize():
        logging.error("Failed to initialize MT5.")
        mt5.shutdown()
        sys.exit()

def shutdown_mt5():
    """
    Затваря връзката с MetaTrader 5.
    """
    mt5.shutdown()

def get_timeframes():
    """
    Определя кои времеви рамки ще извличаме.
    """
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
    Събира исторически данни за дадения символ и указани времеви рамки, 
    изчислява базови технически индикатори и връща речник {tf_name: DataFrame}.
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
                logging.warning(
                    f"No data from {from_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
                    f"for timeframe {tf_name}. Attempt {attempts + 1}/{max_attempts}."
                )
                from_date += timedelta(days=30)
                attempts += 1
            else:
                data_retrieved = True
                logging.info(
                    f"Data retrieved for {tf_name} from {from_date.strftime('%Y-%m-%d')} "
                    f"to {end_date.strftime('%Y-%m-%d')}."
                )

        if not data_retrieved:
            logging.error(f"Max attempts reached for {tf_name}. No data retrieved. Skipping.")
            continue

        # Превръщаме в DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.drop_duplicates(subset='time', inplace=True)
        df.dropna(inplace=True)

        # Връзваме tick_volume към volume
        df['volume'] = df['tick_volume']

        # SMA, EMA (примерно)
        df['sma5'] = ta.sma(df['close'], length=5)
        df['sma10'] = ta.sma(df['close'], length=10)
        df['ema5'] = ta.ema(df['close'], length=5)
        df['ema10'] = ta.ema(df['close'], length=10)

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)

        # RSI
        df['rsi14'] = ta.rsi(df['close'], length=14)

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)

        # CCI
        df['cci20'] = ta.cci(df['high'], df['low'], df['close'], length=20)

        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            df['adx14'] = adx['ADX_14']

        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)

        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])

        # Momentum
        df['momentum'] = ta.mom(df['close'], length=10)

        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # ATR
        df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Почистваме NaN след изчисления
        df.dropna(inplace=True)

        # Слагаме 'time' като индекс
        df.set_index('time', inplace=True)

        # Преименуваме колоните, за да имат суфикс на времевата рамка
        df.columns = [f"{col}_{tf_name}" for col in df.columns]
        data_frames[tf_name] = df
        logging.info(f"Indicators calculated for timeframe {tf_name}.")

    return data_frames

def merge_data(data_frames):
    """
    Обединява всички DataFrame-ове по индекс (време).
    """
    if not data_frames:
        logging.error("No data frames to merge.")
        return None
    try:
        dfs = list(data_frames.values())
        data_merged = reduce(lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how='outer'), dfs)
        data_merged.sort_index(inplace=True)
        data_merged.ffill(inplace=True)
        data_merged.dropna(inplace=True)
        logging.info("Data from all timeframes merged successfully.")
        return data_merged
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return None

def save_data(data, filename='XAUUSD_data_multiple_timeframes.csv'):
    """
    Записва финалния DataFrame във CSV.
    """
    try:
        data.to_csv(filename)
        logging.info(f"Data saved to '{filename}'.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def main():
    """
    Събира данните, смята индикатори и запазва резултата в CSV файл.
    """
    symbol = "XAUUSD"
    timeframes = get_timeframes()
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()
    filename = 'XAUUSD_data_multiple_timeframes.csv'

    initialize_mt5()
    data_frames = collect_data(symbol, timeframes, start_date, end_date, max_attempts=50)
    shutdown_mt5()

    if not data_frames:
        logging.error("No data was collected for any timeframe. Exiting.")
        sys.exit()

    data_merged = merge_data(data_frames)
    if data_merged is not None:
        save_data(data_merged, filename)
        logging.info("Data collection and feature engineering complete.")
    else:
        logging.error("Data merging failed. Exiting.")

if __name__ == "__main__":
    main()
