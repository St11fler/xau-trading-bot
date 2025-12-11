# bot_new.py
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # При желание махнете, ако дава Unicode проблеми в Windows
    ]
)

SYMBOL = "XAUUSD"
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
}
WINDOW_SIZE_SCALPING = 7

LOT_SIZE_SCALPING = 0.1
MAGIC_NUMBER_SCALPING = 123456
MAX_SLIPPAGE = 10
INTERVAL = 60  # секунди между итерациите


def load_models():
    """
    Зареждаме Transformer (скалпинг) и LightGBM класификатора.
    """
    try:
        model_scalping = load_model('scalping_model.keras')
        feature_scaler_scalping = joblib.load('scalping_feature_scaler.pkl')
        target_scalers_scalping = joblib.load('scalping_target_scalers.pkl')
        logging.info("Scalping model и скейлъри заредени успешно.")
    except Exception as e:
        logging.error(f"Failed to load scalping model: {e}")
        sys.exit()

    try:
        (clf, action_features) = joblib.load('action_classifier.pkl')
        logging.info("Action classifier (LightGBM) зареден успешно.")
    except Exception as e:
        logging.error(f"Failed to load action classifier: {e}")
        sys.exit()

    return model_scalping, feature_scaler_scalping, target_scalers_scalping, clf, action_features


def initialize_mt5_connection():
    """
    Инициализация на MT5, проверка за акаунт.
    """
    if not mt5.initialize():
        logging.error("MT5 initialize() failed")
        mt5.shutdown()
        sys.exit()

    account_info = mt5.account_info()
    if account_info is not None and account_info.trade_allowed:
        logging.info(f"Trading is allowed for account {account_info.login}")
    else:
        logging.error(f"Trading is NOT allowed for account {account_info.login}")
        mt5.shutdown()
        sys.exit()


def add_indicators(df, tf_suffix):
    """
    Добавяме нужните индикатори за даден TimeFrame (суфикс, напр. _M5).
    """
    if df.empty:
        return

    # SMA / EMA
    df['sma5' + tf_suffix] = ta.sma(df['close'], length=5)
    df['sma10' + tf_suffix] = ta.sma(df['close'], length=10)
    df['ema5' + tf_suffix] = ta.ema(df['close'], length=5)
    df['ema10' + tf_suffix] = ta.ema(df['close'], length=10)

    # RSI
    df['rsi14' + tf_suffix] = ta.rsi(df['close'], length=14)

    # CCI
    df['cci20' + tf_suffix] = ta.cci(df['high'], df['low'], df['close'], length=20)

    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx14' + tf_suffix] = adx['ADX_14']

    # ATR
    df['atr14' + tf_suffix] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Momentum
    df['momentum' + tf_suffix] = ta.mom(df['close'], length=10)

    # Williams %R
    df['williams_r' + tf_suffix] = ta.willr(df['high'], df['low'], df['close'], length=14)

    # OBV
    df['obv' + tf_suffix] = ta.obv(df['close'], df['tick_volume'])

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD_12_26_9' + tf_suffix] = macd['MACD_12_26_9']
    df['MACDh_12_26_9' + tf_suffix] = macd['MACDh_12_26_9']
    df['MACDs_12_26_9' + tf_suffix] = macd['MACDs_12_26_9']

    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['STOCHk_14_3_3' + tf_suffix] = stoch['STOCHk_14_3_3']
    df['STOCHd_14_3_3' + tf_suffix] = stoch['STOCHd_14_3_3']


def collect_data(num_bars=300):
    """
    Зареждаме данни за M1, M5, M15, добавяме индикатори и ги обединяваме в общ DataFrame.
    """
    data_dict = {}
    for tf_name, tf_val in TIMEFRAMES.items():
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_val, 0, num_bars)
        if rates is None or len(rates) == 0:
            logging.warning(f"No rates for {tf_name} timeframe.")
            continue
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Добавяме индикаторите
        add_indicators(df, '_' + tf_name)
        data_dict[tf_name] = df

    # Проверка дали имаме всичко
    if any(tf not in data_dict for tf in TIMEFRAMES.keys()):
        logging.warning("Missing some timeframes among M1, M5, M15.")
        return None

    df_m1 = data_dict['M1']
    df_m5 = data_dict['M5']
    df_m15 = data_dict['M15']

    # Преименуваме базовите колони (open,high,low,close...) за M5, M15, за да не се застъпват
    # (примерно close -> close_M5). Може да си ги тръгнете ако ви трябват.
    rename_m5 = {}
    for col in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
        if col in df_m5.columns:
            rename_m5[col] = col + "_M5"
    df_m5.rename(columns=rename_m5, inplace=True)

    rename_m15 = {}
    for col in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']:
        if col in df_m15.columns:
            rename_m15[col] = col + "_M15"
    df_m15.rename(columns=rename_m15, inplace=True)

    # Обединяваме в един DataFrame (по индекс-време)
    df_merged = df_m1.join(df_m5, how='outer').join(df_m15, how='outer')
    # Запълваме липсващите данни напред (forward fill)
    df_merged.ffill(inplace=True)
    # Премахваме всички редове, които все още съдържат NaN (ако има такива)
    df_merged.dropna(inplace=True)

    return df_merged


def prepare_data_for_scalping(df_merged, feature_scaler, window_size=7):
    """
    Правим извадка (последните window_size бара), проверяваме дали има 
    всички колони, които scaler-ът очаква, и връщаме X_scaled, latest_close_price.
    """
    if df_merged is None or df_merged.empty:
        return None

    if len(df_merged) < window_size:
        logging.warning(f"Not enough bars in merged DataFrame (need {window_size}).")
        return None

    data_window = df_merged.iloc[-window_size:].copy()

    needed_feats = set(feature_scaler.feature_names_in_)
    df_cols = set(data_window.columns)
    missing_feats = needed_feats - df_cols
    if missing_feats:
        logging.warning(f"Missing features in merged data: {missing_feats}")
        return None

    X_scaled = feature_scaler.transform(data_window[feature_scaler.feature_names_in_])
    # Нека за "latest_close_price" вземем реалната M1 close (която не сме преименували)
    latest_close_price = data_window.iloc[-1]['close']

    return (X_scaled.reshape(1, window_size, -1), latest_close_price)


def get_open_positions(symbol=SYMBOL, magic=MAGIC_NUMBER_SCALPING):
    """
    Връща списък с отворените позиции (buy/sell) за даден символ + magic.
    """
    all_positions = mt5.positions_get(symbol=symbol)
    if all_positions is None:
        return []

    my_positions = [pos for pos in all_positions if pos.magic == magic]
    return my_positions


def close_position(position):
    """
    Затваряме дадена позиция (позицията е обект от positions_get()).
    - Ако позицията е BUY, за да я затворим, пускаме SELL със същия обем.
    - Ако е SELL, пускаме BUY със същия обем.
    """
    symbol = position.symbol
    lot = position.volume

    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        trade_price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        trade_price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "position": position.ticket,
        "price": trade_price,
        "deviation": MAX_SLIPPAGE,
        "magic": position.magic,
        "comment": "Closing position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Failed to close position {position.ticket}, retcode={result.retcode}, comment={result.comment}")
    else:
        logging.info(f"Position {position.ticket} closed (deal={result.deal}).")


def execute_trade(action, symbol=SYMBOL, lot=LOT_SIZE_SCALPING):
    """
    Отваряме пазарна сделка (BUY или SELL), според action (1=BUY, -1=SELL).
    """
    if action == 1:
        order_type = mt5.ORDER_TYPE_BUY
        trade_price = mt5.symbol_info_tick(symbol).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        trade_price = mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": trade_price,
        "deviation": MAX_SLIPPAGE,
        "magic": MAGIC_NUMBER_SCALPING,
        "comment": "Scalping trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order send failed, retcode={result.retcode}, comment={result.comment}")
    else:
        logging.info(f"Trade opened successfully! order={result.order}, deal={result.deal}, volume={lot}")


def main():
    logging.info("Starting the updated trading bot...")

    # Зареждане на модели
    model_scalping, feature_scaling, target_scaling, clf, action_features = load_models()
    # Свързваме се с MT5
    initialize_mt5_connection()

    while True:
        try:
            logging.info("New iteration...")

            # 1) Събираме данни
            df_merged = collect_data(num_bars=300)
            if df_merged is None:
                logging.info("No data collected. Waiting...")
                time.sleep(INTERVAL)
                continue

            # 2) Подготвяме за скалпинг
            scalping_result = prepare_data_for_scalping(df_merged, feature_scaling, WINDOW_SIZE_SCALPING)
            if scalping_result is None:
                logging.info("Not enough data or missing features. Waiting...")
                time.sleep(INTERVAL)
                continue

            X_scaled, latest_close_price = scalping_result

            # 3) Прогноза от скалпинг модела (M1, M5, M15)
            preds_scaled = model_scalping.predict(X_scaled)

            # Дескалираме резултатите
            pred_close_M1 = target_scaling['close_M1'].inverse_transform(
                preds_scaled[:, 0].reshape(-1, 1)
            )[0][0]
            pred_close_M5 = target_scaling['close_M5'].inverse_transform(
                preds_scaled[:, 1].reshape(-1, 1)
            )[0][0]
            pred_close_M15 = target_scaling['close_M15'].inverse_transform(
                preds_scaled[:, 2].reshape(-1, 1)
            )[0][0]

            logging.info(f"Scalping predictions => "
                         f"M1: {pred_close_M1:.2f}, "
                         f"M5: {pred_close_M5:.2f}, "
                         f"M15: {pred_close_M15:.2f}")

            # 4) Подготвяме вход за LightGBM (action)
            current_data = {}
            for feat in action_features:
                if feat.startswith('pred_close_'):
                    if feat == 'pred_close_M1':
                        current_data[feat] = pred_close_M1
                    elif feat == 'pred_close_M5':
                        current_data[feat] = pred_close_M5
                    elif feat == 'pred_close_M15':
                        current_data[feat] = pred_close_M15
                else:
                    # Опитваме да вземем индикатора от последния ред на df_merged
                    if feat in df_merged.columns:
                        current_data[feat] = df_merged.iloc[-1][feat]
                    else:
                        current_data[feat] = 0.0

            X_action = pd.DataFrame([current_data], columns=action_features)
            action = clf.predict(X_action)[0]  # 1 (buy) or -1 (sell)
            logging.info(f"Classifier action: {action}")

            # 5) Управление на позиции
            open_positions = get_open_positions(symbol=SYMBOL, magic=MAGIC_NUMBER_SCALPING)

            if len(open_positions) == 0:
                # Няма никакви позиции -> отваряме според action
                if action == 1:
                    logging.info("Open BUY as no positions are open.")
                    execute_trade(1, SYMBOL, LOT_SIZE_SCALPING)
                else:
                    logging.info("Open SELL as no positions are open.")
                    execute_trade(-1, SYMBOL, LOT_SIZE_SCALPING)

            else:
                # Има вече отворени позиции (може да са 1 или повече, 
                # ако преди това кодът не е контролиран)
                current_position_type = open_positions[0].type  # 0=BUY, 1=SELL
                # За простота приемаме, че имаме само 1 сделка или вземаме първата

                if current_position_type == mt5.POSITION_TYPE_BUY:
                    # Имаме BUY
                    if action == -1:
                        logging.info("Close BUY -> Open SELL")
                        close_position(open_positions[0])
                        time.sleep(1)  # малка пауза
                        execute_trade(-1, SYMBOL, LOT_SIZE_SCALPING)
                    else:
                        logging.info("Already BUY, do nothing.")

                else:
                    # Имаме SELL
                    if action == 1:
                        logging.info("Close SELL -> Open BUY")
                        close_position(open_positions[0])
                        time.sleep(1)
                        execute_trade(1, SYMBOL, LOT_SIZE_SCALPING)
                    else:
                        logging.info("Already SELL, do nothing.")

            # 6) Чакаме до следващия цикъл
            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            logging.info("Stopping the bot.")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
