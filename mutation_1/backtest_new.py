# backtest_new.py
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)

INITIAL_BALANCE = 500.0
LOT_SIZE = 0.01
POINT_VALUE = 0.1

def load_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data = data.sort_index()
    data.dropna(inplace=True)
    return data

def load_action_classifier():
    clf, action_features = joblib.load('action_classifier.pkl')
    return clf, action_features

def simulate_trades(data, clf, action_features):
    balance = INITIAL_BALANCE
    trade_history = []
    position = None  # None, 'long', or 'short'
    entry_price = 0.0

    for i in range(len(data)):
        # Примерен ежедневен/на всяка свещ backtest
        row = data.iloc[i]
        # Проверка дали имаме всички нужни feature-и
        if not all(f in data.columns for f in action_features):
            continue

        X_action = row[action_features].values.reshape(1, -1)
        action = clf.predict(X_action)[0]  # 1 или -1

        if action == 1:  # Buy
            # Ако сме в short, затваряме го
            if position == 'short':
                profit = entry_price - row['close_M1']  # Проста логика
                balance += profit * LOT_SIZE * POINT_VALUE
                trade_history.append(profit * LOT_SIZE * POINT_VALUE)
                position = None
            # Ако нямаме позиция, влизаме long
            if position is None:
                position = 'long'
                entry_price = row['close_M1']

        elif action == -1:  # Sell
            # Ако сме в long, затваряме го
            if position == 'long':
                profit = row['close_M1'] - entry_price
                balance += profit * LOT_SIZE * POINT_VALUE
                trade_history.append(profit * LOT_SIZE * POINT_VALUE)
                position = None
            # Ако нямаме позиция, влизаме short
            if position is None:
                position = 'short'
                entry_price = row['close_M1']

    # Ако в края е останала отворена позиция, затваряме я
    if position == 'long':
        final_profit = data['close_M1'].iloc[-1] - entry_price
        balance += final_profit * LOT_SIZE * POINT_VALUE
        trade_history.append(final_profit * LOT_SIZE * POINT_VALUE)
    elif position == 'short':
        final_profit = entry_price - data['close_M1'].iloc[-1]
        balance += final_profit * LOT_SIZE * POINT_VALUE
        trade_history.append(final_profit * LOT_SIZE * POINT_VALUE)

    return balance, trade_history

def main():
    data = load_data()
    clf, action_features = load_action_classifier()
    final_balance, trade_history = simulate_trades(data, clf, action_features)

    total_trades = len(trade_history)
    net_profit = sum(trade_history)
    wins = sum(1 for x in trade_history if x > 0)
    losses = sum(1 for x in trade_history if x < 0)

    print(f"Final Balance: {final_balance}")
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}")
    if total_trades > 0:
        print(f"Win Rate: {wins/total_trades*100:.2f}%")
    print(f"Net Profit: {net_profit}")

if __name__ == "__main__":
    main()
