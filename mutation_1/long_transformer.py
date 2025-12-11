# long_transformer.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

TARGET_COLUMNS = ['close_H1', 'close_H4', 'close_D1']
WINDOW_SIZE = 37  # По-голям прозорец

def load_and_preprocess_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """
    Зареждаме и филтрираме данните за дългосрочни прогнози (H1, H4, D1).
    """
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data.dropna(inplace=True)

    feature_columns = [col for col in data.columns if (
        any(col.endswith(tf) for tf in ['H1','H4','D1'])
        and ('sma' in col or 'ema' in col or 'rsi' in col or 'MACD' in col 
             or 'atr' in col or 'Bollinger' in col or 'STOCH' in col or
             'cci20' in col or 'adx14' in col or 'williams_r' in col 
             or 'obv' in col or 'momentum' in col)
    )]

    for tcol in TARGET_COLUMNS:
        if tcol not in data.columns:
            raise ValueError(f"Missing target column {tcol} in data.")

    X = data[feature_columns]
    y = data[TARGET_COLUMNS].shift(-1).dropna()
    X = X.iloc[:-1]

    return X, y, feature_columns

def create_dataset(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:(i+window_size)].values)
        ys.append(y.iloc[i + window_size].values)
    return np.array(Xs), np.array(ys)

def build_transformer_model(input_shape, num_targets):
    """
    Transformer, подобен на short_transformer, но може да увеличите броя слоеве/хедове за по-дългосрочните данни.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inputs)

    # Self-attention block
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    # Feed-forward block
    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(64)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_targets)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_longterm_model(X, y, feature_columns, retrain=False):
    """
    Тренира Transformer за дългосрочни прогнози (H1, H4, D1).
    """
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    # Фийчър-скалери
    if retrain and os.path.exists('longterm_feature_scaler.pkl'):
        feature_scaler = joblib.load('longterm_feature_scaler.pkl')
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
    else:
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        joblib.dump(feature_scaler, 'longterm_feature_scaler.pkl')

    # Таргет-скалери
    if retrain and os.path.exists('longterm_target_scalers.pkl'):
        target_scalers = joblib.load('longterm_target_scalers.pkl')
        y_train_scaled = y_train.copy().values
        y_test_scaled = y_test.copy().values
        for i, col in enumerate(TARGET_COLUMNS):
            scaler = target_scalers[col]
            y_train_scaled[:, i] = scaler.transform(y_train[[col]])
            y_test_scaled[:, i] = scaler.transform(y_test[[col]])
    else:
        target_scalers = {}
        y_train_scaled = np.zeros_like(y_train.values)
        y_test_scaled = np.zeros_like(y_test.values)
        for i, col in enumerate(TARGET_COLUMNS):
            scaler = MinMaxScaler()
            y_train_scaled[:, i] = scaler.fit_transform(y_train[[col]]).ravel()
            y_test_scaled[:, i] = scaler.transform(y_test[[col]]).ravel()
            target_scalers[col] = scaler
        joblib.dump(target_scalers, 'longterm_target_scalers.pkl')

    # Dataset
    X_train_lstm, y_train_lstm = create_dataset(
        pd.DataFrame(X_train_scaled, index=X_train.index),
        pd.DataFrame(y_train_scaled, index=y_train.index),
        WINDOW_SIZE
    )
    X_test_lstm, y_test_lstm = create_dataset(
        pd.DataFrame(X_test_scaled, index=X_test.index),
        pd.DataFrame(y_test_scaled, index=y_test.index),
        WINDOW_SIZE
    )

    # Модел
    if retrain and os.path.exists('longterm_model.keras'):
        model = tf.keras.models.load_model('longterm_model.keras')
        print("Loaded existing long-term model for retraining.")
    else:
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        model = build_transformer_model(input_shape, len(TARGET_COLUMNS))
        print("Created a new long-term Transformer model.")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('longterm_model.keras', monitor='val_loss', save_best_only=True)

    # Train
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    loss, mae = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    print(f"Long-Term Transformer - Test MAE (Scaled): {mae:.4f}")

    predictions_scaled = model.predict(X_test_lstm)
    predictions = {}
    y_test_actual = {}
    mae_actual = {}

    for i, col in enumerate(TARGET_COLUMNS):
        scaler = target_scalers[col]
        pred_inv = scaler.inverse_transform(predictions_scaled[:, i].reshape(-1, 1)).ravel()
        y_inv = scaler.inverse_transform(y_test_lstm[:, i].reshape(-1, 1)).ravel()
        predictions[col] = pred_inv
        y_test_actual[col] = y_inv
        mae_actual[col] = mean_absolute_error(y_test_actual[col], predictions[col])
        print(f'Test MAE for {col} (Actual Units): {mae_actual[col]:.4f}')

    # Примерен плот за H1
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_actual['close_H1'], label='Actual H1 Close')
    plt.plot(predictions['close_H1'], label='Predicted H1 Close')
    plt.legend()
    plt.title('Actual vs Predicted H1 (Long-Term Transformer)')
    plt.show()

def main():
    X, y, feature_columns = load_and_preprocess_data()
    train_longterm_model(X, y, feature_columns, retrain=True)

if __name__ == "__main__":
    main()
