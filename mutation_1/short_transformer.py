# short_transformer.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Кои колони (цени) ще прогнозираме
TARGET_COLUMNS = ['close_M1', 'close_M5', 'close_M15']
WINDOW_SIZE = 7  # Прозорецът от барове

def load_and_preprocess_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """
    Зарежда данните от CSV, чисти ги и връща X (feature-и), y (целеви ст-сти) и списък с feature-колоните.
    """
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data.dropna(inplace=True)

    # Подготовка на feature колони - примерна логика:
    feature_columns = [col for col in data.columns if (
        # търсим индикатори в рамките M1, M5, M15
        any(col.endswith(tf) for tf in ['M1','M5','M15'])
        and ('sma' in col or 'ema' in col or 'rsi' in col or 'MACD' in col 
             or 'atr' in col or 'Bollinger' in col or 'STOCH' in col or
             'cci20' in col or 'adx14' in col or 'williams_r' in col 
             or 'obv' in col or 'momentum' in col)
    )]

    # Проверка дали TARGET_COLUMNS съществуват
    for tcol in TARGET_COLUMNS:
        if tcol not in data.columns:
            raise ValueError(f"Missing target column {tcol} in data.")

    X = data[feature_columns]
    y = data[TARGET_COLUMNS].shift(-1).dropna()
    # За да не губим редове, изрязваме и X
    X = X.iloc[:-1]

    return X, y, feature_columns

def create_dataset(X, y, window_size):
    """
    Генерира LSTM/Transformer-съвместим формат: (batch, window_size, num_features).
    """
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:(i + window_size)].values)
        ys.append(y.iloc[i + window_size].values)
    return np.array(Xs), np.array(ys)

def build_transformer_model(input_shape, num_targets):
    """
    Изгражда опростен Transformer за времеви редове.
    """
    inputs = layers.Input(shape=input_shape)

    # Проектираме входа към по-високо измерение (embedding)
    x = layers.Dense(64)(inputs)  # (batch, window, 64)

    # Пример: 2 TransformerEncoder блока
    # NB: В TF 2.11+ има слоеве layers.TransformerEncoder, layers.TransformerDecoder,
    # но могат да изискват experimental флаг или различни версии на Keras.
    # Тук ползваме "patch" с MultiHeadAttention + FeedForward
    # Ако имате KerasNLP, може да ползвате официалния.

    # Self-attention block
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention])  # skip connection
    x = layers.LayerNormalization()(x)

    # Feed-forward block
    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(64)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # GlobalAveragePooling1D за да обобщим по времевата ос
    x = layers.GlobalAveragePooling1D()(x)

    # Финални плътни слоеве
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

def train_scalping_model(X, y, feature_columns, retrain=False):
    """
    Тренира Transformer модела за краткосрочни прогнози. 
    """
    # Делим на train/test
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    # Scale на feature-и
    if retrain and os.path.exists('scalping_feature_scaler.pkl'):
        feature_scaler = joblib.load('scalping_feature_scaler.pkl')
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
    else:
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        joblib.dump(feature_scaler, 'scalping_feature_scaler.pkl')

    # Scale на таргети
    if retrain and os.path.exists('scalping_target_scalers.pkl'):
        target_scalers = joblib.load('scalping_target_scalers.pkl')
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
        joblib.dump(target_scalers, 'scalping_target_scalers.pkl')

    # Създаваме dataset
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

    # Строим или зареждаме модел
    if retrain and os.path.exists('scalping_model.keras'):
        model = tf.keras.models.load_model('scalping_model.keras')
        print("Loaded existing short-term model for retraining.")
    else:
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        model = build_transformer_model(input_shape, len(TARGET_COLUMNS))
        print("Created a new short-term Transformer model.")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('scalping_model.keras', monitor='val_loss', save_best_only=True)

    # Тренираме
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Оценка
    loss, mae = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    print(f"Short-Term Transformer - Test MAE (Scaled): {mae:.4f}")

    # Inverse transform за да видим реалната грешка
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

    # Примерен плот
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_actual['close_M1'], label='Actual M1 Close')
    plt.plot(predictions['close_M1'], label='Predicted M1 Close')
    plt.legend()
    plt.title('Actual vs Predicted M1 (Short-Term Transformer)')
    plt.show()

def main():
    X, y, feature_columns = load_and_preprocess_data()
    train_scalping_model(X, y, feature_columns, retrain=True)

if __name__ == "__main__":
    main()
