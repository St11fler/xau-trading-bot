import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_absolute_error
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
import optuna

# Parameters
TARGET_COLUMNS = ['close_H1', 'close_H4', 'close_D1']
WINDOW_SIZE = 37

def load_and_preprocess_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """Load and preprocess data for long-term model training."""
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    feature_columns = [col for col in data.columns if (
        ('sma' in col or 'ema' in col or 'rsi' in col or 'MACD' in col or 'atr' in col or
         'bollinger_upper' in col or 'bollinger_middle' in col or 'bollinger_lower' in col or
         'STOCHk' in col or 'STOCHd' in col or 'cci20' in col or 'adx14' in col or 'williams_r' in col or
         'obv' in col or 'momentum' in col)
        and (col.endswith('H1') or col.endswith('H4') or col.endswith('D1'))
    )]

    for target in TARGET_COLUMNS:
        if target not in data.columns:
            print(f"Target column '{target}' not found. Available columns: {data.columns.tolist()}")
            exit()

    X = data[feature_columns]
    y = data[TARGET_COLUMNS].shift(-1)
    X = X[:-1]
    y = y[:-1]

    return X, y, feature_columns

def create_dataset(X, y, window_size):
    """Create dataset for LSTM training."""
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X.iloc[i:(i + window_size)].values)
        ys.append(y.iloc[i + window_size].values)
    return np.array(Xs), np.array(ys)

def build_longterm_model(input_shape, num_targets, dropout_rate=0.2):
    """Build a simplified LSTM model with L2 regularization."""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    outputs = layers.Dense(num_targets)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def objective(trial, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, input_shape, num_targets):
    """Optuna objective function for hyperparameter tuning."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    
    model = build_longterm_model(input_shape, num_targets, dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping],
        verbose=0
    )
    
    loss, mae = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    return mae

def train_longterm_model(X, y, feature_columns, retrain=False):
    """Train the long-term LSTM model with hyperparameter tuning."""
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    if retrain and os.path.exists('longterm_feature_scaler.pkl'):
        feature_scaler = joblib.load('longterm_feature_scaler.pkl')
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
    else:
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        joblib.dump(feature_scaler, 'longterm_feature_scaler.pkl')

    if retrain and os.path.exists('longterm_target_scalers.pkl'):
        target_scalers = joblib.load('longterm_target_scalers.pkl')
        y_train_scaled = y_train.values.copy()
        y_test_scaled = y_test.values.copy()
        for col in TARGET_COLUMNS:
            scaler = target_scalers[col]
            idx = TARGET_COLUMNS.index(col)
            y_train_scaled[:, idx] = scaler.transform(y_train[[col]])
            y_test_scaled[:, idx] = scaler.transform(y_test[[col]])
    else:
        target_scalers = {}
        y_train_scaled = np.zeros(y_train.shape)
        y_test_scaled = np.zeros(y_test.shape)
        for col in TARGET_COLUMNS:
            scaler = MinMaxScaler()
            idx = TARGET_COLUMNS.index(col)
            y_train_scaled[:, idx] = scaler.fit_transform(y_train[[col]]).ravel()
            y_test_scaled[:, idx] = scaler.transform(y_test[[col]]).ravel()
            target_scalers[col] = scaler
        joblib.dump(target_scalers, 'longterm_target_scalers.pkl')

    X_train_lstm, y_train_lstm = create_dataset(pd.DataFrame(X_train_scaled, index=X_train.index), pd.DataFrame(y_train_scaled, index=y_train.index), WINDOW_SIZE)
    X_test_lstm, y_test_lstm = create_dataset(pd.DataFrame(X_test_scaled, index=X_test.index), pd.DataFrame(y_test_scaled, index=y_test.index), WINDOW_SIZE)

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, (X_train_lstm.shape[1], X_train_lstm.shape[2]), len(TARGET_COLUMNS)), n_trials=10)
    
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # Train final model with best hyperparameters
    model = build_longterm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]), len(TARGET_COLUMNS), best_params['dropout_rate'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('longterm_model.keras', monitor='val_loss', save_best_only=True)

    model.fit(
        X_train_lstm, y_train_lstm,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    loss, mae = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    print(f"Long-Term Model - Test MAE (Scaled): {mae}")

    predictions_scaled = model.predict(X_test_lstm)
    mae_actual = {}
    for i, col in enumerate(TARGET_COLUMNS):
        scaler = target_scalers[col]
        pred_inv = scaler.inverse_transform(predictions_scaled[:, i].reshape(-1, 1)).ravel()
        y_inv = scaler.inverse_transform(y_test_lstm[:, i].reshape(-1, 1)).ravel()
        mae_actual[col] = mean_absolute_error(y_inv, pred_inv)
        print(f'Test MAE for {col} (Actual Units): {mae_actual[col]}')

def main():
    X, y, feature_columns = load_and_preprocess_data()
    train_longterm_model(X, y, feature_columns, retrain=True)

if __name__ == "__main__":
    main()