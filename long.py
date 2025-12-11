# model_longterm.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Parameters
TARGET_COLUMNS = ['close_H1', 'close_H4', 'close_D1']
WINDOW_SIZE = 37  # Adjust based on your data and strategy

def load_and_preprocess_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """Load and preprocess data for long-term model training."""
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    # Feature selection for Long-Term: Adjust this logic as needed
    feature_columns = [col for col in data.columns if (
        ('sma' in col or 'ema' in col or 'rsi' in col or 'MACD' in col or 'atr' in col or
         'bollinger_upper' in col or 'bollinger_middle' in col or 'bollinger_lower' in col or
         'STOCHk' in col or 'STOCHd' in col or 'cci20' in col or 'adx14' in col or 'williams_r' in col or
         'obv' in col or 'momentum' in col)
        and (col.endswith('H1') or col.endswith('H4') or col.endswith('D1'))
    )]

    # Ensure target columns exist
    for target in TARGET_COLUMNS:
        if target not in data.columns:
            print(f"Target column '{target}' not found. Available columns: {data.columns.tolist()}")
            exit()

    X = data[feature_columns]
    y = data[TARGET_COLUMNS].shift(-1)  # Predict next close prices
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

def build_longterm_model(input_shape, num_targets):
    """
    Build the long-term LSTM model with Bidirectional layers and Self-Attention.
    This is an advanced architecture; you may simplify if needed.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Self-attention layer
    attention = layers.Attention()([x, x])
    
    # Flatten and Dense layers
    attention_flat = layers.Flatten()(attention)
    x = layers.Dense(64, activation='relu')(attention_flat)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_targets)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def train_longterm_model(X, y, feature_columns, retrain=False):
    """Train the long-term LSTM model."""
    # Split data
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    # Feature scaling
    if retrain and os.path.exists('longterm_feature_scaler.pkl'):
        feature_scaler = joblib.load('longterm_feature_scaler.pkl')
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
    else:
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        joblib.dump(feature_scaler, 'longterm_feature_scaler.pkl')

    # Target scaling
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

    # Create datasets for LSTM
    X_train_lstm, y_train_lstm = create_dataset(pd.DataFrame(X_train_scaled, index=X_train.index), pd.DataFrame(y_train_scaled, index=y_train.index), WINDOW_SIZE)
    X_test_lstm, y_test_lstm = create_dataset(pd.DataFrame(X_test_scaled, index=X_test.index), pd.DataFrame(y_test_scaled, index=y_test.index), WINDOW_SIZE)

    # Build or load the model
    if retrain and os.path.exists('longterm_model.keras'):
        model = tf.keras.models.load_model('longterm_model.keras')
        print("Loaded existing long-term model for retraining.")
    else:
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        model = build_longterm_model(input_shape, len(TARGET_COLUMNS))
        print("Created a new long-term model.")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('longterm_model.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Evaluate the model
    loss, mae = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    print(f"Long-Term Model - Test MAE (Scaled): {mae}")

    # Inverse transform predictions
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
        print(f'Test MAE for {col} (Actual Units): {mae_actual[col]}')

    # Example plotting for H1
    plt.figure(figsize=(14,7))
    plt.plot(y_test_actual['close_H1'], label='Actual H1 Close')
    plt.plot(predictions['close_H1'], label='Predicted H1 Close')
    plt.legend()
    plt.title('Actual vs Predicted Close Price H1')
    plt.show()

def main():
    """Main function to train the long-term model."""
    X, y, feature_columns = load_and_preprocess_data()
    train_longterm_model(X, y, feature_columns, retrain=True)

if __name__ == "__main__":
    main()
