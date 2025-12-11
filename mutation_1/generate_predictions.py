# generate_predictions.py
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

FILENAME = 'XAUUSD_data_multiple_timeframes.csv'
SCALPING_MODEL_FILE = 'scalping_model.keras'
SCALPING_FEATURE_SCALER_FILE = 'scalping_feature_scaler.pkl'
SCALPING_TARGET_SCALERS_FILE = 'scalping_target_scalers.pkl'

WINDOW_SIZE_SCALPING = 7

def load_models_and_scalers():
    model_scalping = load_model(SCALPING_MODEL_FILE)
    feature_scaler_scalping = joblib.load(SCALPING_FEATURE_SCALER_FILE)
    target_scalers_scalping = joblib.load(SCALPING_TARGET_SCALERS_FILE)
    return (model_scalping, feature_scaler_scalping, target_scalers_scalping)

def load_data():
    data = pd.read_csv(FILENAME, index_col='time', parse_dates=True)
    data.dropna(inplace=True)
    return data

def generate_sequences(data_scaled, window_size):
    """
    Създава припокриващи се прозорци от цялата последователност.
    """
    sequences = []
    for i in range(len(data_scaled) - window_size + 1):
        seq = data_scaled[i:i + window_size]
        sequences.append(seq)
    return np.array(sequences)

def generate_predictions_for_dataset(data, model, feature_scaler, target_scalers):
    """
    Генерира прогнози за целия датасет и добавя колони pred_close_M1, pred_close_M5, pred_close_M15.
    """
    # Кои feature колони са нужни
    scalping_columns = feature_scaler.feature_names_in_
    if not set(scalping_columns).issubset(data.columns):
        missing_feats = set(scalping_columns) - set(data.columns)
        raise ValueError(f"Missing features for scalping: {missing_feats}")

    X = data[scalping_columns].values
    X_scaled = feature_scaler.transform(X)

    sequences = generate_sequences(X_scaled, WINDOW_SIZE_SCALPING)
    if len(sequences) == 0:
        print("Not enough data to generate predictions.")
        return data

    preds_scaled = model.predict(sequences)
    # preds_scaled има форма (n_samples, 3) за M1, M5, M15

    # Обратно трансформиране
    pred_close_M1 = target_scalers['close_M1'].inverse_transform(
        preds_scaled[:, 0].reshape(-1, 1)
    ).ravel()
    pred_close_M5 = target_scalers['close_M5'].inverse_transform(
        preds_scaled[:, 1].reshape(-1, 1)
    ).ravel()
    pred_close_M15 = target_scalers['close_M15'].inverse_transform(
        preds_scaled[:, 2].reshape(-1, 1)
    ).ravel()

    # Индексите, на които отговарят прогнози
    pred_indices = data.index[WINDOW_SIZE_SCALPING - 1:]

    data.loc[pred_indices, 'pred_close_M1'] = pred_close_M1
    data.loc[pred_indices, 'pred_close_M5'] = pred_close_M5
    data.loc[pred_indices, 'pred_close_M15'] = pred_close_M15

    return data

def main():
    model_scalping, feature_scaler_scalping, target_scalers_scalping = load_models_and_scalers()
    data = load_data()
    data = generate_predictions_for_dataset(
        data, model_scalping, feature_scaler_scalping, target_scalers_scalping
    )
    data.to_csv(FILENAME)
    print("Updated CSV with scalping predictions: pred_close_M1, pred_close_M5, pred_close_M15")

if __name__ == "__main__":
    main()
