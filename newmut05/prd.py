# generate_predictions.py
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Files:
FILENAME = 'XAUUSD_data_multiple_timeframes.csv'
SCALPING_MODEL_FILE = 'scalping_model.keras'
LONGTERM_MODEL_FILE = 'longterm_model.keras'
SCALPING_FEATURE_SCALER_FILE = 'scalping_feature_scaler.pkl'
SCALPING_TARGET_SCALERS_FILE = 'scalping_target_scalers.pkl'
LONGTERM_FEATURE_SCALER_FILE = 'longterm_feature_scaler.pkl'
LONGTERM_TARGET_SCALERS_FILE = 'longterm_target_scalers.pkl'

WINDOW_SIZE_SCALPING = 7
WINDOW_SIZE_LONGTERM = 37

def load_models_and_scalers():
    model_scalping = load_model(SCALPING_MODEL_FILE)
    # If you need long-term predictions, uncomment the next line
    # model_longterm = load_model(LONGTERM_MODEL_FILE)
    feature_scaler_scalping = joblib.load(SCALPING_FEATURE_SCALER_FILE)
    target_scalers_scalping = joblib.load(SCALPING_TARGET_SCALERS_FILE)
    # feature_scaler_longterm = joblib.load(LONGTERM_FEATURE_SCALER_FILE)
    # target_scalers_longterm = joblib.load(LONGTERM_TARGET_SCALERS_FILE)
    return (model_scalping, feature_scaler_scalping, target_scalers_scalping)

def load_data():
    data = pd.read_csv(FILENAME, index_col='time', parse_dates=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)
    return data

def generate_predictions_for_dataset(data, model_scalping, feature_scaler_scalping, target_scalers_scalping):
    """
    Generate predictions for the entire dataset using vectorized operations.
    """
    # Identify the columns required by the scalping model
    scalping_columns = feature_scaler_scalping.feature_names_in_

    # Prepare the scalping data
    data_scalping = data[scalping_columns]
    data_scalping_scaled = feature_scaler_scalping.transform(data_scalping)

    # Create sequences
    num_samples = data_scalping_scaled.shape[0] - WINDOW_SIZE_SCALPING + 1
    if num_samples <= 0:
        print("Not enough data to create sequences for scalping model.")
        return data

    X_scalping = np.array([data_scalping_scaled[i:i+WINDOW_SIZE_SCALPING] for i in range(num_samples)])

    # Run predictions
    predictions_scaled = model_scalping.predict(X_scalping)

    # Inverse transform predictions
    pred_close_M1 = target_scalers_scalping['close_M1'].inverse_transform(predictions_scaled[:,0].reshape(-1,1)).ravel()
    pred_close_M5 = target_scalers_scalping['close_M5'].inverse_transform(predictions_scaled[:,1].reshape(-1,1)).ravel()
    pred_close_M15 = target_scalers_scalping['close_M15'].inverse_transform(predictions_scaled[:,2].reshape(-1,1)).ravel()

    # Append predictions to the dataframe
    # The predictions correspond to indices from WINDOW_SIZE_SCALPING-1 to end
    pred_indices = data.index[WINDOW_SIZE_SCALPING-1:]

    data.loc[pred_indices, 'pred_close_M1'] = pred_close_M1
    data.loc[pred_indices, 'pred_close_M5'] = pred_close_M5
    data.loc[pred_indices, 'pred_close_M15'] = pred_close_M15

    return data

def main():
    (model_scalping, feature_scaler_scalping, target_scalers_scalping) = load_models_and_scalers()

    data = load_data()
    data_with_preds = generate_predictions_for_dataset(data, model_scalping, feature_scaler_scalping, target_scalers_scalping)
    # Save updated CSV with predictions
    data_with_preds.to_csv(FILENAME)
    print(f"Updated {FILENAME} with predicted columns.")

if __name__ == "__main__":
    main()
