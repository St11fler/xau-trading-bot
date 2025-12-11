import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_action_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """Load data, prepare features and labels for classification."""
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data.dropna(inplace=True)

    feature_columns = [col for col in data.columns if 
                       ('ema' in col or 'rsi' in col or 'MACD' in col or 
                        'sma' in col or 'atr' in col or 'cci' in col or 
                        'ADX' in col or 'momentum' in col or 'obv' in col or 
                        'STOCHk' in col or 'STOCHd' in col or 'williams_r' in col)]
    feature_columns += ['pred_close_M1', 'pred_close_M5', 'pred_close_M15']

    missing_preds = [col for col in ['pred_close_M1', 'pred_close_M5', 'pred_close_M15'] if col not in data.columns]
    if missing_preds:
        raise ValueError(f"Missing predicted columns in data: {missing_preds}. Please run generate_predictions.py first.")

    future_return = data['close_M1'].shift(-1) / data['close_M1'] - 1
    data['action'] = np.where(future_return > 0.001, 1, 
                              np.where(future_return < -0.001, -1, np.nan))

    data.dropna(inplace=True, subset=['action'])
    
    X = data[feature_columns]
    y = data['action'].astype(int)

    return X, y, feature_columns

def cross_validate_classifier(clf, X, y, n_splits=5):
    """Perform TimeSeries cross-validation and print average scores."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(clf, X, y, cv=tscv, scoring='accuracy')
    print(f"TimeSeries Split Cross-Validation Scores: {scores}")
    print(f"Average CV Accuracy: {scores.mean():.4f}")

def plot_confusion_matrix(y_true, y_pred, classes=['Sell', 'Buy']):
    """Plot confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def plot_feature_importances(clf, feature_names, top_n=20):
    """Plot feature importances from the RandomForestClassifier."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title('Top Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_classification_report(y_true, y_pred):
    """Visualize the classification report."""
    report = classification_report(y_true, y_pred, target_names=['Sell', 'Buy'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.show()

def train_action_classifier(visualize=False):
    """Train the RandomForestClassifier with feature selection, evaluate, and save the model."""
    X, y, feature_columns = prepare_action_data()
    
    # Feature selection using RFE
    clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=clf_temp, n_features_to_select=20)
    rfe.fit(X, y)
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_]
    print(f"Selected features: {selected_features}")
    
    X = X[selected_features]
    
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    print("Performing TimeSeries Cross-Validation...")
    cross_validate_classifier(clf, X_train, y_train, n_splits=5)

    clf.fit(X_train, y_train)
    print("Action Classifier Training Score:", clf.score(X_train, y_train))
    print("Action Classifier Test Score:", clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    if visualize:
        plot_confusion_matrix(y_test, y_pred)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Sell', 'Buy']))
        plot_classification_report(y_test, y_pred)
        plot_feature_importances(clf, selected_features)

    joblib.dump((clf, selected_features), 'action_classifier.pkl')
    print("Action classifier saved as 'action_classifier.pkl'")

if __name__ == "__main__":
    train_action_classifier(visualize=False)