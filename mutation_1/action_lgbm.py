# action_lgbm.py
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_action_data(filename='XAUUSD_data_multiple_timeframes.csv'):
    """
    Зарежда данни, добавя action = buy(1) / sell(-1) според бъдеща доходност и връща X, y.
    """
    data = pd.read_csv(filename, index_col='time', parse_dates=True)
    data.dropna(inplace=True)

    # feature_columns може да е произволен набор от индикатори + pred_close_xx
    feature_columns = [
        col for col in data.columns if (
            'ema' in col or 'rsi' in col or 'MACD' in col or
            'sma' in col or 'atr' in col or 'cci' in col or
            'ADX' in col or 'momentum' in col or 'obv' in col or
            'STOCHk' in col or 'STOCHd' in col or 'williams_r' in col or
            col.startswith('pred_close_')
        )
    ]

    # Пример: ако close_M1 утре е +0.1%, => buy; ако е -0.1% => sell.
    future_return = data['close_M1'].shift(-1) / data['close_M1'] - 1
    data['action'] = np.where(future_return > 0.001, 1,
                              np.where(future_return < -0.001, -1, np.nan))

    data.dropna(subset=['action'], inplace=True)

    X = data[feature_columns]
    y = data['action'].astype(int)

    return X, y, feature_columns

def cross_validate_classifier(clf, X, y, n_splits=5):
    """
    TimeSeriesSplit cross-validation за по-реалистична оценка.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(clf, X, y, cv=tscv, scoring='accuracy')
    print(f"TimeSeries Split CV Scores: {scores}")
    print(f"Average CV Accuracy: {scores.mean():.4f}")

def plot_confusion_matrix(y_true, y_pred, classes=['Sell', 'Buy']):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def plot_feature_importances(clf, feature_names, top_n=20):
    """
    Извежда feature importances от LightGBM.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title('Top Feature Importances (LGBM)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Sell', 'Buy'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.show()

def train_action_classifier():
    """
    Тренира LightGBM на база на вече подготвени данни (вкл. pred_close_xx), 
    cross-validate, визуализира и записва модела.
    """
    X, y, feature_columns = prepare_action_data()

    # Разделяме хронологично: 80% train, 20% test
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Инициализираме LGBM
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, class_weight='balanced')

    # Cross-validation
    print("Performing TimeSeries Cross-Validation...")
    cross_validate_classifier(clf, X_train, y_train, n_splits=5)

    # Тренираме
    clf.fit(X_train, y_train)
    print("Training Score:", clf.score(X_train, y_train))
    print("Test Score:", clf.score(X_test, y_test))

    # Предсказваме
    y_pred = clf.predict(X_test)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sell', 'Buy']))
    plot_classification_report(y_test, y_pred)

    # Feature Importances
    plot_feature_importances(clf, feature_columns)

    # Запазваме модела
    joblib.dump((clf, feature_columns), 'action_classifier.pkl')
    print("Action classifier saved as 'action_classifier.pkl'")

if __name__ == "__main__":
    train_action_classifier()
