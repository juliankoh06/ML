"""
COMPARE TRAINED RANDOM FOREST MODELS
(Untuned vs Tuned)
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

print("="*70)
print("COMPARING TRAINED RANDOM FOREST MODELS")
print("="*70)

# 1. LOAD MODELS + PREPROCESSOR
print("\nLoading models...")

untuned_model = joblib.load("models/best_rf_regressor_untuned.pkl")
tuned_model = joblib.load("models/best_rf_regressor.pkl")
preprocessor = joblib.load("models/preprocessor_regression.pkl")

print("Models loaded successfully.")

# 2. LOAD TEST DATASET
df = pd.read_csv('data/housing.csv').dropna().drop_duplicates().reset_index(drop=True)

# Feature engineering (must match training)
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']
df = df.drop(['total_rooms', 'total_bedrooms', 'population'], axis=1)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Preprocess
X_processed = preprocessor.transform(X)


# 3. PREDICT USING BOTH MODELS
print("\nGenerating predictions...")

y_pred_untuned = untuned_model.predict(X_processed)
y_pred_tuned = tuned_model.predict(X_processed)


# 4. DEFINE METRIC FUNCTIONS
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 5. CALCULATE METRICS


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }

metrics_untuned = calculate_metrics(y, y_pred_untuned)
metrics_tuned = calculate_metrics(y, y_pred_tuned)

print("\nUNTUNED MODEL METRICS")
print(metrics_untuned)

print("\nTUNED MODEL METRICS")
print(metrics_tuned)


# 6. VISUAL COMPARISON DIAGRAM

labels = ["MAE", "RMSE", "R²", "MAPE"]
untuned_vals = [metrics_untuned["MAE"], metrics_untuned["RMSE"],
                metrics_untuned["R2"], metrics_untuned["MAPE"]]
tuned_vals = [metrics_tuned["MAE"], metrics_tuned["RMSE"],
              metrics_tuned["R2"], metrics_tuned["MAPE"]]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, untuned_vals, width, label="Untuned", alpha=0.7)
plt.bar(x + width/2, tuned_vals, width, label="Tuned", alpha=0.7)

plt.xticks(x, labels)
plt.ylabel("Error / Score")
plt.title("Comparison: Tuned vs Untuned Random Forest Regression")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

