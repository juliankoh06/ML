"""
Regression Models Training Script 
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

print("="*70)
print("REGRESSION - UNTUNED vs. TUNED - (RANDOM FOREST)")
print("="*70)


# 1. Preprocessing & Feature Engineering
df = pd.read_csv('data/housing.csv').dropna().drop_duplicates().reset_index(drop=True)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X['rooms_per_household'] = X['total_rooms'] / X['households']
X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
X['population_per_household'] = X['population'] / X['households']
X = X.drop(['total_rooms', 'total_bedrooms', 'population'], axis=1)

# 2. Identify feature types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 3. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 6. UNTUNED MODEL
print("\n" + "="*70)
print("Training UNTUNED Model (Baseline)")
print("="*70)

base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
base_rf.fit(X_train_processed, y_train)

# Evaluate baseline
y_pred_base = base_rf.predict(X_test_processed)
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mape_base = mape(y_test, y_pred_base)

print(f"Untuned RF Results: MAE=${mae_base:,.2f}, R²={r2_base:.4f}, RMSE=${rmse_base:,.2f}, MAPE={mape_base:.2f}%")

# 7. TUNED MODEL (GRIDSEARCHCV)
print("\n" + "="*70)
print("Training TUNED Model (GridSearchCV)")
print("="*70)
print("\nPerforming GridSearchCV...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_processed, y_train)
best_rf_model = grid_search.best_estimator_

print(f"\nBest Parameters Found: {grid_search.best_params_}")

# Evaluate tuned model
y_pred_tuned = best_rf_model.predict(X_test_processed)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mape_tuned = mape(y_test, y_pred_tuned)


print(f"Tuned RF Results:   MAE=${mae_tuned:,.2f}, R²={r2_tuned:.4f}, RMSE=${rmse_tuned:,.2f}, MAPE={mape_tuned:.2f}%")


# 8. COMPARISON SUMMARY
print("\n" + "="*70)
print("UNTUNED vs. TUNED COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': ['Untuned (Baseline)', 'Tuned (GridSearchCV)'],
    'MAE': [mae_base, mae_tuned],
    'RMSE': [rmse_base, rmse_tuned],
    'R²': [r2_base, r2_tuned],
    'MAPE (%)': [mape_base, mape_tuned]
})

print(results_df.to_string(index=False))


# 9. COMPARISON DIAGRAM
labels = ["MAE", "RMSE", "R²", "MAPE"]
untuned_vals = [mae_base, rmse_base, r2_base, mape_base]
tuned_vals = [mae_tuned, rmse_tuned, r2_tuned, mape_tuned]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, untuned_vals, width, label="Untuned", alpha=0.7)
plt.bar(x + width/2, tuned_vals, width, label="Tuned", alpha=0.7)

plt.title("Random Forest Regression Comparison (Untuned vs. Tuned)")
plt.xticks(x, labels)
plt.ylabel("Metric Value")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 10. SAVE BEST MODEL
joblib.dump(best_rf_model, 'models/best_rf_regressor.pkl')
joblib.dump(preprocessor, 'models/preprocessor_regression.pkl')
