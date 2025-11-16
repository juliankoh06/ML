"""
Keras MLP Regressor 
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("REGRESSION -(KERAS MLP)")
print("="*70)

# 1.  Preprocess dataset and Feature engineering
df = pd.read_csv('data/housing.csv').dropna().drop_duplicates().reset_index(drop=True)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X['rooms_per_household'] = X['total_rooms'] / X['households']
X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
X['population_per_household'] = X['population'] / X['households']
X = X.drop(['total_rooms', 'total_bedrooms', 'population'], axis=1)

# 2. Preprocessing Pipeline
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Fit Preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 5. Remove Outlier
uncapped_mask = y_train < 500001.0
y_train_uncapped = y_train[uncapped_mask]
X_train_uncapped = X_train_processed[uncapped_mask]

Q1 = y_train_uncapped.quantile(0.25); Q3 = y_train_uncapped.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
mask = (y_train_uncapped >= lower) & (y_train_uncapped <= upper)

X_train_clean = X_train_uncapped[mask]
y_train_clean = y_train_uncapped[mask]

# Log and Scale Target
y_train_log = np.log1p(y_train_clean)
y_test_log = np.log1p(y_test) 
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_log.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test_log.values.reshape(-1, 1)).flatten()

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_clean, y_train_scaled, test_size=0.2, random_state=42
)
INPUT_SHAPE = (X_train_clean.shape[1],)


# 6. UNTUNED MODEL (BASELINE)
print("\n" + "="*70)
print("Training UNTUNED Model (Baseline)")
print("="*70)

baseline_model = keras.Sequential([
    layers.Input(shape=INPUT_SHAPE),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])
baseline_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
baseline_model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split),
                   epochs=100, batch_size=64,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                   verbose=0)

# Evaluate baseline
y_pred_base_scaled = baseline_model.predict(X_test_processed, verbose=0).flatten()
y_pred_base_log = y_scaler.inverse_transform(y_pred_base_scaled.reshape(-1, 1)).flatten()
y_pred_base = np.expm1(y_pred_base_log)
y_pred_base = np.nan_to_num(y_pred_base, nan=y_test.mean())

mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

print(f"Untuned Keras Results: MAE=${mae_base:,.2f}, R²={r2_base:.4f}, RMSE=${rmse_base:,.2f}")


# 7. TUNED MODEL (KERASTUNER)
print("\n" + "="*70)
print("Training TUNED Model (KerasTuner)")
print("="*70)

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=INPUT_SHAPE))
    for i in range(hp.Int('num_layers', 2, 4)):
        units = hp.Int(f'units_{i}', 64, 256, step=64)
        dropout_rate = hp.Float(f'dropout_{i}', 0.1, 0.4, step=0.1)
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='linear'))
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mae',
    max_trials=10,
    directory='tuning_results',
    project_name='keras_mlp_housing_updated'
)

tuner.search(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=100, batch_size=64,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.hypermodel.build(best_hp)
history = best_model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=150, batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=0
)

# Evaluate tuned model
y_pred_tuned_scaled = best_model.predict(X_test_processed, verbose=0).flatten()
y_pred_tuned_log = y_scaler.inverse_transform(y_pred_tuned_scaled.reshape(-1, 1)).flatten()
y_pred_tuned = np.expm1(y_pred_tuned_log)
y_pred_tuned = np.nan_to_num(y_pred_tuned, nan=y_test.mean())

mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
print(f"Tuned Keras Results:   MAE=${mae_tuned:,.2f}, R²={r2_tuned:.4f}, RMSE=${rmse_tuned:,.2f}")


# 8. COMPARISON SUMMARY
print("\n" + "="*70)
print("UNTUNED vs. TUNED COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': ['Untuned (Baseline)', 'Tuned (KerasTuner)'],
    'MAE': [mae_base, mae_tuned],
    'RMSE': [rmse_base, rmse_tuned],
    'R²': [r2_base, r2_tuned]
})
print(results_df.to_string(index=False))

# 9. SAVE BEST MODEL
os.makedirs('models', exist_ok=True)
best_model.save('models/keras_mlp_regressor_tuned.h5')
joblib.dump(preprocessor, 'models/preprocessor_keras_regression.pkl')
joblib.dump(y_scaler, 'models/y_scaler_keras_regression.pkl')
