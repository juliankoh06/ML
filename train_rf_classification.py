"""
RANDOM FOREST CLASSIFIER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import warnings
import os 
warnings.filterwarnings('ignore')

print("="*70)
print("CLASSIFICATION - RANDOM FOREST")
print("="*70)


# Preprocess Dataset and Feature engineering
df = pd.read_csv('data/adult.csv').dropna().drop_duplicates().reset_index(drop=True)
def feature_engineer(df_in):
    df_out = df_in.copy()
    
    # Create new features
    df_out['capital_net'] = df_out['capital.gain'] - df_out['capital.loss']
    df_out['age_group'] = pd.cut(df_out['age'], bins=[0, 25, 40, 60, 100],
                                 labels=['Young', 'Mid', 'Mature', 'Senior'])
    df_out['work_hours_cat'] = pd.cut(df_out['hours.per.week'], bins=[0, 30, 40, 60, 100],
                                      labels=['Low', 'Normal', 'High', 'Extreme'])
    
    # Drop original columns that have been engineered or are not needed
    cols_to_drop = ['fnlwgt', 'capital.gain', 'capital.loss', 'age', 'hours.per.week']
    df_out = df_out.drop(columns=cols_to_drop, errors='ignore')
    return df_out

# Apply feature engineering
df_engineered = feature_engineer(df)

# Separate X and y from the engineered data
X = df_engineered.drop('income', axis=1)
y = df_engineered['income']

# Encode the *target variable* (y)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)


# Identify feature names from the engineered dataframe
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"   Numeric features: {numeric_features}")
print(f"   Categorical features: {categorical_features}")

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# 3. TRAIN-TEST SPLIT

print("\n3. Splitting data...")
# Split the *raw* X and *encoded* y
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"   Training features shape: {X_train.shape}")
print(f"   Test features shape: {X_test.shape}")


# 4. UNTUNED MODEL (BASELINE)
print("\n" + "="*70)
print("Training UNTUNED Model (Baseline)")
print("="*70)

# Create pipeline
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
])

# Fit the pipeline
base_rf = base_pipeline.fit(X_train, y_train)

# Evaluate baseline
y_pred_base = base_rf.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base, average='macro') 

print(f"Untuned RF Results: Accuracy={acc_base:.4f}, F1-Score (Macro)={f1_base:.4f}")


# 5. TUNED MODEL (GRIDSEARCHCV)
print("\n" + "="*70)
print("Training TUNED Model (GridSearchCV)")
print("="*70)
print("\nPerforming GridSearchCV...")

# Create the pipeline for tuning
tuned_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': ['balanced']
}
# GridSearchCV call 
grid = GridSearchCV(
    estimator=tuned_pipeline, 
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
# Fit on the raw training data
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print("\nBest Parameters:")
for k, v in grid.best_params_.items():
    print(f"   {k}: {v}")


# 5. Evaluate Tuned Model
print("\n5. Evaluating tuned model...")
y_pred_tuned = best_rf.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, average='macro') 

print("\nTuned Model Results:")
print(f"   Accuracy: {acc_tuned:.4f}")
print(f"   F1-Score (Macro): {f1_tuned:.4f}")


# 6. COMPARISON SUMMARY
print("\n" + "="*70)
print("UNTUNED vs. TUNED COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': ['Untuned (Baseline)', 'Tuned (GridSearchCV)'],
    'Accuracy': [acc_base, acc_tuned],
    'F1-Score (Macro)': [f1_base, f1_tuned]
})
print(results_df.to_string(index=False))

print("\n--- TUNED MODEL CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred_tuned, target_names=target_encoder.classes_))

# 7. Save Model
os.makedirs('models', exist_ok=True) 

# 'best_rf' IS the full pipeline (preprocessor + classifier)
joblib.dump(best_rf, 'models/random_forest_pipeline.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')

