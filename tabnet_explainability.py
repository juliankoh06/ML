"""
EXPLAINABILITY & PERFORMANCE ANALYSIS FOR TABNET CLASSIFIER
"""

import os
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
from pytorch_tabnet.tab_model import TabNetClassifier

# ensure output folder exists
os.makedirs("figures", exist_ok=True)

# --- Custom preprocessor class (load-only variant used in explain script) ---
class TabNetPreprocessor:
    UNSEEN_TOKEN = "__UNSEEN__"

    def __init__(self, categorical_cols=None, numeric_cols=None):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.label_encoders = {}
        self.scaler = None
        self.cat_idxs = []
        self.cat_dims = []
        self.fitted = False

    def _safe_map_series(self, le, series: pd.Series) -> pd.Series:
        s = series.astype(str)
        mask = ~s.isin(le.classes_)
        if mask.any():
            s = s.where(~mask, self.UNSEEN_TOKEN)
        return le.transform(s)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be loaded before transform(). Call load() first.")
        X = X.copy()
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            X[col] = self._safe_map_series(le, X[col])
        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X.values.astype(np.float32)

    def load(self, path_prefix):
        self.label_encoders = joblib.load(f"{path_prefix}_label_encoders.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        meta = joblib.load(f"{path_prefix}_meta.pkl")
        self.categorical_cols = meta["categorical_cols"]
        self.numeric_cols = meta["numeric_cols"]
        self.cat_idxs = meta["cat_idxs"]
        self.cat_dims = meta["cat_dims"]
        self.fitted = True
        return self

# ============================================================
# 1. LOAD DATA + FEATURE ENGINEERING
# ============================================================
def feature_engineer(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["capital_net"] = df.get("capital.gain", 0) - df.get("capital.loss", 0)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 40, 60, 100], labels=["Young", "Mid", "Mature", "Senior"]
    )
    df["work_hours_cat"] = pd.cut(
        df["hours.per.week"], bins=[0, 30, 40, 60, 100], labels=["Low", "Normal", "High", "Extreme"]
    )
    cols_to_drop = ["fnlwgt", "capital.gain", "capital.loss", "age", "hours.per.week"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    return df

df = pd.read_csv("data/adult.csv").dropna().drop_duplicates().reset_index(drop=True)
df_eng = feature_engineer(df)

X = df_eng.drop("income", axis=1)
y = df_eng["income"]

target_encoder = joblib.load("models/target_label_encoder.pkl")
y_encoded = target_encoder.transform(y)

_, X_test, _, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ============================================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================================
print("Loading TabNet model and preprocessor...")

preprocessor = TabNetPreprocessor()
preprocessor.load(os.path.join("models", "tabnet_preprocessor"))

# Load the tuned TabNet model (saved by TabNet.save_model -> tabnet_tuned_model.zip)
tuned_model = TabNetClassifier()
tuned_model.load_model(os.path.join("models", "tabnet_tuned_model.zip"))

# Preprocess the test data
X_test_np = preprocessor.transform(X_test)

# ============================================================ 
# 3. EVALUATION
# ============================================================ 
print("\nEvaluating model performance...")

y_pred_test = tuned_model.predict(X_test_np)

accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average="macro")
recall = recall_score(y_test, y_pred_test, average="macro")
f1 = f1_score(y_test, y_pred_test, average="macro")

print("\n==============================================================")
print("TUNED TABNET - PERFORMANCE METRICS")
print("==============================================================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1-score (Macro): {f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_test, target_names=target_encoder.classes_))

# ============================================================
# 4. FEATURE IMPORTANCE (TabNet explain -> masks)
# ============================================================
print("\nGenerating TabNet feature importance (feature masks)...")

# TabNet explain returns a tuple: (explain_matrix, masks) in most versions.
# We'll handle both possible return formats robustly.
explain_result = tuned_model.explain(X_test_np)

# Unpack safely
if isinstance(explain_result, tuple) and len(explain_result) >= 1:
    explain_matrix = explain_result[0]
else:
    explain_matrix = explain_result  # fallback if explain returns array directly

# explain_matrix shape should be (n_samples, n_features). If it's nested per-step, try to aggregate.
explain_matrix = np.asarray(explain_matrix)

# If explain_matrix has shape (n_samples, n_steps, n_features) average across steps
if explain_matrix.ndim == 3:
    # average over steps dimension (axis=1) -> (n_samples, n_features)
    explain_matrix = explain_matrix.mean(axis=1)

# Now compute global importance by averaging across samples
global_importance = explain_matrix.mean(axis=0)

# Build dataframe and keep top-10 for readability
feat_imp_df = pd.DataFrame({
    'feature': X.columns,
    'importance': global_importance
}).sort_values(by='importance', ascending=False).head(10)

print("\nTop 10 Most Important Features (TabNet):")
print(feat_imp_df)

# Plot Top-10
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_imp_df, color='blue')
plt.title('TabNet Feature Importances (Top 10)', fontsize=14)
plt.xlabel('Importance', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.tight_layout()
out_path = 'figures/tabnet_feature_importance_top10.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved feature importance plot to {out_path}")
plt.show()

# ============================================================ 
# 5. CONFUSION MATRIX
# ============================================================ 
print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title('TabNet Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('True Label', fontsize=11)
plt.tight_layout()
out_cm = 'figures/tabnet_confusion_matrix.png'
plt.savefig(out_cm, dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix to {out_cm}")
plt.show()

print("\nExplainability analysis complete.")
