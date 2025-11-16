"""
UNTUNED TabNet Baseline
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import torch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")


# CONFIG
DATA_PATH = "data/adult.csv"   
MODELS_DIR = "models"
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 80)
print("UNTUNED TABNET BASELINE")
print("=" * 80)


# Feature engineering
def feature_engineer(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["capital_net"] = df.get("capital.gain", 0) - df.get("capital.loss", 0)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 40, 60, 100],
        labels=["Young", "Mid", "Mature", "Senior"]
    )
    df["work_hours_cat"] = pd.cut(
        df["hours.per.week"], bins=[0, 30, 40, 60, 100],
        labels=["Low", "Normal", "High", "Extreme"]
    )
    cols_to_drop = ["fnlwgt", "capital.gain", "capital.loss", "age", "hours.per.week"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


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

    def _ensure_columns(self, X):
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.numeric_cols is None:
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    def fit(self, X):
        X = X.copy()
        self._ensure_columns(X)

        for col in self.categorical_cols:
            le = LabelEncoder()
            uniques = X[col].astype(str).unique().tolist()
            if self.UNSEEN_TOKEN not in uniques:
                uniques.append(self.UNSEEN_TOKEN)
            le.fit(uniques)
            self.label_encoders[col] = le

        self.cat_idxs = [X.columns.get_loc(c) for c in self.categorical_cols]
        self.cat_dims = [len(self.label_encoders[c].classes_) for c in self.categorical_cols]

        self.scaler = StandardScaler()
        if len(self.numeric_cols) > 0:
            self.scaler.fit(X[self.numeric_cols])

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Fit first!")
        X = X.copy()

        for col in self.categorical_cols:
            le = self.label_encoders[col]
            s = X[col].astype(str)
            mask = ~s.isin(le.classes_)
            s = s.where(~mask, self.UNSEEN_TOKEN)
            X[col] = le.transform(s)

        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        return X.values.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# LOAD DATA
df = pd.read_csv(DATA_PATH).dropna().drop_duplicates().reset_index(drop=True)
df = feature_engineer(df)

X = df.drop("income", axis=1)
y = df["income"]

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols)


# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, " Test:", X_test.shape)


# PREPROCESS
preprocessor = TabNetPreprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
X_train_np = preprocessor.fit_transform(X_train)
X_test_np = preprocessor.transform(X_test)

cat_idxs = preprocessor.cat_idxs
cat_dims = preprocessor.cat_dims


# BASELINE TABNET MODEL 
print("\nTraining untuned baseline TabNet...\n")

baseline_model = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,      
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.3,
    lambda_sparse=1e-6,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    seed=RANDOM_STATE,
    verbose=1
)

baseline_model.fit(
    X_train_np, y_train,
    eval_set=[(X_test_np, y_test)],
    eval_metric=["accuracy"],
    max_epochs=50,
    patience=10,
    batch_size=1024,
)


# EVALUATION
proba_test = baseline_model.predict_proba(X_test_np)[:, 1]
y_pred = (proba_test > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n=== UNTUNED TABNET RESULTS ===")
print("Accuracy:", acc)
print("Macro F1:", f1_macro)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# SAVE
baseline_model.save_model(os.path.join(MODELS_DIR, "tabnet_untuned_model"))
preprocessor.save(os.path.join(MODELS_DIR, "tabnet_untuned_preprocessor"))
joblib.dump(target_le, os.path.join(MODELS_DIR, "tabnet_target_encoder.pkl"))

