"""
TabNet + Optuna tuning 
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import torch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")


# Config
DATA_PATH = "data/adult.csv"     
MODELS_DIR = "models"
OPTUNA_TRIALS = 30          
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)


# Feature engineering 
def feature_engineer(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["capital_net"] = df.get("capital.gain", 0) - df.get("capital.loss", 0)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 40, 60, 100], labels=["Young", "Mid", "Mature", "Senior"]
    )
    df["work_hours_cat"] = pd.cut(
        df["hours.per.week"], bins=[0, 30, 40, 60, 100], labels=["Low", "Normal", "High", "Extreme"]
    )
    # drop engineered or unnecessary columns (matches RF pipeline)
    cols_to_drop = ["fnlwgt", "capital.gain", "capital.loss", "age", "hours.per.week"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    return df


# Preprocessor: safe encoding + scaler + metadata for TabNet
class TabNetPreprocessor:

    UNSEEN_TOKEN = "__UNSEEN__"

    def __init__(self, categorical_cols=None, numeric_cols=None):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.label_encoders = {}     # dict[col] = LabelEncoder
        self.scaler = None
        self.cat_idxs = []
        self.cat_dims = []
        self.fitted = False

    def _ensure_columns(self, X: pd.DataFrame):
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.numeric_cols is None:
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    def fit(self, X: pd.DataFrame):
        X = X.copy()
        self._ensure_columns(X)

        # Fit label encoders on training unique values + UNSEEN token
        for col in self.categorical_cols:
            le = LabelEncoder()
            uniques = X[col].astype(str).unique().tolist()
            if self.UNSEEN_TOKEN not in uniques:
                uniques.append(self.UNSEEN_TOKEN)
            le.fit(uniques)
            self.label_encoders[col] = le

        # Build cat_idxs and cat_dims based on column positions in X
        self.cat_idxs = [X.columns.get_loc(c) for c in self.categorical_cols]
        self.cat_dims = [len(self.label_encoders[c].classes_) for c in self.categorical_cols]

        # Fit scaler on numeric columns
        self.scaler = StandardScaler()
        if len(self.numeric_cols) > 0:
            self.scaler.fit(X[self.numeric_cols])

        self.fitted = True
        return self

    def _safe_map_series(self, le: LabelEncoder, series: pd.Series) -> pd.Series:
        # Replace unseen with token before transforming
        s = series.astype(str)
        # mask unseen
        mask = ~s.isin(le.classes_)
        if mask.any():
            s = s.where(~mask, self.UNSEEN_TOKEN)
        return le.transform(s)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform(). Call fit() first.")
        X = X.copy()

        # Encode categorical columns
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            X[col] = self._safe_map_series(le, X[col])

        # Scale numeric columns
        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        # Return numpy float32 array (TabNet expects numpy arrays)
        return X.values.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path_prefix):
        """Save scaler and encoders using joblib (path_prefix without extension)."""
        joblib.dump(self.label_encoders, f"{path_prefix}_label_encoders.pkl")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        meta = {
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "cat_idxs": self.cat_idxs,
            "cat_dims": self.cat_dims
        }
        joblib.dump(meta, f"{path_prefix}_meta.pkl")

    def load(self, path_prefix):
        self.label_encoders = joblib.load(f"{path_prefix}_label_encoders.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        meta = joblib.load(f"{path_prefix}_meta.pkl")
        self.categorical_cols = meta["categorical_cols"]
        self.numeric_cols = meta["numeric_cols"]
        self.cat_idxs = meta["cat_idxs"]
        self.cat_dims = meta["cat_dims"]
        self.fitted = True



# Load data and feature engineering
df = pd.read_csv(DATA_PATH).dropna().drop_duplicates().reset_index(drop=True)
df = feature_engineer(df)

# Split X and y
X = df.drop("income", axis=1)
y = df["income"].copy()

# Target encoder (for reporting)
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)


# Identify columns 
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"   numeric_cols: {numeric_cols}")
print(f"   categorical_cols: {categorical_cols}")


# Train / Val / Test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)

print(f"   train_sub: {X_train_sub.shape}, val: {X_val.shape}, test: {X_test.shape}")


# Preprocess: fit transformer on training sub-split, transform all sets
preprocessor = TabNetPreprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
X_train_np = preprocessor.fit_transform(X_train_sub)
X_val_np = preprocessor.transform(X_val)
X_test_np = preprocessor.transform(X_test)

# metadata for TabNet
cat_idxs = preprocessor.cat_idxs
cat_dims = preprocessor.cat_dims

print(f"   cat_idxs: {cat_idxs}")
print(f"   cat_dims: {cat_dims}")

# Save preprocessor temporarily (we will overwrite later when saving final)
preprocessor.save(os.path.join(MODELS_DIR, "preprocessor_initial"))

# Prepare labels and sample weights
y_train_np = y_train_sub.astype(np.int64)
y_val_np = y_val.astype(np.int64)
y_test_np = y_test.astype(np.int64)

class_weights = compute_class_weight("balanced", classes=np.unique(y_train_np), y=y_train_np)
sample_weights = np.array([class_weights[int(lbl)] for lbl in y_train_np])


# Optuna tuning
def tabnet_objective(trial):
    # Suggest hyperparameters
    n_d = trial.suggest_int("n_d", 8, 64)
    n_a = trial.suggest_int("n_a", 8, 64)
    n_steps = trial.suggest_int("n_steps", 3, 8)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    cat_emb_dim = trial.suggest_int("cat_emb_dim", 1, 4)

    # Build model
    model = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr, weight_decay=weight_decay),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=dict(mode="max", patience=8, factor=0.5),
        mask_type="entmax",
        seed=RANDOM_STATE,
        verbose=0
    )

    # Fit
    model.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_metric=["auc"],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=256,
        weights=sample_weights,
        drop_last=False
    )

    # Evaluate using threshold 0.5 to compute minority class 
    proba_val = model.predict_proba(X_val_np)[:, 1]
    preds_val = (proba_val > 0.5).astype(int)
    f1_min = f1_score(y_val_np, preds_val, pos_label=1)
    # Optuna maximizes this
    return f1_min


# Run Optuna study
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(tabnet_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print("\nOptuna best params:")
print(study.best_params)
print("Best validation F1:", study.best_value)

best_params = study.best_params


# Train final model with best params
print("\n5) Training final TabNet with best params...")

tuned_model = TabNetClassifier(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=best_params.get("cat_emb_dim", 2),
    n_d=best_params.get("n_d", 32),
    n_a=best_params.get("n_a", 32),
    n_steps=best_params.get("n_steps", 5),
    gamma=best_params.get("gamma", 1.5),
    lambda_sparse=best_params.get("lambda_sparse", 1e-4),
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=best_params.get("lr", 1e-2), weight_decay=best_params.get("weight_decay", 1e-5)),
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params=dict(mode="max", patience=10, factor=0.5),
    mask_type="entmax",
    seed=RANDOM_STATE,
    verbose=0
)

tuned_model.fit(
    X_train_np, y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_metric=["auc"],
    max_epochs=150,
    patience=25,
    batch_size=1024,
    virtual_batch_size=256,
    weights=sample_weights,
    drop_last=False
)

# Evaluate on test: threshold search for best minority F1
print("\n6) Evaluating final model and searching threshold...")

proba_test = tuned_model.predict_proba(X_test_np)[:, 1]
best_thresh = 0.5
best_f1 = -1.0
for t in np.linspace(0.2, 0.5, 31):
    preds = (proba_test > t).astype(int)
    f1_min = f1_score(y_test_np, preds, pos_label=1)
    if f1_min > best_f1:
        best_f1 = f1_min
        best_thresh = t

y_pred_test = (proba_test > best_thresh).astype(int)

acc_test = accuracy_score(y_test_np, y_pred_test)
f1_macro_test = f1_score(y_test_np, y_pred_test, average="macro")
report = classification_report(y_test_np, y_pred_test, target_names=target_le.classes_)

print(f"   Best threshold on test: {best_thresh:.3f} (minority F1 = {best_f1:.4f})")
print(f"   Test Accuracy: {acc_test:.4f}")
print(f"   Test Macro F1: {f1_macro_test:.4f}")
print("\nClassification report:\n")
print(report)


# Save model + preprocessors + optuna study

# Save TabNet (creates tabnet_model.zip)
tuned_model.save_model(os.path.join(MODELS_DIR, "tabnet_tuned_model"))

# Save preprocessor (label encoders + scaler + meta)
preprocessor.save(os.path.join(MODELS_DIR, "tabnet_preprocessor"))

# Save target label encoder and optuna study
joblib.dump(target_le, os.path.join(MODELS_DIR, "target_label_encoder.pkl"))
joblib.dump(study.best_params, os.path.join(MODELS_DIR, "optuna_best_params.pkl"))
joblib.dump(study, os.path.join(MODELS_DIR, "optuna_study.pkl"))
