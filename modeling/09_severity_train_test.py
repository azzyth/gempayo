"""TAHAP "SEBERAPA BESAR" — KLASIFIKASI SEVERITY (master_prompt_classification.md).

Train/test split STRATIFIED (80/20) + OOF di train + macro-F1 di test.
Fitur: Magnitude, Depth_km, MMI_max, lokasi, Tsunami, Region.
Model: LogisticRegression (baseline linear) + LightGBM (class_weight balanced).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "gempabumi_model.csv"), encoding="utf-8-sig")
y = df["Severity"].values

NUM_COLS = ["Magnitude", "Depth_km", "MMI_max", "Latitude", "Longitude"]
CAT_COLS = ["Tsunami", "Region"]
CLASSES = [0, 1, 2]
CLASS_NAMES = {0: "Ringan", 1: "Sedang", 2: "Berat"}


def prep_features(d):
    X = d[NUM_COLS + CAT_COLS].copy()
    X["MMI_max"] = X["MMI_max"].fillna(0)
    return X


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


X = prep_features(df)

# ---- train/test stratified ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print("distribusi kelas train vs test:")
print(pd.DataFrame({
    "train": pd.Series(y_tr).value_counts().sort_index(),
    "test": pd.Series(y_te).value_counts().sort_index(),
}).to_string())

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])
X_tr = pre.fit_transform(X_tr)
X_te = pre.transform(X_te)

# ---- baseline majority (titik nol) ----
maj = pd.Series(y_tr).mode()[0]
print(f"\nbaseline majority macro-F1 (test) = {macro_f1(y_te, np.full(len(y_te), maj)):.4f}")

# ---- LogisticRegression ----
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lr.fit(X_tr, y_tr)
lr_pred = lr.predict(X_te)
print(f"LogisticRegression macro-F1 (test) = {macro_f1(y_te, lr_pred):.4f}")
print("per-class F1:", {CLASS_NAMES[c]: round(f1_score(y_te, lr_pred, labels=[c], average='macro', zero_division=0), 3) for c in CLASSES})

# ---- OOF di train (fold beku) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_tr), dtype=int)
for tr, va in skf.split(X_tr, y_tr):
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    m.fit(X_tr[tr], y_tr[tr])
    oof[va] = m.predict(X_tr[va])
print(f"\nOOF train macro-F1 = {macro_f1(y_tr, oof):.4f}")

# ---- LightGBM ----
lgbm = lgb.LGBMClassifier(
    objective="multiclass", num_class=3, n_estimators=200, learning_rate=0.05,
    num_leaves=15, min_child_samples=10, class_weight="balanced",
    random_state=42, verbosity=-1,
)
lgbm.fit(X_tr, y_tr)
lgbm_pred = lgbm.predict(X_te)
print(f"\nLightGBM macro-F1 (test) = {macro_f1(y_te, lgbm_pred):.4f}")
print("per-class F1:", {CLASS_NAMES[c]: round(f1_score(y_te, lgbm_pred, labels=[c], average='macro', zero_division=0), 3) for c in CLASSES})

# ---- confusion matrix (LightGBM) ----
print("\nconfusion matrix (test, LightGBM) [baris=aktual, kolom=prediksi]:")
print(confusion_matrix(y_te, lgbm_pred, labels=CLASSES))
print("\ndistribusi prediksi test:", dict(pd.Series(lgbm_pred).value_counts(normalize=True).round(3)))
print("distribusi aktual test   :", dict(pd.Series(y_te).value_counts(normalize=True).round(3)))
