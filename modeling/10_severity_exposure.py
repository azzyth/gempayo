"""FINE-TUNE SEVERITY DENGAN FITUR EXPOSURE (master_prompt_classification.md).

Bandingkan LightGBM base vs base+exposure pada split stratified 80/20 yang sama
(random_state=42) — exposure = jarak & populasi kota (data GeoNames).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp = pd.read_csv(os.path.join(ROOT, "gempabumi_exposure.csv"), encoding="utf-8-sig")
sev = pd.read_csv(os.path.join(ROOT, "gempabumi_model.csv"), encoding="utf-8-sig")["Severity"]
df = exp.copy()
df["Severity"] = sev.values
y = df["Severity"].values

BASE_NUM = ["Magnitude", "Depth_km", "MMI_max", "Latitude", "Longitude"]
EXP_NUM = ["dist_nearest_settlement_km", "n_settlements_50km", "n_settlements_100km", "log_pop_100km"]
CAT = ["Tsunami", "Region"]


def prep(d, nums):
    X = d[nums + CAT].copy()
    X["MMI_max"] = X["MMI_max"].fillna(0)
    return X


def run_model(nums, label):
    X = prep(df, nums)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    X_tr = pre.fit_transform(X_tr)
    X_te = pre.transform(X_te)
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=200, learning_rate=0.05,
        num_leaves=15, min_child_samples=10, class_weight="balanced",
        random_state=42, verbosity=-1,
    )
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    mf = f1_score(y_te, pred, labels=[0, 1, 2], average="macro", zero_division=0)
    per = {f"kelas{c}": round(f1_score(y_te, pred, labels=[c], average="macro", zero_division=0), 3)
           for c in [0, 1, 2]}
    print(f"{label:<18} macro-F1={mf:.4f}  {per}")
    return m, mf


print("== FINE-TUNE SEVERITY: base vs base+exposure vs exposure(tanpa lat/lon) ==")
m_base, f_base = run_model(BASE_NUM, "base")
m_exp, f_exp = run_model(BASE_NUM + EXP_NUM, "base+exposure")
m_geo, f_geo = run_model(["Magnitude", "Depth_km", "MMI_max"] + EXP_NUM, "exposure(no latlon)")
print(f"\nselisih base+exposure : {f_exp - f_base:+.4f} macro-F1")
print(f"selisih exposure(no-latlon) vs base: {f_geo - f_base:+.4f} macro-F1")

# feature importance (base+exposure)
feat_names = (BASE_NUM + EXP_NUM +
              list(OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                   .fit(df[CAT]).get_feature_names_out(CAT)))
imp = pd.Series(m_exp.booster_.feature_importance(importance_type="gain"),
                index=feat_names).sort_values(ascending=False)
print("\n== FEATURE IMPORTANCE base+exposure (top 12) ==")
print(imp.head(12).round(0).astype(int).to_string())
