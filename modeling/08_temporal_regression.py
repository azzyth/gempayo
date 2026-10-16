"""TAHAP "KAPAN" — REGRESI WAKTU KE GEMPA BERIKUTNYA (master_prompt_regression.md).

Target: `gap_days` = jumlah hari hingga gempa signifikan berikutnya (global,
urut kronologis). Split TIME-BASED (train = periode awal, test = periode akhir)
TANPA shuffle — mencegah lookahead.

Rantai metrik -> loss -> transform (FASE 0):
  - Metrik utama MAE (hari) -> loss L1 -> objective="mae" (tanpa transform).
  - Target right-skewed tapi MAE robust; RMSLE dilaporkan sekunder.
  - Ridge selalu hadir sebagai pembanding linear (RMSE-aligned).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "gempabumi_clean.csv"), encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ---- target: hari hingga gempa berikutnya ----
df["gap_days"] = (df["Date"].shift(-1) - df["Date"]).dt.days
df = df.dropna(subset=["gap_days"])  # event terakhir tak punya "next"
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

NUM_COLS = ["Magnitude", "Depth_km", "MMI_max", "Latitude", "Longitude", "Year", "Month"]
CAT_COLS = ["Region", "Tsunami"]

# ---- FASE 1: audit target ----
print("== AUDIT TARGET gap_days ==")
print(df["gap_days"].describe().round(2).to_string())
print(f"skew: {df['gap_days'].skew():.2f}")

# ---- time-based split (tanpa shuffle) ----
n = len(df)
cut = int(n * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"\ntrain: {len(train)} ({train.Date.min():%Y-%m-%d} .. {train.Date.max():%Y-%m-%d})")
print(f"test : {len(test)} ({test.Date.min():%Y-%m-%d} .. {test.Date.max():%Y-%m-%d})")

y_tr = train["gap_days"].values.astype(float)
y_te = test["gap_days"].values.astype(float)


def prep_features(d):
    X = d[NUM_COLS + CAT_COLS].copy()
    X["MMI_max"] = X["MMI_max"].fillna(0)  # tidak ada laporan rasa -> 0
    return X


pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])
X_tr = pre.fit_transform(prep_features(train))
X_te = pre.transform(prep_features(test))

# ---- baseline mean & median (titik nol) ----
for name, fn in [("mean", np.mean), ("median", np.median)]:
    pred = np.full(len(test), fn(y_tr))
    print(f"\nbaseline {name}: MAE={mean_absolute_error(y_te, pred):.1f} hari | "
          f"RMSE={mean_squared_error(y_te, pred) ** 0.5:.1f}")

# ---- Ridge (pembanding linear) ----
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
ridge_pred = np.clip(ridge.predict(X_te), 0, None)
print(f"\nRidge        : MAE={mean_absolute_error(y_te, ridge_pred):.1f} | "
      f"RMSE={mean_squared_error(y_te, ridge_pred) ** 0.5:.1f}")

# ---- LightGBM objective selaras metrik (MAE -> L1) ----
lgbm = lgb.LGBMRegressor(
    objective="mae", metric="mae", n_estimators=300, learning_rate=0.05,
    num_leaves=15, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=-1,
)
lgbm.fit(X_tr, y_tr)
lgbm_pred = np.clip(lgbm.predict(X_te), 0, None)
mae = mean_absolute_error(y_te, lgbm_pred)
rmse = mean_squared_error(y_te, lgbm_pred) ** 0.5
print(f"LightGBM(mae): MAE={mae:.1f} | RMSE={rmse:.1f}")

# RMSLE (sekunder, ruang log1p)
rmsle = mean_squared_error(np.log1p(y_te), np.log1p(lgbm_pred)) ** 0.5
print(f"LightGBM RMSLE: {rmsle:.4f}")

# ---- FASE 6: residual per decile target (test) ----
print("\n== ERROR PER DECILE TARGET (test, MAE hari) ==")
dec = pd.qcut(y_te, 5, labels=False, duplicates="drop")
for d in sorted(pd.Series(dec).dropna().unique()):
    m = dec == d
    print(f"decile {d}: n={m.sum():2d} | MAE={mean_absolute_error(y_te[m], lgbm_pred[m]):6.1f}")

# ---- contoh "kapan" untuk interpretasi ----
print("\n== CONTOH PREDIKSI TEST ==")
ex = test[["Date", "Region", "Magnitude", "gap_days"]].copy()
ex["pred_days"] = lgbm_pred.round(1)
print(ex.head(12).to_string(index=False))

# feature importance (Region mendominasi laju kejadian)
feat_names = pre.get_feature_names_out()
imp = pd.Series(lgbm.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\n== FEATURE IMPORTANCE (top 10) ==")
print(imp.head(10).round(3).to_string())

# ---- "kapan" yang actionable: jarak antar gempa per wilayah ----
print("\n== ARAHAN 'KAPAN' PER WILAYAH (median gap, hari) ==")
per_region = df.groupby("Region")["gap_days"].agg(["median", "mean", "count"]).round(1)
per_region.columns = ["median_hari", "mean_hari", "n"]
per_region = per_region.sort_values("median_hari")
print(per_region.to_string())

print("\nKESIMPULAN: gap (kapan gempa berikutnya) ~ proses Poisson. Model berbasis fitur")
print("tidak mengalahkan baseline median -> sinyal 'kapan' utamanya = LAJU PER WILAYAH")
print("(konsisten dengan occurrence_risk.csv), bukan fitur magnitudo/depth event.")
