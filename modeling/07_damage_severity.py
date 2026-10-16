"""KOMPONEN "SEBERAPA BESAR KERUSAKAN" - KLASIFIKASI SEVERITY DAMPAK.

Target Severity (3 kelas, dari korban meninggal):
  0 = Ringan (0 korban jiwa)
  1 = Sedang (1-9 korban jiwa)
  2 = Berat (>=10 korban jiwa)

Fitur (grounded EDA): Magnitude, Depth_km, MMI_max, lokasi (lat/lon), Region, Tsunami.
Model: baseline majority + rule fisik interpretabel + linear (LogisticRegression).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "gempabumi_clean.csv"), encoding="utf-8-sig")

# ---- target Severity ----
def severity_from_fatalities(v):
    if pd.isna(v) or v < 1:
        return 0
    if v < 10:
        return 1
    return 2

df["Severity"] = df["Korban_meninggal_est"].apply(severity_from_fatalities)
y = df["Severity"].values

print("distribusi Severity (0=ringan, 1=sedang, 2=berat):")
print(pd.DataFrame({"n": df["Severity"].value_counts().sort_index(),
                    "prop": df["Severity"].value_counts(normalize=True).sort_index().round(3)}))

df.to_csv(os.path.join(ROOT, "gempabumi_model.csv"), index=False, encoding="utf-8-sig")

# ---- fitur ----
num_cols = ["Magnitude", "Depth_km", "MMI_max", "Latitude", "Longitude"]
cat_cols = ["Tsunami", "Region"]
X = df[num_cols + cat_cols].copy()
X["MMI_max"] = X["MMI_max"].fillna(0)  # tidak ada laporan rasa -> 0 (konstanta, bukan belajar)

CLASSES = [0, 1, 2]

def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in CLASSES:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.mean(f1s))

# validasi metrik vs sklearn
_t = np.array([0, 0, 1, 1, 2, 2, 0])
_p = np.array([0, 1, 1, 1, 2, 0, 0])
assert abs(macro_f1(_t, _p) - f1_score(_t, _p, labels=CLASSES, average="macro")) < 1e-9

# ---- fold beku (stratify severity) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = [(tr, va) for tr, va in skf.split(X, y)]

# ---- baseline majority ----
scores = []
for tr, va in folds:
    maj = pd.Series(y[tr]).mode()[0]
    scores.append(macro_f1(y[va], np.full(len(va), maj)))
print(f"\nbaseline majority macro-F1 = {np.mean(scores):.4f} +- {np.std(scores):.4f}")

# ---- rule fisik interpretabel ----
def rule_pred(d):
    r = np.zeros(len(d), dtype=int)
    ts = (d["Tsunami"] == "Tsunami").values
    strong_shallow = ((d["Magnitude"] >= 7.0) & (d["Depth_km"] <= 40)).values
    moderate = ((d["Magnitude"] >= 6.0) & (d["Depth_km"] <= 40)).values
    r[moderate] = 1
    r[strong_shallow | ts] = 2
    return r

scores = []
for tr, va in folds:
    scores.append(macro_f1(y[va], rule_pred(X.iloc[va])))
print(f"rule fisik macro-F1 = {np.mean(scores):.4f} +- {np.std(scores):.4f}")

# ---- linear baseline ----
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
clf = Pipeline([("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])

oof = np.zeros(len(df), dtype=int)
scores = []
for tr, va in folds:
    clf.fit(X.iloc[tr], y[tr])
    p = clf.predict(X.iloc[va])
    oof[va] = p
    scores.append(macro_f1(y[va], p))
print(f"LogisticRegression macro-F1 = {np.mean(scores):.4f} +- {np.std(scores):.4f}")

print("\nper-class F1 (OOF, LogisticRegression):")
for c in CLASSES:
    print(f"  kelas {c}: {f1_score(y, oof, labels=[c], average='macro', zero_division=0):.3f}")

print("\nconfusion matrix (OOF) [baris=aktual, kolom=prediksi]:")
print(confusion_matrix(y, oof, labels=CLASSES))

# collapse / over-correction guard
dist = pd.Series(oof).value_counts(normalize=True).reindex(CLASSES, fill_value=0)
print("\ndistribusi prediksi OOF:", dict(dist.round(3)))
print("distribusi aktual     :", dict(pd.Series(y).value_counts(normalize=True).round(3)))

np.save(os.path.join(ART, "severity_oof_logreg.npy"), oof)
df["Severity_oof"] = oof
print("\nsaved: artifacts/severity_oof_logreg.npy")
