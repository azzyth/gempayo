"""TAHAP 3 - VALIDATION STRATEGY & OOF SETUP (master_prompt_classification.md).

Deliverable: folds.csv (beku + sha256), metrik macro-F1 tervalidasi vs sklearn,
baseline majority = titik nol. Data kecil (307) -> StratifiedKFold(5) dibekukan;
RepeatedStratifiedKFold dipakai sebagai estimator stabilitas CV.
"""
import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "gempabumi_clean.csv"), encoding="utf-8-sig")
df["row_id"] = np.arange(len(df))
y = (df["Tsunami"] == "Tsunami").astype(int).values


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in (0, 1):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.mean(f1s))


# validasi metrik vs sklearn
_t = np.array([0, 0, 1, 1, 0, 1])
_p = np.array([0, 1, 0, 1, 1, 1])
assert abs(macro_f1(_t, _p) - f1_score(_t, _p, average="macro")) < 1e-9, "macro_f1 != sklearn"

# ---- fold beku ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_of = np.zeros(len(df), dtype=int)
for f, (_, va) in enumerate(skf.split(df, y)):
    fold_of[va] = f
df["fold"] = fold_of

folds = df[["row_id", "Tsunami", "fold"]].copy()
folds["label"] = folds["Tsunami"].map({"Tsunami": 1, "Tidak Tsunami": 0})
folds = folds[["row_id", "label", "fold"]]
folds.to_csv(os.path.join(ART, "folds.csv"), index=False)

print("distribusi kelas per fold:")
print(df.groupby("fold")["Tsunami"].value_counts().unstack().fillna(0).astype(int))

# leakage guard: setiap fold wajib memuat kedua kelas (stratifikasi bekerja)
assert (df.groupby("fold")["Tsunami"].value_counts().unstack().fillna(0).min().min()) >= 1

sha = hashlib.sha256(open(os.path.join(ART, "folds.csv"), "rb").read()).hexdigest()
print("folds.csv sha256:", sha[:16])

# ---- baseline majority (titik nol) ----
scores = []
for f in range(5):
    va = fold_of == f
    pred = np.zeros(va.sum(), dtype=int)  # mayoritas = 'Tidak Tsunami'
    scores.append(macro_f1(y[va], pred))
print("baseline majority macro-F1 per fold:", [round(s, 4) for s in scores])
print(f"baseline majority CV macro-F1 = {np.mean(scores):.4f} +- {np.std(scores):.4f}")

# estimator stabilitas (data kecil)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
rscores = []
for tr, va in rskf.split(df, y):
    pred = np.zeros(len(va), dtype=int)
    rscores.append(macro_f1(y[va], pred))
print(f"RepeatedStratifiedKFold(5x3) majority = {np.mean(rscores):.4f} +- {np.std(rscores):.4f}")
