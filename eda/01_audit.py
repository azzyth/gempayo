"""TAHAP 1 - LOAD & AUDIT (master_prompt_classification.md sec 1.x).

Audit target: kolom Wilayah memuat '<nama wilayah> <Tsunami|Tidak Tsunami>'.
Checkpoint: tabel distribusi kelas + label konflik + struktur + missing.
"""
import os
import pandas as pd
import numpy as np
import re

pd.set_option("display.width", 200)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "gempabumi_2000-2023.csv"), encoding="utf-8-sig")
print(f"shape: {df.shape}")
print(f"kolom: {list(df.columns)}")

# ---- pisahkan label tsunami dari kolom Wilayah ----
def split_wilayah(s):
    s = str(s).strip()
    m = re.search(r"\s+(Tidak\s+)?Tsunami$", s, flags=re.I)
    if not m:
        return s, None
    return s[: m.start()].strip(), ("Tsunami" if not m.group(1) else "Tidak Tsunami")

df[["Wilayah_nama", "Tsunami"]] = df["Wilayah"].apply(
    lambda s: pd.Series(split_wilayah(s))
)

print("\n== DISTRIBUSI KELAS (Tsunami) ==")
dist = df["Tsunami"].value_counts(dropna=False)
print(pd.DataFrame({"count": dist, "prop": (dist / len(df)).round(4)}))
print(f"baseline majority macro-F1 titik nol -> Tidak Tsunami = {dist.max()} baris")

# ---- parse numerik ----
for c in ["Latitude", "Longitude", "Depth_km", "Magnitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")

print("\n== DESKRIPSI NUMERIK ==")
print(df[["Latitude", "Longitude", "Depth_km", "Magnitude", "Tahun"]].describe().round(3).T)

print("\n== MISSING PER KOLOM ==")
miss = df.isna().sum()
print(miss[miss > 0].to_frame("n_missing") if miss.sum() else "tidak ada missing")

print("\n== MISSING PER KOLOM x KELAS (missing as signal?) ==")
for c in ["Latitude", "Longitude", "Depth_km", "Magnitude"]:
    g = df.groupby("Tsunami")[c].apply(lambda x: x.isna().mean().round(3))
    if (g > 0).any():
        print(f"{c}: {g.to_dict()}")

# ---- rentang & validitas ----
print("\n== VALIDITAS NILAI ==")
print(f"lat  range: [{df.Latitude.min():.2f}, {df.Latitude.max():.2f}] (wajar Indonesia: -11..6)")
print(f"long range: [{df.Longitude.min():.2f}, {df.Longitude.max():.2f}] (wajar: 95..141)")
print(f"depth range: [{df.Depth_km.min()}, {df.Depth_km.max()}] km; negatif/absurd: {(df.Depth_km<0).sum()}")
print(f"mag   range: [{df.Magnitude.min()}, {df.Magnitude.max()}]; <3.5: {(df.Magnitude<3.5).sum()}")
print(f"mag==0: {(df.Magnitude==0).sum()}, depth==0: {(df.Depth_km==0).sum()}")

# ---- label konflik: fitur identik, label beda ----
feat_cols = ["Origin_Time", "Latitude", "Longitude", "Depth_km", "Magnitude", "Tanggal"]
dups = df.groupby(feat_cols, dropna=False).agg(
    n=("Tsunami", "size"), n_label=("Tsunami", "nunique")
)
conflict = dups[(dups.n > 1) & (dups.n_label > 1)]
print(f"\n== LABEL KONFLIK (fitur identik, label beda): {len(conflict)} grup ==")
if len(conflict):
    print(conflict)
    print("-> estimasi noise floor untuk ceiling diagnosis")

# ---- duplikat persis ----
print(f"duplikat baris persis (semua kolom): {df.duplicated().sum()}")
print(f"duplikat fitur (tanpa label): {df.duplicated(subset=feat_cols).sum()}")

# ---- audit kolom bebas ----
print("\n== TEKS BEBAS ==")
print(f"Wilayah_nama unik: {df.Wilayah_nama.nunique()}")
print(df["Wilayah_nama"].value_counts().head(10).to_frame("n"))
print(f"Sumber unik: {df['Sumber'].nunique()} -> {df['Sumber'].value_counts().head(5).to_dict()}")
print(f"baris dengan Wilayah_yang_merasakan kosong: {df['Wilayah_yang_merasakan'].isna().sum()}")
print(f"baris dengan Korban_Kerusakan kosong: {df['Korban_Kerusakan'].isna().sum()}")

# ---- per tahun (time structure) ----
print("\n== PER TAHUN ==")
per_year = df.groupby("Tahun").agg(
    n=("Tsunami", "size"), tsunami=("Tsunami", lambda s: (s == "Tsunami").sum())
)
print(per_year.T)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_audit_clean.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print("\nsaved:", out)
