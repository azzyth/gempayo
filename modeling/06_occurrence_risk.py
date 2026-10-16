"""KOMPONEN "KAPAN" (reframe jujur) - RISIKO/FREKUENSI KEJADIAN.

Prediksi tanggal/jam gempa tidak ilmiah. Yang dibangun = laju kejadian per
wilayah (Poisson) + pola temporal. Output interpretabel untuk prototype:
P(>=1 gempa signifikan) per wilayah per tahun.

CATATAN PENTING: katalog BMKG ini = gempa SIGNIFIKAN (yang dirasakan/merusak),
bukan seluruh seismisitas. Jadi "frekuensi" = frekuensi gempa signifikan.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "gempabumi_clean.csv"), encoding="utf-8-sig")
df["Year"] = pd.to_datetime(df["Date"]).dt.year
df["Month"] = pd.to_datetime(df["Date"]).dt.month

YEARS = df["Year"].nunique()
print(f"periode: {int(df.Year.min())}-{int(df.Year.max())} ({YEARS} tahun), {len(df)} gempa signifikan")

# ---- 1. laju kejadian per wilayah (Poisson) ----
reg = df.groupby("Region").agg(
    n=("Year", "size"),
    first_year=("Year", "min"),
    last_year=("Year", "max"),
    max_mag=("Magnitude", "max"),
).reset_index()
reg["lambda_per_year"] = (reg["n"] / YEARS).round(3)
reg["p_ge1_next_year"] = (1 - np.exp(-reg["lambda_per_year"])).round(3)
reg = reg.sort_values("lambda_per_year", ascending=False)

print("\n== RISIKO KEJADIAN PER WILAYAH (laju Poisson) ==")
print(reg[["Region", "n", "first_year", "last_year", "lambda_per_year", "p_ge1_next_year", "max_mag"]]
      .to_string(index=False))
reg.to_csv(os.path.join(ART, "occurrence_risk.csv"), index=False)

# ---- 2. tren tahunan ----
yr = df.groupby("Year").size()
print("\n== TREN TAHUNAN (count) ==")
print(yr.to_string())
print(f"rata-rata/tahun: {yr.mean():.1f}, std: {yr.std():.1f}")

# ---- 3. musiman bulanan ----
mo = df.groupby("Month").size().reindex(range(1, 13), fill_value=0)
print("\n== SEASONALITAS BULANAN (count) ==")
print(mo.to_string())
print(f"bulan paling aktif: {mo.idxmax()}, paling sepi: {mo.idxmin()}")
