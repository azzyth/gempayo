"""TAHAP 1 lanjutan - investigasi baris bermasalah (longitude absurd, label NaN)."""
import os
import pandas as pd
import re

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "gempabumi_2000-2023.csv"), encoding="utf-8-sig")

def split_wilayah(s):
    s = str(s).strip()
    m = re.search(r"\s+(Tidak\s+)?Tsunami$", s, flags=re.I)
    if not m:
        return s, None
    return s[: m.start()].strip(), ("Tsunami" if not m.group(1) else "Tidak Tsunami")

df[["Wilayah_nama", "Tsunami"]] = df["Wilayah"].apply(lambda s: pd.Series(split_wilayah(s)))
for c in ["Latitude", "Longitude", "Depth_km", "Magnitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

cols_show = ["Tahun", "No", "Tanggal", "Wilayah", "Origin_Time", "Latitude", "Longitude", "Depth_km", "Magnitude"]

print("== 1. BARIS LONGITUDE ABSURD (>150) ==")
bad = df[(df.Longitude > 150) | (df.Longitude < 85)]
print(bad[cols_show].to_string(index=False))

print("\n== 2. BARIS LABEL TSUNAMI NaN (Wilayah tanpa tag) ==")
nanlab = df[df.Tsunami.isna()]
print(nanlab[cols_show].to_string(index=False))

print("\n== 3. BARIS KOORDINAT MISSING ==")
print(df[df.Latitude.isna() | df.Longitude.isna()][cols_show].to_string(index=False))

print("\n== 4. LATITUDE DI LUAR INDONESIA (-11..6.5) ==")
print(df[(df.Latitude > 6.5) | (df.Latitude < -11)][cols_show].to_string(index=False))

print("\n== 5. TSUNAMI: magnitude & depth vs non-tsunami (sanity domain) ==")
print(df.groupby("Tsunami")[["Magnitude", "Depth_km"]].agg(["min", "median", "max"]))

print("\n== 6. TSA NAMA WILAYAH: tsunami events, koordinatnya laut atau darat? ==")
ts = df[df.Tsunami == "Tsunami"]
print(ts[["Tahun", "Tanggal", "Wilayah", "Magnitude", "Depth_km"]].to_string(index=False))
