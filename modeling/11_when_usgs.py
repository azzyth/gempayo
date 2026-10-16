"""TAHAP "KAPAN" — LAJU SEISMISITAS DARI KATALOG USGS (lebih lengkap).

BMKG hanya 307 gempa signifikan; USGS M>=5.0 memberi ~5.700 event -> laju
kejadian & gap antar-gempa jauh lebih akurat untuk "kapan".
Output: artifacts/usgs_occurrence_risk.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

usgs = pd.read_csv(os.path.join(DATA, "usgs_earthquakes_clean.csv"))
usgs["time"] = pd.to_datetime(usgs["time"], format="ISO8601", utc=True)
usgs = usgs.sort_values("time").reset_index(drop=True)

# region (sama dengan 04_eda)
def region_of(lat, lon):
    if lon < 105 and lat < -2: return "Sumatra/NSB-barat"
    if lon < 106: return "Sumatra utara"
    if lat < -9 and lon < 130: return "NTT/Timor-selatan"
    if lat < -4 and 105 <= lon < 120: return "Jawa-selatan"
    if 120 <= lon < 130 and lat < -3: return "Nusa-selatan"
    if lon >= 128 and lat < -2: return "Maluku/Banda"
    if lat >= -2: return "Sulawesi-utara/Papua-barat"
    return "lain"

usgs["Region"] = [region_of(a, b) for a, b in zip(usgs["latitude"], usgs["longitude"])]
usgs["Year"] = usgs["time"].dt.year

YEARS = usgs["Year"].max() - usgs["Year"].min() + 1

# inter-event gap (global & per region)
usgs["gap_days"] = usgs["time"].diff().dt.total_seconds() / 86400

reg = usgs.groupby("Region").agg(
    n=("mag", "size"),
    lambda_per_year=("mag", lambda s: round(len(s) / YEARS, 2)),
    median_gap_days=("gap_days", "median"),
    mean_gap_days=("gap_days", "mean"),
    max_mag=("mag", "max"),
).reset_index()
reg["p_ge1_next_year"] = (1 - np.exp(-reg["lambda_per_year"])).round(3)
reg = reg.sort_values("lambda_per_year", ascending=False)

print(f"USGS M>=5.0: {len(usgs)} event | {int(usgs.Year.min())}-{int(usgs.Year.max())} ({YEARS} th)")
print(f"median gap global (M>=5.0): {usgs.gap_days.median():.1f} hari")
print(f"mean gap global            : {usgs.gap_days.mean():.1f} hari")
print("\n== LAJU SEISMISITAS PER WILAYAH (USGS M>=5.0) ==")
print(reg[["Region", "n", "lambda_per_year", "median_gap_days", "mean_gap_days", "p_ge1_next_year", "max_mag"]]
      .to_string(index=False))

reg.to_csv(os.path.join(ART, "usgs_occurrence_risk.csv"), index=False)
print("\nsaved: artifacts/usgs_occurrence_risk.csv")
