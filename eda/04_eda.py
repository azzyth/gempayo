"""TAHAP 2 - EDA CLASS-FIRST (master_prompt_classification.md Tahap 2).

Checkpoint: fitur discriminator per kelas + rencana imbalance (dengan alasan).
Adaptasi: tidak ada test set terpisah -> adversarial validation TIDAK relevan
(satu populasi); validasi via CV (Tahap 3).
"""
import os
import re
import pandas as pd
import numpy as np

pd.set_option("display.width", 220)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "gempabumi_clean.csv"), encoding="utf-8-sig")
y = (df["Tsunami"] == "Tsunami").astype(int)

# ---- 2.1 density numerik per kelas ----
print("== DENSITY PER KELAS (kuantil numerik) ==")
for c in ["Magnitude", "Depth_km", "Latitude", "Longitude"]:
    g = df.groupby(y)[c].quantile([0, .25, .5, .75, 1]).unstack().round(2)
    g.index = ["tidak", "tsunami"]
    print(f"\n{c}:\n{g}")
# log-depth karena skewed
df["log_depth"] = np.log1p(df.Depth_km)

# overlap M: berapa % non-tsunami dengan M >= 6.1 (batas bawah tsunami)?
m_thr = df.loc[y == 1, "Magnitude"].min()
n_above = ((df.Magnitude >= m_thr) & (y == 0)).sum()
print(f"\nbatas M tsunami terendah = {m_thr}; non-tsunami dengan M >= {m_thr}: {n_above} baris "
      f"({n_above / (1 - y).sum():.1%} dari kelas 0) -> separability parsial")

# ---- 2.2 WoE & Information Value ----
def woe_iv(bins, y, name):
    d = pd.DataFrame({"b": bins, "y": y})
    g = d.groupby("b")["y"].agg(["sum", "count"])
    g["good"] = g["count"] - g["sum"]
    g["dist_bad"] = g["sum"] / g["sum"].sum()
    g["dist_good"] = g["good"] / g["good"].sum()
    g["woe"] = np.log((g.dist_bad + 1e-6) / (g.dist_good + 1e-6))
    g["iv"] = (g.dist_bad - g.dist_good) * g.woe
    print(f"\n-- {name} --\n{g[['sum', 'count', 'woe', 'iv']].round(3)}")
    return g["iv"].sum()

print("\n== WoE & IV (bad = tsunami) ==")
ivs = {}
ivs["Magnitude_bin"] = woe_iv(pd.qcut(df.Magnitude, 5, duplicates="drop"), y, "Magnitude (qcut)")
ivs["log_depth_bin"] = woe_iv(pd.qcut(df.log_depth, 5, duplicates="drop"), y, "log_depth (qcut)")
ivs["is_offshore"] = woe_iv(df.is_offshore_nama, y, "is_offshore_nama")
ivs["lat_bin"] = woe_iv(pd.qcut(df.Latitude, 5, duplicates="drop"), y, "Latitude (qcut)")
ivs["lon_bin"] = woe_iv(pd.qcut(df.Longitude, 5, duplicates="drop"), y, "Longitude (qcut)")
if df.MMI_max.notna().sum():
    mmi = df.MMI_max.fillna(0)
    ivs["MMI_bin"] = woe_iv(pd.cut(mmi, [0, 2, 4, 5, 6, 10]), y, "MMI_max (cut)")
print("\nRINGKASAN IV:", {k: round(v, 3) for k, v in sorted(ivs.items(), key=lambda x: -x[1])})
print("IV: <0.02 tak berguna | 0.02-0.1 lemah | 0.1-0.3 sedang | >0.3 kuat")

# ---- 2.3 rule fisik sederhana (interpretabilitas utk prototype) ----
print("\n== RULE FISIK SEDERHANA: M>=6.5 & depth<=70 ==")
rule = ((df.Magnitude >= 6.5) & (df.Depth_km <= 70)).astype(int)
tp = (rule & y).sum(); fp = (rule & ~y.astype(bool)).sum()
fn = (~rule.astype(bool) & y).sum(); tn = ((1 - rule) & (1 - y)).sum()
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f"TP={tp} FP={fp} FN={fn} TN={tn} | precision={prec:.2f} recall={rec:.2f} f1={f1:.2f}")
print("FN (tsunami yang lolos rule):")
print(df[(y == 1) & (rule == 0)][["Tahun", "Wilayah_nama", "Magnitude", "Depth_km"]].to_string(index=False))

# grid search threshold rule M>=t1 & depth<=t2 untuk f1 positif
print("\n== GRID RULE M>=t1 & depth<=t2 (f1 positif) ==")
best = (0, 0, 0)
for t1 in np.arange(5.0, 7.51, 0.1):
    for t2 in [30, 40, 50, 60, 70, 100]:
        r = ((df.Magnitude >= t1) & (df.Depth_km <= t2)).astype(int)
        tp2 = (r & y).sum(); fp2 = (r & (1 - y)).sum(); fn2 = ((1 - r) & y).sum()
        p = tp2 / max(tp2 + fp2, 1); rr = tp2 / max(tp2 + fn2, 1)
        f = 2 * p * rr / max(p + rr, 1e-9)
        if f > best[2]:
            best = (round(t1, 1), t2, round(f, 3))
print(f"best rule: M>={best[0]} & depth<={best[1]} -> f1_positif={best[2]} (in-sample, hanya gambaran)")

# ---- 2.4 crosstab kategorikal ----
print("\n== OFFSHORE x KELAS ==")
print(pd.crosstab(df.is_offshore_nama, y, normalize="index").round(3))

def region(row):
    if row.Longitude < 105 and row.Latitude < -2: return "Sumatra/NSB-barat"
    if row.Longitude < 106: return "Sumatra utara"
    if row.Latitude < -9 and row.Longitude < 130: return "NTT/Timor-selatan"
    if row.Latitude < -4 and 105 <= row.Longitude < 120: return "Jawa-selatan"
    if 120 <= row.Longitude < 130 and row.Latitude < -3: return "Nusa-selatan"
    if row.Longitude >= 128 and row.Latitude < -2: return "Maluku/Banda"
    if row.Latitude >= -2: return "Sulawesi-utara/Papua-barat"
    return "lain"

df["Region"] = df.apply(region, axis=1)
print("\n== REGION x KELAS ==")
ct = pd.crosstab(df.Region, y)
ct["p_tsunami"] = (ct[1] / ct.sum(axis=1)).round(3)
print(ct.sort_values("p_tsunami", ascending=False).to_string())

# ---- 2.5 MMI & korban per kelas ----
print("\n== MMI_max & korban per kelas ==")
print(df.groupby(y)[["MMI_max", "Korban_meninggal_est"]].agg(["median", "max"]).round(1))

# ---- 2.6 baca contoh FN potensial ----
print("\n== 5 TSUNAMI TERKECIL (M) ==")
print(df[y == 1].nsmallest(5, "Magnitude")[["Tahun", "Wilayah_nama", "Magnitude", "Depth_km"]].to_string(index=False))

df.to_csv(os.path.join(ROOT, "gempabumi_clean.csv"), index=False, encoding="utf-8-sig")
print("\nsaved fitur EDA (log_depth, Region) ke gempabumi_clean.csv")
