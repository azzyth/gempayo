"""TAHAP 1 -> TAHAP 2: CLEANING + FITUR TURUNAN.

Menerapkan perbaikan terdokumentasi di findings.md (TAHAP 1 - AUDIT DATA):
1. Longitude 12783 -> 127.83 (Ambon 2017)
2. 5 baris latitude hilang tanda negatif -> flip
3. Label "Tidak Ts unami" (typo PDF) -> Tidak Tsunami
4. Exclude 2 baris non-lokal (Jepang 2011, swarm Jailolo 2015)
5. 4 baris tanpa tag -> Tidak Tsunami (asumsi domain, dicatat)
6. Parse Tanggal/Origin_Time + fitur turunan (is_offshore, MMI max, dsb.)
Output: gempabumi_clean.csv + log baris yang diubah/dibuang.
"""
import os
import re
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 250)

df = pd.read_csv(os.path.join(ROOT, "gempabumi_2000-2023.csv"), encoding="utf-8-sig")
log = []

# ---- 1. split Wilayah -> nama + label (toleran typo "Tidak Ts unami") ----
def split_wilayah(s):
    s = str(s).strip()
    compact = re.sub(r"\s+", "", s)  # buang semua spasi utk deteksi suffix
    if compact.endswith("TidakTsunami"):
        base = re.sub(r"\s*Tidak\s*Ts\s*unami$", "", s, flags=re.I).strip()
        return base, "Tidak Tsunami"
    if re.search(r"\s+Tsunami$", s, flags=re.I):
        base = re.sub(r"\s+Tsunami$", "", s, flags=re.I).strip()
        return base, "Tsunami"
    return s, None

df[["Wilayah_nama", "Tsunami"]] = df["Wilayah"].apply(lambda s: pd.Series(split_wilayah(s)))
fixed = int((df.Tsunami == "Tidak Tsunami").sum())
log.append(f"label terparse (setelah toleran typo): {fixed} Tidak Tsunami, "
           f"{(df.Tsunami=='Tsunami').sum()} Tsunami, {df.Tsunami.isna().sum()} tanpa tag")

for c in ["Latitude", "Longitude", "Depth_km", "Magnitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- 2. perbaikan koordinat ----
# Kunci = (Tahun, latitude asal) karena (Tahun, No) tidak unik (2023 #20/#21 duplikat).
lat_fix = [  # (Tahun, lat_salah, lat_benar, catatan)
    (2018, 7.39, -7.39, "Situbondo"),
    (2019, 6.90, -6.90, "Kab. Bogor"),
    (2019, 8.05, -8.05, "Sukabumi"),
    (2023, 7.84, -7.84, "Bima"),
    (2023, 7.26, -7.26, "Sukabumi"),
]
for y, old, new, note in lat_fix:
    m = (df.Tahun == y) & (df.Latitude.round(2) == old)
    assert m.sum() == 1, f"lat fix {note}: {m.sum()} baris cocok"
    df.loc[m, "Latitude"] = new
log.append(f"latitude flip (LS): {len(lat_fix)} baris -> {[f'{y} lat {old}->{new} ({n})' for y, old, new, n in lat_fix]}")

# lon fix: 2017 Ambon 12783 -> 127.83
m = (df.Tahun == 2017) & (df.Longitude == 12783.0)
assert m.sum() == 1, f"lon fix: {m.sum()} baris cocok"
df.loc[m, "Longitude"] = 127.83
log.append("longitude fix: 1 baris (2017 Ambon 12783 -> 127.83)")
log.append("CATATAN: penomoran No. di PDF 2023 duplikat (#20, #21 masing-masing 2x) "
           "-> kolom No tidak reliabel sebagai ID")

# ---- 3. exclude baris non-lokal ----
mask_excl = df.Latitude.isna() | df.Longitude.isna()
excl = df[mask_excl][["Tahun", "No", "Tanggal", "Wilayah"]]
log.append(f"EXCLUDE {mask_excl.sum()} baris non-lokal/koordinat-NaN:\n{excl.to_string(index=False)}")
df = df[~mask_excl].copy()

# ---- 4. label tanpa tag -> Tidak Tsunami (asumsi domain) ----
untagged = df[df.Tsunami.isna()][["Tahun", "No", "Tanggal", "Wilayah_nama"]]
if len(untagged):
    log.append(f"label default 'Tidak Tsunami' utk {len(untagged)} baris tanpa tag "
               f"(asumsi domain, tidak ada tsunami tercatat):\n{untagged.to_string(index=False)}")
    df["Tsunami"] = df["Tsunami"].fillna("Tidak Tsunami")

# ---- 5. parse tanggal & waktu ----
MONTHS = {m.lower(): i for i, m in enumerate(
    ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
     "Agustus", "September", "Oktober", "November", "Desember"], 1)}

def parse_tanggal(s):
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", str(s).strip())
    if not m:
        return pd.NaT
    mon = MONTHS.get(m.group(2).lower())
    if not mon:
        return pd.NaT
    return pd.Timestamp(int(m.group(3)), mon, int(m.group(1)))

df["Date"] = df["Tanggal"].apply(parse_tanggal)

def parse_time(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"^(\d{1,2})[:.](\d{2})[:.](\d{2})", str(s).strip())
    return int(m.group(1)) if m else np.nan

df["Hour"] = df["Origin_Time"].apply(parse_time)

# ---- 6. fitur turunan ----
SEA_KW = re.compile(r"\b(Laut|Samudra|Selat|Teluk|Kep\.?|Kepulauan|Pedalaman|Baratdaya|Sweta)\b", re.I)
df["is_offshore_nama"] = df["Wilayah_nama"].str.contains(SEA_KW).astype(int)

def mmi_max(s):
    if pd.isna(s):
        return np.nan
    romans = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10}
    vals = [romans[r.upper()] for r in re.findall(r"\b(X|IX|VIII|VII|VI|V|IV|III|II|I)\b(?=\s*MMI)", str(s), re.I)]
    return max(vals) if vals else np.nan

df["MMI_max"] = df["Wilayah_yang_merasakan"].apply(mmi_max)

def n_korban(s):
    """Estimasi korban meninggal: angka setelah 'meninggal', berhenti di kata kunci
    bagian lain ('luka', 'kerusakan', dst.) agar tidak salah ambil angka luka/kerusakan."""
    if pd.isna(s):
        return np.nan
    t = str(s).replace("\n", " ")
    m = re.search(r"[Mm]eninggal", t)
    if not m:
        return np.nan
    seg = t[m.end():m.end() + 60]
    seg = re.split(
        r"luka|kerusak|kehilangan|mengungsi|rumah|gedung|bangunan|bandara|pelabuhan|jembatan|jalan",
        seg, flags=re.I,
    )[0]
    num = re.search(r"(\d[\d\.]*)", seg)
    if not num:
        return 0
    return int(num.group(1).replace(".", ""))

df["Korban_meninggal_est"] = df["Korban_Kerusakan"].apply(n_korban)

# ---- 7. validasi akhir (GATE) ----
y = (df["Tsunami"] == "Tsunami").astype(int)
print("=" * 70)
for line in log:
    print(line)
print("=" * 70)
print(f"rows final: {len(df)} | tsunami: {y.sum()} ({y.mean():.1%}) | tidak: {(1-y).sum()}")
print(f"lat  [{df.Latitude.min():.2f}, {df.Latitude.max():.2f}] | long [{df.Longitude.min():.2f}, {df.Longitude.max():.2f}]")
print(f"M    [{df.Magnitude.min()}, {df.Magnitude.max()}] | depth [{df.Depth_km.min()}, {df.Depth_km.max()}]")
assert df.Latitude.abs().max() <= 11.5 and df.Longitude.between(90, 145).all()
assert df[["Latitude", "Longitude", "Depth_km", "Magnitude", "Tsunami"]].notna().all().all()
print("Date terparse:", df.Date.notna().sum(), "/", len(df), "| Hour:", df.Hour.notna().sum())
print("MMI_max terparse:", df.MMI_max.notna().sum(), "| is_offshore=1:", int(df.is_offshore_nama.sum()))
print("semua assert lolos")

out = os.path.join(ROOT, "gempabumi_clean.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print("saved:", out)
