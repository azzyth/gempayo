# Task Plan: Prototype ML Gempa — Risiko Kejadian & Severity Kerusakan

## Goal

Membangun prototype untuk Hackathon JOINTS UGM 2026 (tema Sustainability) yang
**memprediksi risiko/frekuensi kejadian gempa** (reframe jujur dari "kapan") dan
**severity kerusakan** (reframe dari "seberapa besar"), dari katalog gempa BMKG 2000–2023.

## Next Step

Packaging & presentasi: satukan (1) laju seismisitas USGS ("kapan") dan (2) severity + exposure
("seberapa besar") jadi satu dashboard + narasi interpretabel. Exposure sudah dicoba: jadi
alternatif interpretabel yang setara (macro-F1 0.552, "Sedang" naik ke 0.222), bukan lompatan
akurasi besar.

## Current Phase

Phase 4 (complete)

## Phases

### Phase 1: Setup, Audit & Cleaning (TAHAP 0–1)

- [x] Task & metrik: klasifikasi tsunami (awal) → pivot ke occurrence + severity
- [x] Audit + cleaning + fitur turunan (`eda/01..03`, `gempabumi_clean.csv`, 307 baris)
- [x] Fix bug parser korban (bullet "- 54 orang" gagal terparse)
- **Status:** complete

### Phase 2: EDA Class-First (TAHAP 2)

- [x] WoE/IV, rule fisik, region crosstab (`eda/04_eda.py`)
- [x] Keputusan: M & MMI_max diskriminator utama; is_offshore_nama gagal
- **Status:** complete

### Phase 3: Validation & OOF (TAHAP 3)

- [x] Fold beku + baseline (`modeling/05_validation.py`, `artifacts/folds.csv`)
- **Status:** complete

### Phase 4: Model Risiko, Regresi "Kapan", & Severity (train/test)

- [x] `modeling/06_occurrence_risk.py` — laju Poisson per wilayah + pola temporal
- [x] `modeling/07_damage_severity.py` — severity 3 kelas (rule + LogisticRegression, OOF)
- [x] `modeling/08_temporal_regression.py` — regresi "kapan" (gap_days, time-based split)
- [x] `modeling/09_severity_train_test.py` — klasifikasi severity (stratified split)
- [x] Severity (test): LightGBM macro-F1 **0.554** (baseline 0.291); "Berat" F1 0.667
- [x] "Kapan": gap ~ Poisson (median 17 h); model fitur kalah dari baseline median 13.1 h
- **Status:** complete

### Phase 5: Data Eksternal & Fine-tune Exposure

- [x] Download USGS (5.722 event) + GeoNames (258.055 permukiman) → CSV
- [x] Fitur exposure (jarak + kepadatan permukiman) → `gempabumi_exposure.csv`
- [x] Fine-tune severity: exposure gantikan lat/lon → macro-F1 0.552, "Sedang" 0.222
- [x] "Kapan" dari USGS: median gap 0.8 hari; laju per wilayah
- **Status:** complete

### Phase 6: Packaging Dashboard & Presentasi Hackathon

- [x] Backend pakai LightGBM + exposure + seismisitas USGS (`backend/app.py` v2)
- [x] Frontend narasi "kapan" + "seberapa besar" + readout exposure (`frontend/`)
- [ ] Video pitch, deck, repo GitHub (deliverable hackathon non-kode)
- **Status:** complete (dashboard) / pending (pitch & deck)

## Key Questions

1. Bisakah severity diprediksi akurat tanpa exposure? → **Tidak; "Berat" terdeteksi 78%,
   tapi "Sedang vs Ringan" lemah. Exposure = lever terbesar.**
2. Apakah "kapan" bisa diprediksi pasti? → **Tidak ilmiah; diganti risiko/frekuensi (Poisson).**
3. Sumber data exposure mana? → **GHSL/WorldPop (grid populasi) atau jarak ke kota terdekat.**

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| "kapan" → risiko/frekuensi (Poisson), bukan prediksi waktu | Waktu pasti gempa belum terpecahkan sains |
| "kerusakan" → severity 3 kelas (Ringan/Sedang/Berat dari korban) | Interpretabel & robust ke outlier Aceh 2004 |
| Fix parser korban sebelum modeling | Under-count 41→70 kejadian merusak label |
| Linear + rule interpretabel dulu, bukan langsung GBDT | Prioritas interpretabilitas prototype |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Parser `n_korban` gagal baca bullet ("- 54 orang") | 2 | Regex + batasi segmen 'meninggal' |
| `class_weight="balanced"` over-correct severity | 1 | Didokumentasikan; kandidat threshold tuning fase 5 |

## Notes

- Status: `pending` → `in_progress` → `complete`.
- Baca ulang **Goal** & **Next Step** sebelum keputusan besar.
- Semua eksperimen model WAJIB memakai fold yang sama (`artifacts/folds.csv`).
