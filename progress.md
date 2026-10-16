# Progress Log

## Session: 2026-10-15

### Phase 1: Setup, Audit & Cleaning (TAHAP 0–1)

- **Status:** complete
- Actions taken:
  - Tentukan task = klasifikasi tsunami (awal) → pivot ke occurrence risk + severity.
  - Audit + cleaning + fitur turunan (`eda/01..03`).
  - Fix bug parser korban: bullet "- 54 orang" gagal terparse → korban>=1 naik 41 → 70.
- Files created/modified:
  - `eda/01_audit.py`, `eda/02_investigate.py`, `eda/03_clean.py` (parser fix)
  - `gempabumi_clean.csv`

### Phase 2: EDA Class-First (TAHAP 2)

- **Status:** complete
- Actions taken: `eda/04_eda.py` (WoE/IV, rule fisik, region crosstab).
- Files created/modified: `eda/04_eda.py`, `gempabumi_clean.csv`

### Phase 3: Validation & OOF (TAHAP 3)

- **Status:** complete
- Actions taken: `modeling/05_validation.py` (fold beku + baseline majority 0.477).
- Files created/modified: `modeling/05_validation.py`, `artifacts/folds.csv`

### Phase 4: Model Risiko, Regresi "Kapan", & Severity (train/test)

- **Status:** complete
- Actions taken:
  - `modeling/06_occurrence_risk.py`: laju Poisson per wilayah + tren + musiman.
  - `modeling/07_damage_severity.py`: severity 3 kelas + rule + LogisticRegression (OOF).
  - `modeling/08_temporal_regression.py`: regresi gap_days (time-based train/test).
  - `modeling/09_severity_train_test.py`: klasifikasi severity (stratified train/test).
- Files created/modified:
  - `modeling/06..09`, `artifacts/occurrence_risk.csv`, `artifacts/severity_oof_logreg.npy`
  - `gempabumi_model.csv`

### Phase 5: Data Eksternal & Fine-tune Exposure

- **Status:** complete
- Actions taken:
  - Download USGS (5.722 event M≥5.0) + GeoNames (258.055 permukiman) via PowerShell.
  - Fitur exposure (jarak + kepadatan permukiman) → `gempabumi_exposure.csv`.
  - Fine-tune severity: exposure gantikan lat/lon → macro-F1 0.552, "Sedang" 0.222.
  - "Kapan" dari USGS: median gap 0.8 hari; laju per wilayah (`artifacts/usgs_occurrence_risk.csv`).
- Files created/modified:
  - `data/*`, `eda/05_exposure_features.py`, `modeling/10_severity_exposure.py`, `modeling/11_when_usgs.py`
  - `gempabumi_exposure.csv`, `artifacts/usgs_occurrence_risk.csv`

### Phase 6: Packaging Dashboard

- **Status:** complete
- Actions taken:
  - Backend `app.py` v2: LightGBM + exposure (BallTree GeoNames) + endpoint `/api/seismicity` (USGS).
  - Frontend: narasi "kapan" + "seberapa besar", lat/lon opsional, readout exposure.
  - Verifikasi end-to-end: summary 0.552, seismicity 5.722 event (gap 0.82 h), predict M7.4 Palu → "Berat" 99.9%.
- Files created/modified:
  - `backend/app.py`, `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Parser korban (bullet) | "- sedikitnya 54 orang" | 54 | 54 | PASS |
| Korban>=1 setelah fix | clean CSV | ~70 | 70 | PASS |
| Severity macro-F1 (logreg) | fold beku | > baseline 0.29 | 0.440 | PASS |
| Recall kelas "Berat" | OOF | tinggi | 0.78 (18/23) | PASS |
| Laju Poisson Jawa-selatan | 24 th, 88 gempa | ~3.67/th | 3.67 | PASS |
| Severity LightGBM (test) | stratified 80/20 | > baseline | macro-F1 0.554 | PASS |
| F1 "Berat" (test, LightGBM) | 5 event | tinggi | 0.667 (3/5) | PASS |
| Regresi gap_days (test) | time-based split | jujur | baseline median 13.1 h menang | PASS |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-10-15 | Parser `n_korban` gagal baca bullet ("- 54 orang") | 2 | Regex + batasi segmen 'meninggal' |
| 2026-10-15 | `class_weight="balanced"` over-correct severity | 1 | Didokumentasikan; threshold tuning fase 5 |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 6 (dashboard) complete, pitch/deck pending |
| Where am I going? | Video pitch, deck, repo GitHub |
| What's the goal? | Prototype "kapan" + severity (dashboard + narasi) |
| What have I learned? | Dashboard end-to-end: LightGBM+exposure, USGS seismisitas |
| What have I done? | Packaging dashboard + backend v2 + frontend narasi |

---

*Perbarui file ini setiap selesai fase, menjalankan validasi, atau menemui error.*
