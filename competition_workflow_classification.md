# COMPETITION WORKFLOW   CLASSIFICATION (SPESIALIS)
# =================================================================================================
# Runbook operasional pendamping master_ _classification.md. INI BUKAN  PROMPT   ini
# panduan eksekusi per fase dengan GATE, kontrak artefak, dan checkpoint terverifikasi.
# Berlaku untuk klasifikasi tabular/mixed. Untuk gambar -> workflow CNN; untuk teks ->
# workflow NLP.
#
# Inti khusus klasifikasi yang membedakan dari workflow umum:
#   (A) Metrik imbalance-sensitive: macro vs micro F1 menentukan SELURUH strategi
#   (B) Class distribution = EDA pertama; imbalance handling dipilih dengan alasan
#   (C) Probabilitas OOF [N,C] adalah kontrak utama (blend, threshold, class-scaling)
#   (D) Sanity distribusi prediksi test vs prior train WAJIB sebelum submit
#
# Aturan emas: FASE 3 (Validasi & OOF) WAJIB selesai sebelum FASE 4 (FE) dimulai.
# =================================================================================================

## PETA ALUR

```
[0 Setup & Bedah Metrik Klasifikasi]
   |  macro vs micro vs weighted F1 / AUC / logloss -> menentukan strategi imbalance
   v
[1 Load & Audit Label] --------------------> SUBMIT BASELINE (majority) AWAL
   |  distribusi kelas, label konflik, group, ID overlap
   v
[2 EDA Class-First + Adversarial Validation]
   |  distribusi kelas, density per kelas, WoE/IV, missing per kelas, drift
   v
[3 VALIDASI & OOF]  <-- INTI
   |  StratifiedKFold / StratifiedGroupKFold, fold DIBEKUKAN, baseline majority
   v
[4 Preprocessing & FE]
   |  encoding menu, target encoding K-fold, imbalance handling DI TRAIN SAJA
   v
[5 Modeling]
   |  linear baseline wajib, GBDT, class weight per fold, kalibrasi sesuai metrik
   v
[6 Error Analysis]
   |  confusion pair, per-class F1, worst confidence errors, estimasi label noise
   v
[7 Ensembling & Post-processing]
   |  blend probs di OOF, threshold opt, class-scaling (gain gate), kalibrasi
   v
[8 Final Fit Decision]
   |  refit-full vs fold-mean + parity
   v
[9 Submission + Sanity Distribusi] ----------> MULTI-KANDIDAT, pilih via CV
   v
[10 Wrap-up]
```

## KONTRAK ARTEFAK

| Artefak | Fase | Isi wajib | Dipakai oleh |
|---|---|---|---|
| `rules_checklist.md` | 0 | rumus metrik (macro?), arah, format, kuota | semua fase |
| `folds.csv` | 3 | id, label, target, group, fold   DIBEKUKAN | semua eksperimen |
| `oof/<model>_oof.npz` | 5 | keys `ids`, `probs[N,C]`   urutan folds.csv | blend/threshold |
| `oof/<model>_test.npz` | 5 | keys `ids`, `probs[N,C]`   urutan sample_submission | blend |
| `error_analysis/` | 6 | confusion matrix, per-class F1, worst-N list | fase 7-8 |
| `submission_<kandidat>.csv` | 9 | per kandidat + `submission_log.csv` | seleksi final |

---

## FASE 0   SETUP & BEDAH METRIK KLASIFIKASI

**Tujuan**: metrik menentukan strategi imbalance, threshold, dan kalibrasi   kunci keputusan
klasifikasi yang tidak ada di task lain.

### 0.1 Bedah metrik (jawab tertulis)
- **F1 macro / micro / weighted?** Pada data imbalance 3 kelas (72/25/3):
  - micro-F1 = accuracy -> didominasi kelas mayoritas; model yang mengabaikan kelas 3%
    tetap dapat micro tinggi.
  - macro-F1 -> kelas langka bobot sama penuh; SELURUH strategi (class weight, threshold,
    scaling) mengarah ke kelas langka.
  - Selisih macro vs micro bisa 0.2+   salah implementasi = mengejar angka yang salah.
- **AUC** (binary / ovr / ovo)? AUC tidak peduli threshold   berbeda strategi dari F1.
- **Logloss / Brier**? -> probabilitas harus TERKALIBRASI (kalibrasi jadi fase kritis).
- **Per-entity atau global?** Per-user lalu di-average = segmen kecil bobot besar.
- Arah metrik, public/private split, format submission (label string vs index   cek mapping).

### 0.2 Titik nol (dihitung di FASE 3, ditetapkan di sini)
- Majority-class macro-F1 dan prior-acak   dua titik nol wajib.

**GATE 0**: rumus metrik + implikasi strategi (imbalance/calibration/threshold) tertulis.

---

## FASE 1   LOAD & AUDIT LABEL

**Tujuan**: memahami struktur label + submission valid di papan skor.

### 1.1 Audit label (khusus klasifikasi)
- **Distribusi kelas**: count + proporsi per kelas, train DAN test (test tidak punya label  
  cek prior dari aturan/diskusi bila disebut).
- **Label konflik**: baris dengan fitur identik tapi label berbeda = estimasi label noise
  (menentukan ceiling di FASE 6).
- Konsistensi nama kelas: train vs sample_submission (huruf besar/kecil, spasi).
- Multi-label vs multi-class (satu kelas vs banyak label per baris)   menentukan loss
  (softmax vs sigmoid) dan stratifikasi.

### 1.2 Audit struktur umum
- Overlap ID train vs test (assert 0); entity hantu; group (user/pasien/item); time.
- Duplikat baris persis lintas train/test; downcast dtype; nilai unik kolom numerik
  (sentinel 99/-1, kata-angka).

### 1.3 Fast baseline + submit awal
- Majority class -> submit SEGERA. Catat macro-F1 baseline sebagai titik nol.

**GATE 1**: tabel distribusi kelas + label konflik tercatat + baseline ter-submit.

---

## FASE 2   EDA CLASS-FIRST + ADVERSARIAL VALIDATION

### 2.1 Urutan EDA klasifikasi
1. **Distribusi kelas** (WAJIB paling awal)   menentukan metrik valid & strategi resampling.
2. **Distribusi fitur per kelas**: overlapping density / violin (numerik vs kelas)   fitur
   yang memisahkan kelas langka terlihat di sini.
3. **Stacked bar / crosstab heatmap** (kategorikal x kelas)   interaksi 2 kategorikal.
4. **WoE & Information Value** untuk ranking fitur (biner; untuk multiclass per kelas vs rest).
5. **Missingness per kelas**   missing yang berkorelasi dengan kelas = sinyal (missing as
   feature), bukan noise.
6. **Adversarial validation** (target `is_test`): AUC > 0.8 -> drift; pin fitur penyebab.

### 2.2 Jebakan
- Kolom ID/kardinalitas tinggi jangan masuk korelasi/numerik   catat untuk encoding saja.
- Kelas langka di EDA jangan di-drop dari plot   justru itu fokus utama.

**GATE 2**: daftar fitur discriminator per kelas + verdict adversarial + rencana imbalance
handling (dengan alasan) tertulis.

---

## FASE 3   VALIDASI & OOF (INTI   JANGAN DILEWAT)

### 3.1 Pilih skema CV (meniru cara test dipisah)
| Struktur data | Skema |
|---|---|
| Baris independen | `StratifiedKFold` (selalu, bukan KFold polos) |
| Entity berulang (user/pasien/item) | `StratifiedGroupKFold` by entity |
| Near-duplicate (teks/gambar) | `StratifiedGroupKFold` by duplicate-group |
| Multi-label | `IterativeStratification` (sklearn 1.1+) |
| Data kecil (<5rb) | `RepeatedStratifiedKFold` |
| Time-indexed | `TimeSeriesSplit` + stratifikasi manual bila memungkinkan |

### 3.2 Bekukan fold + bangun OOF
- `folds.csv` (id, label, target, group, fold) + sha256. Semua eksperimen pakai fold ini.
- Struktur OOF: `oof_probs[N, C]` + `ids` persis urutan folds.csv; test `probs[N_test, C]`
  urutan sample_submission. Kontrak ini dipakai FASE 7.

### 3.3 Baseline & metrik
- Implementasi metrik PERSIS rumus (macro-F1 manual, divalidasi `== sklearn.f1_score(average="macro")`).
- Baseline majority per fold -> CV mean±std = titik nol. Catat juga prior-acak.

### 3.4 Leakage guard
- Assert: tidak ada group yang menyeberang fold (`df.groupby("group")["fold"].nunique().max() == 1`).
- Assert: rasio kelas per fold konsisten (stratifikasi bekerja).

**GATE 3**: folds.csv + sha256 + CV baseline (mean±std) + fungsi metrik tervalidasi.

---

## FASE 4   PREPROCESSING & FE

**Prinsip**: encoder/scaler/imputer/resampler di-fit HANYA di train-fold.

### 4.1 Menu encoding
| Kardinalitas | Encoding | Catatan |
|---|---|---|
| Rendah (<15) | One-Hot | aman untuk linear & GBDT |
| Berurutan alami | Ordinal | eksplisit mapping-nya |
| Tinggi | Target encoding K-fold | WAJIB per-fold + smoothing; full-train = leakage |
| Tinggi | Frequency / count encoding | murah, bebas leakage (hitung dari train-fold) |
| ID-like | drop (atau group feature) | jangan pernah fitur model |

### 4.2 Interaksi & fitur turunan
- Interaksi kategorikal x numerik (mean target per kategori   per-fold), rasio domain,
  binning + WoE (untuk interpretasi & model linear).

### 4.3 Imbalance handling (pilih SATU, dengan alasan, DI TRAIN SAJA)
| Metode | Kapan | Risiko |
|---|---|---|
| `class_weight` / `scale_pos_weight` | default pertama | over-correction -> precision kelas langka runtuh (kasus nyata: weight ~10x -> precision 0.29). CAP weight ~3.0 |
| Focal loss | kelas langka sulit + neural net | butuh tuning gamma |
| Undersampling mayoritas | data besar | buang informasi |
| Oversampling/SMOTE | tabular & data kecil | WAJIB setelah split, di train-fold saja; jangan pakai bersama class weight tanpa alasan |
- **Aturan**: apa pun pilihannya, hanya diterapkan pada train-fold; distribusi prediksi
  validasi dicek setelahnya (lihat FASE 5.4).

**GATE 4**: tabel fitur + encoding final + pilihan imbalance + alasan + CV sebelum/sesudah.

---

## FASE 5   MODELING

### 5.1 Urutan model
1. **Linear baseline wajib**: LogisticRegression/LinearSVC di atas OHE+scaled   2 menit,
   pembanding semua model berikutnya. Kalau GBDT kalah dari ini -> ada bug/overfit.
2. **GBDT**: LightGBM/XGBoost/CatBoost (CatBoost native categorical).
3. **kNN/NN** untuk diversitas kalau relevan.

### 5.2 Aturan training klasifikasi
- Class weight dihitung dari **train-fold** saja (bukan full train).
- Early stopping by fold metric (metrik lomba, bukan logloss default).
- Simpan OOF + test probs per model (`oof/<model>_oof.npz`)   kontrak FASE 7.

### 5.3 Tuning
- Optuna objective = metrik lomba di fold sama; ruang pencarian wajar; pruner aktif.

### 5.4 Cek distribusi prediksi (WAJIB per model)
- `value_counts` prediksi validasi per fold. 100% satu kelas = collapse -> karantina.
- Kelas langka over-predicted jauh melebihi prior train = class weight over-correction
  -> turunkan/cap weight. Target: distribusi prediksi mendekati prior train.

**GATE 5**: tabel CV (mean±std) semua model + linear baseline tercatat + OOF/test npz
tersimpan + distribusi prediksi dicek per model.

---

## FASE 6   ERROR ANALYSIS

### 6.1 Analisis (urutan)
1. **Confusion matrix + per-class F1**   kelas mana yang F1 terendah; pasangan kelas yang
   paling sering tertukar.
2. **Confusion pair deep-dive**: baca 10-20 baris salah dari pasangan terburuk. Ambigu
   label atau kelemahan fitur?
3. **Worst high-confidence errors**: prediksi salah dengan confidence tertinggi   sering
   label noise atau segmen khusus.
4. **Error per segmen** (group/sumber/panjang/rentang fitur)   segmen terburuk = arah FE.
5. **Estimasi label noise**: dari label konflik (FASE 1.1) + inspeksi   kalau error model
   terkonsentrasi di baris yang "menurut kita model benar", ceiling = label noise; pindah
   ke kalibrasi/ensemble, bukan kapasitas model.

**GATE 6**: pasangan kelas bermasalah + verdict headroom/ceiling + contoh error dianalisis.

---

## FASE 7   ENSEMBLING & POST-PROCESSING

**Prinsip**: semua bobot/threshold/scale dioptimasi di OOF.

### 7.1 Blend probabilitas
- Equal average dulu; weighted (Nelder-Mead/NNLS/hill-climb) kalau ada sinyal.
- **Jangan blend lemah ke kuat**: kasus nyata 0.663 + 0.846 -> equal-mean 0.838 < 0.846.
  Kalau satu model dominan, bandingkan "best single vs blend" dan kirim yang menang.
- Diversitas = representasi (fitur beda, model beda family), bukan 3 varian config mirip.
- Stacking: nested CV wajib; sering kalah dari simple blend   bandingkan jujur.

### 7.2 Threshold optimization (binary / multi-label)
- Threshold per kelas dari OOF (grid search)   jangan asumsi 0.5. Pada imbalance, 0.5
  hampir selalu suboptimal untuk macro-F1.

### 7.3 Class probability scaling (multiclass, macro-F1)
- Pengali probabilitas per kelas (anchor kelas 0 = 1.0, renormalize), grid 0.5x-2x di OOF.
- **Validasi jujur**: pilih scale di fold lain, skor di fold sisa (per-fold honest). Percaya
  hanya kalau honest ~= in-sample DAN semua fold memilih scale serupa.
- **Gain gate**: pakai HANYA kalau gain >= 0.02 (di bawah itu = overfit OOF).
- Kasus nyata: scale [1, 1, 0.5] dipilih 5/5 fold, honest == in-sample, +0.11 macro-F1.

### 7.4 Kalibrasi (isotonic/Platt)
- HANYA kalau metrik butuh probabilitas terkalibrasi (logloss/Brier). Jangan dobel
  koreksi: model yang sudah pakai class weight + scaling tidak butuh kalibrasi lagi.

**GATE 7**: OOF ensemble vs best single (mean±std) + threshold/scale final + keputusan.

---

## FASE 8   FINAL FIT DECISION

- **Fold-mean** default untuk data kecil/model high-variance (efek ensemble fold gratis).
- **Refit-full** kalau model stabil & data cukup; iterasi disamakan ke konvergensi CV.
- Test-time parity: encoder/scaler/weight dari training dipakai persis; tidak ada fit di test.

**GATE 8**: kebijakan + alasan + parity checklist tertulis.

---

## FASE 9   SUBMISSION + SANITY DISTRIBUSI

1. **Assert vs sample_submission**: shape, kolom, URUTAN, NaN, ID, nilai label valid
   (mapping string persis).
2. **Sanity distribusi prediksi vs prior train**   khusus klasifikasi, WAJIB:
   proporsi kelas prediksi harus mendekati prior train (kecuali memang ada alasan drift).
   Prediksi 100% satu kelas = collapse; kelas langka 3x prior = over-correction.
3. Multi-kandidat (best single, blend, blend+scaling) + `submission_log.csv`.
4. Pilih via CV; LB kalibrasi saja.

**GATE 9**: assert lolos + sanity distribusi OK + log lengkap.

---

## FASE 10   WRAP-UP

- Decision log (kenapa X bukan Y + eksperimen gagal dengan angka).
- metrics_summary.csv + write-up; karantina run collapse dengan nama eksplisit.

---

## PERBANDINGAN: STANDARD vs KAGGLE-TIER (KLASIFIKASI)

| Aspek | Standard | Kaggle-tier (workflow ini) |
|---|---|---|
| Metrik | "F1" | macro/micro/micro dibedah + implikasi strategi |
| Split | train_test_split | Stratified/StratifiedGroupKFold dibekukan |
| Baseline | model pertama | majority + prior-acak + submit awal |
| Imbalance | asal SMOTE | satu metode dipilih dengan alasan, cap weight, train-fold saja |
| Encoding | one-hot semua | menu per kardinalitas + target encoding per-fold |
| Evaluasi model | accuracy | metrik lomba + per-class F1 + distribusi prediksi |
| Threshold | 0.5 | dioptimasi di OOF (per kelas) |
| Ensemble | model terbaik | blend OOF + aturan "jangan blend lemah ke kuat" + SERI |
| Submission | save & submit | assert + sanity distribusi prediksi vs prior |

## CHECKLIST PRE-FLIGHT

- [ ] Macro vs micro dikonfirmasi dari rules; rumus + arah tercatat.
- [ ] Distribusi kelas + label konflik diaudit.
- [ ] Skema CV dipilih menurut struktur group/duplikat; folds.csv + sha256.
- [ ] Baseline majority di-submit; macro-F1 titik nol tercatat.

## CHECKLIST PRE-SUBMIT FINAL

- [ ] Semua eksperimen fold sama; CV mean±std tercatat; linear baseline tercatat.
- [ ] Distribusi prediksi validasi dicek (tidak collapse, tidak over-correct).
- [ ] Error analysis selesai; verdict headroom/ceiling.
- [ ] Blend/threshold/scale dioptimasi di OOF dengan validasi honest per fold.
- [ ] Sanity distribusi prediksi test vs prior train OK.
- [ ] Assert vs sample_submission lolos; submission_log lengkap.

## JEBAKAN KHAS KLASIFIKASI

| Jebakan | Gejala | Perbaikan |
|---|---|---|
| Micro-F1 dipakai untuk lomba macro | CV tinggi, LB rendah | Implementasi metrik persis + validasi vs sklearn |
| SMOTE sebelum split | CV optimistis palsu | Resampling train-fold saja |
| Target encoding full-train | CV naik, LB jatuh | K-fold per fold |
| Class weight tak di-cap | Precision kelas langka runtuh (0.29) | Cap ~3.0, cek distribusi prediksi |
| Class weight + oversampling bareng | Over-correction ganda | Pilih satu |
| Threshold 0.5 asumsi | Macro-F1 jauh dari optimal | Threshold opt di OOF |
| Blend lemah ke kuat | Blend < best single | Bandingkan best single vs blend |
| Class-scaling tanpa honest fold | Overfit OOF | Per-fold honest + gain gate 0.02 |
| Kalibrasi + class weight dobel | Probabilitas rusak dua arah | Satu saja, sesuai metrik |
| Label string beda kapital | Submission ditolak/0 skor | Assert mapping vs sample_submission |
| Collapse tak dicek | Submit all-one-class | value_counts tiap fold sebelum submit |
| Group leak (user/pasien) | CV 0.95, LB 0.55 | StratifiedGroupKFold |


---

## MODUL PENINGKATAN MODEL   HUGGING FACE + GITHUB (OPSIONAL)

> Jalankan SETELAH FASE 3 (fold beku). Klasifikasi tabular: tabular foundation + AutoML.

### Strategi pencarian & seleksi (filter berurutan)
1. **Task type**   HF: `pipeline_tag=tabular-classification`; GitHub: `"tabular classification kaggle solution"`.
2. **License (WAJIB)**   apache-2.0/mit/bsd aman; cc-by-nc & gpl berisiko; sinkronkan aturan lomba.
3. **Leaderboard performance**   HF sort `downloads`+`likes`; GitHub ⭐ tinggi + aktif + weights.
4. **Verifikasi murah**   TabPFN/AutoGluon zero-shot di fold beku DULU; kalah dari LGBM =
   kandidat blend, jangan paksa.
5. **Compliance**   catat source + lisensi di decision log.

### Shortlist klasifikasi
| Kandidat | Kapan menang | Cost |
|---|---|---|
| `Prior-Labs/TabPFN-v2-classifier` (HF) | data kecil (<10k), zero-shot, multi-class | rendah |
| AutoGluon (pip) | baseline otomatis + ensemble | sedang |
| LightGBM/XGBoost/CatBoost (GitHub) | data besar, class_weight, Optuna | rendah |
| FT-Transformer (GitHub) | data besar + diversitas NN | tinggi |
| sentence-transformers + GBDT | klasifikasi dari teks | rendah |

**GATE**: kandidat + lisensi + hasil zero-shot probe (CV mean±std) tercatat.


---

## BEST PRACTICES REVERSE-ENGINEERED   KAGGLE WINNERS (KLASIFIKASI)

> Gain tipikal antar-kompetisi, BUKAN jaminan; validasi fold beku; selisih < std = SERI.

| Teknik | Kapan dipakai | Expected gain | Cost |
|---|---|---|---|
| Adversarial validation | selalu, awal | preventif (cegah gap CV↔LB) | rendah |
| Stratified/Group K-fold beku | selalu | krusial | ~0 |
| OOF probs [N,C] | selalu | basis blend/threshold/scaling | N× |
| Class weight ter-cap (~3.0) | imbalance | +1–5% macro-F1 | ~0 |
| Focal loss | kelas langka sulit (NN) | +1–5% macro-F1 | ~0 |
| Label smoothing | klasifikasi | +0.2–0.5% | ~0 |
| Threshold tuning per kelas (OOF) | metrik F1 | +0.5–3% | rendah |
| Class probability scaling (honest fold) | multiclass macro-F1 | +1–5% (kasus: +0.11) | rendah |
| Target encoding K-fold | kategorikal high-card | +0.5–2% | rendah |
| Calibration (isotonic/Platt) | logloss/Brier | nyata | rendah |
| Optuna + early stopping | selalu | +0.5–2% | sedang |
| Multi-seed / seed averaging | NN/GBDT | +0.1–0.5% | N× |
| Blending probs (OOF) | model beragam | +0.3–1.5% | rendah |
| Stacking (nested CV) | model beragam | +0.5–2% | sedang |
| Pseudo-labeling | train kecil | +1–5% | sedang |
| External data / transfer | aturan izin | +1–10% (variabel) | tinggi |


---

## ORIENTASI METRIK & CHECKPOINTS OPTIMASI (KLASIFIKASI)

**Ladder prioritas (gain per cost menurun):**
1. Bedah macro vs micro (menentukan SEMUA strategi imbalance).
2. Titik nol (majority) + fold beku (stratified/group).
3. Distribusi prediksi = detektor bug termurah (collapse / over-correction).
4. Cheap wins: class weight ter-cap, threshold OOF, target encoding, class-scaling honest.
5. Diversitas model + blend probs OOF. 6. Pseudo-label / stacking (setelah 1–5 mentok).

**📊 METRIC CHECKPOINT tiap fase (GATE):**
- [ ] FASE 3: CV baseline macro-F1 (mean±std) tercatat.
- [ ] FASE 4: tiap fitur/encoding Δ metrik; gain ≥ std fold.
- [ ] FASE 5: tiap model mean±std + distribusi prediksi dicek (collapse guard).
- [ ] FASE 7: class-scaling honest per-fold + gain gate ≥ 0.02.
- [ ] FASE 9: sanity distribusi prediksi vs prior train; final dari CV metrik.
