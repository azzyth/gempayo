# COMPETITION WORKFLOW    REGRESSION (SPESIALIS)
# =================================================================================================
# Runbook operasional pendamping master_  _regression.md. INI BUKAN  PROMPT     ini panduan
# eksekusi per fase: setiap fase punya GATE yang wajib lolos sebelum fase berikutnya, kontrak
# artefak eksplisit, dan checkpoint yang diverifikasi dengan angka (bukan "sepertinya ok").
#
# Inti khusus regresi yang membedakan dari workflow umum:
#   (A) Metrik -> Loss -> Transform HARUS satu rantai keputusan yang selaras
#   (B) Stratifikasi fold berbasis BIN TARGET (qcut), bukan split acak
#   (C) Residual analysis adalah error analysis-nya regresi (fan shape, per-decile, outlier)
#   (D) Round-trip inverse transform (expm1) diuji sebelum submission
#
# Aturan emas: FASE 3 (Validasi & OOF) WAJIB selesai sebelum FASE 4 (FE) dimulai.
# =================================================================================================

## PETA ALUR

```
[0 Setup & Bedah Metrik]
   |  RMSE vs MAE vs MAPE vs RMSLE vs QWK -> tentukan loss training + transform target
   v
[1 Load & Audit Target] -------------------> SUBMIT BASELINE (mean/median) AWAL
   |  audit target: range, zeros, negative, skew, duplikat konflik
   v
[2 EDA Target-First + Adversarial Validation]
   |  distribusi target, korelasi, heteroskedastisitas, VIF, drift train-vs-test
   v
[3 VALIDASI & OOF]  <-- INTI
   |  StratifiedKFold di atas bin target (qcut), fold DIBEKUKAN, baseline mean+median
   v
[4 Preprocessing & FE]
   |  transform target (log1p/Yeo-Johnson), rasio & interaksi, aggregasi per-fold
   v
[5 Modeling & Loss]
   |  Ridge selalu ada, GBDT dengan objective SELARUS metrik, Optuna, simpan OOF/test
   v
[6 Residual Analysis & Ceiling Diagnosis]
   |  residual vs prediksi, error per decile target, inspeksi worst-N
   v
[7 Ensembling & Post-processing]
   |  bobot dari OOF (Nelder-Mead/NNLS/hill-climb), clipping ke rentang train
   v
[8 Final Fit Decision]
   |  refit-full vs fold-mean, dipilih dengan alasan + test-time parity
   v
[9 Submission & Round-trip Test] ----------> MULTI-KANDIDAT, pilih via CV
   |  assert format + uji expm1 round-trip + sanity distribusi
   v
[10 Wrap-up & Decision Log]
```

## KONTRAK ARTEFAK (file wajib ada di tiap fase)

| Artefak | Fase | Isi wajib | Dipakai oleh |
|---|---|---|---|
| `rules_checklist.md` | 0 | aturan, rumus metrik, arah, format submission, kuota | semua fase |
| `metrics_summary.csv` | 1+ | eksperimen, fitur, model, CV mean, CV std, catatan "kenapa" | fase 7-10 |
| `folds.csv` | 3 | row_id, target, bin, fold    DIBEKUKAN + sha256 dicatat | semua eksperimen |
| `oof/<nama_model>_oof.npy` | 5 | prediksi OOF per model (urutan = urutan train) | fase 7 blend |
| `oof/<nama_model>_test.npy` | 5 | prediksi test per model (urutan = sample_submission) | fase 7 blend |
| `residual_analysis/` | 6 | plot residual, tabel per-decile, worst-N list | fase 7-8 keputusan |
| `submission_<kandidat>.csv` | 9 | satu file per kandidat + `submission_log.csv` | seleksi final |

---

## FASE 0    SETUP & BEDAH METRIK

**Tujuan**: mengunci rantai keputusan Metrik -> Loss -> Transform SEBELUM melihat data.
Ini keputusan paling berbeda di regresi: metrik menentukan loss training DAN transformasi target.

### 0.1 Tabel keputusan metrik (isi dari rules, jangan tebak)

| Metrik | Loss training yang selaras | Transform target | Catatan kritis |
|---|---|---|---|
| RMSE | MSE (L2) | tanpa transform (atau sesuai skew) | outlier didenda kuadrat -> model mengejar outlier |
| MAE | L1 (`objective="mae"` / MAE loss) | tanpa transform | robust ke outlier; kalau GBDT tak stabil pakai pseudo-Huber |
| RMSLE | MSE di atas `log1p(target)` | `log1p`, prediksi `expm1` | penalize underestimation; wajib round-trip test (FASE 9) |
| MAPE | L1 di `log(target)` atau weighted | log (hanya target>0) | hancur kalau target dekat 0 -> cek distribusi target dulu |
| QWK (ordinal) | MSE/CE + rounding optimization |    | kandidat: soft labels + threshold rounding per level |

### 0.2 Pertanyaan wajib dijawab tertulis
- Arah metrik (lower better untuk semua error metric    konfirmasi, jangan asumsi).
- Public/private split berapa persen? Per-entity atau global?
- Format submission: kolom, urutan, jumlah desimal (rounding), rentang nilai valid.
- External data / code competition / kuota submit per hari.

**GATE 0**: tabel 0.1 terisi dari rules + format submission tercatat + rencana kuota submit.
**Kenapa**: salah memilih loss (MSE untuk lomba MAE) membuat seluruh tuning berubah arah   
model akan fokus ke outlier padahal metrik tidak menghukumnya.

---

## FASE 1    LOAD & AUDIT TARGET

**Tujuan**: memahami struktur target + punya submission valid di papan skor.

### 1.1 Audit target (khusus regresi    ini "class distribution"-nya)
- `min, max, mean, median, std, skew, kurtosis` + histogram.
- **Jumlah target == 0** dan **target negatif**    menentukan transform valid (log1p vs Yeo-Johnson).
- **Duplikat baris fitur dengan target BERBEDA**    estimasi noise floor: kalau fitur identik
  punya target beda jauh, itu batas bawah error yang mungkin (irreducible noise).
- Target konstan/ quasi-konstan di segmen tertentu (zero-inflation).

### 1.2 Audit struktur umum
- Downcast dtype (float64->float32) segera; polars untuk data besar.
- Overlap ID train vs test; entity hantu; group (user/pasien/item berulang); time leakage.
- Nilai unik kolom "numerik"    cari kata-angka ("Nol","Dua") dan sentinel invalid (99, -1, -3).

### 1.3 Fast baseline + submit awal
- Baseline: mean target (untuk RMSE) DAN median target (untuk MAE)    keduanya dihitung.
- Submit SEGERA. Titik nol LB tercatat di `submission_log.csv`.

**GATE 1**: tabel audit + baseline ter-submit + noise floor (duplikat konflik) tercatat.

---

## FASE 2    EDA TARGET-FIRST + ADVERSARIAL VALIDATION

**Tujuan**: setiap plot berakhir jadi SATU keputusan FE/modeling.

### 2.1 Urutan EDA regresi (target dulu, baru fitur)
1. **Distribusi target**: histogram + skewness + kurtosis. Skew > 2 -> kandidat transform.
2. **Scatter fitur vs target + LOESS/trend line**    cari non-linearitas (mandat fitur
   interaksi/non-linear model) dan heteroskedastisitas (lebar residu naik seiring prediksi ->
   kandidat variance-stabilizing transform).
3. **Box/violin target per kategorikal**    kategori mana yang memisahkan target.
4. **Correlation heatmap + VIF**    multikolinearitas tinggi (VIF>10) tidak fatal untuk GBDT
   tapi fatal untuk interpretasi & model linear.
5. **Adversarial validation** (train vs test, target `is_test`): AUC ~0.5 aman; AUC > 0.8 =
   drift -> identifikasi fitur penyebab, pertimbangkan sample weight / drop fitur.
6. Residual & Q-Q plot SETELAH model pertama (di FASE 6)    bukan sekarang.

### 2.2 Jebakan data kotor
- Nilai unik kolom numerik (bukan hanya describe())    sentinel & kata-angka.
- Korelasi lemah tapi ARAH konsisten = sinyal interaksi, bukan fitur tak berguna.

**GATE 2**: keputusan tertulis    (a) transform target kandidat, (b) daftar fitur non-linear/
interaksi kandidat, (c) verdict adversarial (drift/tidak + fitur penyebab).

---

## FASE 3    VALIDASI & OOF (INTI    JANGAN DILEWAT)

**Tujuan**: satu set fold yang meniru cara test dinilai, DIBEKUKAN, dipakai semua eksperimen.

### 3.1 Stratifikasi regresi (berbeda dari klasifikasi)
Regresi tidak punya kelas    stratifikasi dibuat dari **bin target**:
```python
df["bin"] = pd.qcut(df["target"].rank(method="first"), q=10, labels=False)  # skew-proof
# lalu StratifiedKFold(n_splits=5, shuffle=True, random_state=42) di atas df["bin"]
```
- `rank(method="first")` menghindari error duplicate edges pada target diskrit/skewed.
- Skew ekstrem: stratifikasi di atas `pd.qcut(np.log1p(target), 10)`.
- Group/time structure tetap menang atas stratifikasi: GroupKFold by entity,
  TimeSeriesSplit untuk data kronologis (tanpa shuffle).
- Data kecil (<5rb): `RepeatedKFold` / `RepeatedStratifiedKFold` di bin.

### 3.2 Bekukan fold
- Simpan `folds.csv` (row_id, target, bin, fold) + catat sha256.
- Semua eksperimen berikutnya WAJIB fold sama. Ganti fold = semua angka lama tidak comparable.

### 3.3 Baseline per fold + metrik
- Implementasi metrik PERSIS dari rumus FASE 0; validasi vs sklearn
  (`mean_squared_error(...)**0.5` dll) pada contoh kecil.
- Jalankan baseline mean DAN median per fold -> CV mean±std = titik nol.
- Catat selisih mean vs median: kalau median jauh lebih baik di MAE, distribusi target
  berat ekor -> strategi robust (median-regression / clipping) naik prioritas.

**GATE 3**: `folds.csv` + sha256 tercatat + CV baseline (mean±std) untuk mean & median +
fungsi metrik tervalidasi vs sklearn.

---

## FASE 4    PREPROCESSING & FEATURE ENGINEERING

**Prinsip**: semua yang belajar dari data (scaler, encoder, target encoding, imputer,
binning, selector) di-fit HANYA di train-fold, di-apply ke valid/test.

### 4.1 Keputusan transform target (pohon keputusan)
```
target positif semua & skew tinggi (>2)      -> log1p (ingat expm1 di inference)
ada nol/negatif & skew                       -> Yeo-Johnson (PowerTransformer, fit per-fold)
ekor sangat berat / outlier ekstrem          -> QuantileTransformer (normal)    fit per-fold
metrik RMSLE                                 -> log1p WAJIB (bagian definisi metrik)
metrik MAE & median menang                   -> pertahankan skala asli, loss L1
```
- Laporkan skew sebelum/sesudah transform. Transform yang tidak mereduksi skew = jangan dipakai.
- **Hati-hati**: transform target untuk MENANGANI metrik (RMSLE) vs untuk STABILITAS training
  (centering) adalah dua keputusan berbeda    keduanya bisa aktif bersamaan.

### 4.2 Menu FE regresi
- **Rasio domain** (harga per m2, jumlah per kapasitas)    sering lebih informatif dari
  fitur mentah untuk tree sekalipun (tree sulit mengekspresikan rasio).
- **Interaksi** (fitur x kategorikal -> slope berbeda per grup) kalau EDA menunjukkan tren
  berbeda arah antar grup.
- **Aggregasi entitas** (mean/max/count per group key)    WAJIB dihitung dari train-fold saja.
- **Target encoding K-fold** untuk kategorikal kardinalitas tinggi (smoothing + noise);
  per-fold, bukan full-train.
- **Frekuensi encoding**    murah, aman leakage.
- **Fitur waktu** (hari minggu, bulan, trend) untuk data bertimestamp.

### 4.3 Feature selection berbasis CV
- Permutation importance di OOF; buang fitur yang menurunkan CV saat disensor.
- Jangan buang fitur hanya karena korelasi rendah    tree memakai interaksi.

**GATE 4**: tabel fitur final + assert NaN-free train & test + tabel CV "sebelum vs sesudah"
tiap batch fitur (mean±std, selisih < std = SERI).

---

## FASE 5    MODELING & LOSS OPTIMIZATION

**Tujuan**: beberapa model KUAT dan BERAGAM (diversitas informasi), fold sama.

### 5.1 Urutan model (jangan lompat)
1. **Ridge / LinearRegression di atas fitur + transform**    baseline linear wajib;
   murah, jadi pembanding apakah non-linearitas benar-benar membayar.
2. **LightGBM / XGBoost / CatBoost**    default kuat tabular.
   - objective SELARUS metrik (lihat FASE 0.1): MAE -> `objective="mae"` /
     pseudo-Huber; RMSLE -> reg:squarederror di log1p(target).
   - CatBoost unggul di kategorikal native; LGBM tercepat untuk tuning.
3. **kNN / SVM** untuk diversitas (hanya kalau skala fitur wajar).
4. **NN (MLP)** HANYA kalau data besar (>100rb) atau ada modalitas non-tabular.

### 5.2 Tuning
- Optuna (TPE) dengan pruner; objective = metrik lomba di fold yang sama.
- Regularisasi & early stopping > tuning buta. Label noisy: kapasitas berlebih = hafal noise.
- Ruang pencarian masuk akal (lr 0.01-0.1, depth 4-10, leaves 31-255, dll).

### 5.3 Protokol eksperimen
- Satu eksperimen = satu hipotesis. CV mean±std dicatat di metrics_summary.csv.
- Simpan `oof/<model>_oof.npy` + `oof/<model>_test.npy` PER MODEL (urutan row train /
  sample_submission). Ini kontrak untuk FASE 7.

**GATE 5**: tabel CV semua model (mean±std) + OOF/test tersimpan + Ridge sebagai pembanding.

---

## FASE 6    RESIDUAL ANALYSIS & CEILING DIAGNOSIS

**Tujuan**: tahu DI MANA dan MENGAPA model salah sebelum mengejar model baru.

### 6.1 Analisis residual (urutan wajib)
1. **Residual vs predicted scatter**: pola fan (lebar residual naik dengan prediksi) =
   heteroskedastisitas -> kandidat transform target / two-stage model. Kurva U = bias
   sistematis -> fitur kurang / interaksi kurang.
2. **Error per decile target**: MAE per decile. Kalau decile ujung jauh lebih buruk
   (dalam %), model tidak menangkap ekor -> clipping, weighting, atau transform.
3. **Worst-N inspection**: print N=20 baris dengan residual terbesar. BACA SATU-SATU.
   Sering kali: (a) label salah/outlier nyata, (b) segmen khusus yang butuh fitur sendiri.
4. **Residual per segmen kategorikal / group**    segmen dengan error jauh lebih buruk =
   arah FE berikutnya.
5. **Residual vs fitur penting**    pola sisa = sinyal fitur yang belum tertangkap.

### 6.2 Diagnosis ceiling
- Noise floor dari duplikat konflik (FASE 1.1): kalau error model sudah mendekati noise floor,
  berhenti mengejar fitur    pindah ke ensemble & kalibrasi.
- Kalau error tersisa terkonsentrasi di outlier yang tidak eksplisit di data, itu irreducible.

**GATE 6**: verdict tertulis    "masih headroom (segmen X, fitur Y)" ATAU "sudah ceiling
(alasan)". Plot residual + tabel per-decile tersimpan.

---

## FASE 7    ENSEMBLING & POST-PROCESSING

**Prinsip**: bobot dioptimasi di OOF (leak-free), bukan prediksi train (overfit), bukan test.

### 7.1 Blend
- Equal average dulu, baru weighted: Nelder-Mead / NNLS / hill climbing di OOF.
- **Aturan SERI**: gain blend < std fold -> pakai model tunggal terbaik (yang lebih sederhana).
- Rank averaging kalau metrik rank-based; logit/probability averaging untuk klasifikasi saja.
- **Diversitas = informasi**, bukan nama algoritma: Ridge + LGBM + kNN > 3 varian LGBT
  config mirip.
- Stacking: meta-model di atas OOF, WAJIB nested CV; bandingkan vs simple blend    sering kalah.

### 7.2 Post-processing regresi
- **Clipping** prediksi ke [min, max] target train.
- Rounding sesuai format submission (cek jumlah desimal dari sample_submission).
- Kalau target di-transform: prediksi final = `expm1` / inverse_transform    uji round-trip.

**GATE 7**: skor OOF blend vs best single (mean±std) + keputusan dipakai/tidak + bobot final.

---

## FASE 8    FINAL FIT DECISION & TEST-TIME PARITY

### 8.1 Refit-full vs fold-mean
- **Fold-mean** (rata-rata prediksi test dari model per fold): robust, default untuk data
  kecil / model high-variance; efek ensemble fold gratis.
- **Refit full train**: 100% data; n_estimators/epoch disamakan ke region konvergensi dari
  CV (TIDAK early stopping    tidak ada validation set). Hanya kalau model stabil & data cukup.
- Keputusan ditulis dengan alasan di decision log.

### 8.2 Test-time parity
- Transformasi test identik train; tidak ada fit di test; scaler/encoder dari train-fold
  (atau full-train kalau refit-full) dipakai persis.
- Code competition: inference script mandiri + bobot lokal + uji runtime.

**GATE 8**: kebijakan final + alasan + checklist parity (scaler, encoder, transform target,
clip) tertulis.

---

## FASE 9    SUBMISSION & ROUND-TRIP TEST

1. **Assert vs sample_submission**: shape, kolom, URUTAN, dtype, NaN, inf, ID persis,
   rentang nilai.
2. **Round-trip test (khusus regresi transform)**:
   `assert np.allclose(inverse(transform(y)), y)` pada sampel target train. Kasus nyata:
   lupa `expm1` setelah training di log1p -> skor hancur tanpa error apapun dari kode.
3. **Sanity distribusi**: histogram prediksi test vs distribusi target train. Prediksi
   menumpuk di satu titik / mean jauh bergeser = bug pipeline.
4. **Multi-kandidat**: baseline, best single, blend    semua disimpan + di-log
   (`submission_log.csv`: kandidat, CV, LB, status).
5. Pilih final via CV; LB hanya kalibrasi.

**GATE 9**: submission lolos semua assert + round-trip + sanity plot + log terisi.

---

## FASE 10    WRAP-UP

- Decision log: setiap keputusan "kenapa X bukan Y" + eksperimen GAGAL dengan angkanya.
- metrics_summary.csv final + write-up teknis (MENGAPA, bukan hanya APA).
- Simpan semua OOF/test preds + folds + submission final untuk reproducibility.

---

## PERBANDINGAN: STANDARD vs KAGGLE-TIER (REGRESI)

| Aspek | Standard | Kaggle-tier (workflow ini) |
|---|---|---|
| Metrik | Diasumsikan dari nama | Bedah rumus -> rantai metrik-loss-transform eksplisit |
| Split | train_test_split acak | StratifiedKFold di bin target (qcut), fold dibekukan |
| Baseline | model pertama | mean & median per fold + submit awal |
| Outlier | langsung drop | dianalisis (noise floor, per-decile) baru diputuskan |
| Transform target | asal log | pohon keputusan skew/zeros/metrik + round-trip test |
| Fitur | semua dipakai | per-fold fit + selection by CV |
| Error analysis | tidak ada | residual-first: fan shape, per-decile, worst-N dibaca |
| Ensembling | model terbaik | bobot dari OOF + aturan SERI (< std) |
| Final | refit buta | refit-full vs fold-mean dipilih dengan alasan |
| Submission | save & submit | assert + round-trip + sanity distribusi + log |

## CHECKLIST PRE-FLIGHT (SEBELUM SUBMIT PERTAMA)

- [ ] Rumus metrik + arah + implikasi loss/transform tercatat dari rules.
- [ ] Target diaudit (zeros, negatif, skew, duplikat konflik).
- [ ] Fold dibekukan + sha256; baseline mean & median tercatat.
- [ ] Baseline sudah di-submit; format valid.

## CHECKLIST PRE-SUBMIT FINAL

- [ ] Semua eksperimen pakai fold sama; CV mean±std tercatat.
- [ ] Residual analysis selesai; verdict headroom/ceiling tertulis.
- [ ] Blend dioptimasi di OOF; aturan SERI diterapkan jujur.
- [ ] Round-trip inverse transform lolos assert.
- [ ] Prediksi di-clip ke rentang train; rounding sesuai format.
- [ ] Assert vs sample_submission lolos; sanity distribusi OK.
- [ ] Kandidat final dipilih via CV; submission_log lengkap.

## JEBAKAN KHAS REGRESI

| Jebakan | Gejala | Perbaikan |
|---|---|---|
| Loss MSE untuk lomba MAE | Model mengejar outlier, CV-MAE jelek | `objective="mae"` / pseudo-Huber |
| Lupa `expm1` (RMSLE/log target) | LB error raksasa tanpa crash | Round-trip test FASE 9.2 |
| Split acak tanpa stratifikasi target | Fold tak seimbang, std membengkak | qcut bin + StratifiedKFold |
| Target encoding di full-train | CV naik palsu | K-fold target encoding per fold |
| qcut error duplicate edges | Exception di data skewed | `rank(method="first")` sebelum qcut |
| MAPE dengan target ~0 | Metrik meledak tak hingga | Cek distribusi; negosiasikan clip/transform |
| Clipping ke rentang train dilupakan | Prediksi di luar range ditolak/dihukum | Clip [min,max] train |
| Duplikat fitur target beda tak diaudit | Noise floor tak diketahui, kejar fitur sia-sia | Audit FASE 1.1 sebagai ceiling estimate |
| Refit-full di data kecil | Skor final anjlok vs CV | Fold-mean default data kecil |
| GradientBoostingRegressor default di data besar | Lambat, underperform | LightGBM/CatBoost |
| Early stopping saat refit-full | Tidak ada validation set -> arbitrary | Samakan n_estimators ke konvergensi CV |
| Adversarial validation dilewati | CV bagus, LB jelek (shakeup) | FASE 2.1 poin 5 |


---

## MODUL PENINGKATAN MODEL    HUGGING FACE + GITHUB (OPSIONAL)

> Jalankan SETELAH FASE 3 (fold beku). Untuk regresi tabular: tabular foundation (zero-shot)
> + AutoML.

### Strategi pencarian & seleksi (filter berurutan)
1. **Task type**    HF: `pipeline_tag=tabular-regression`; GitHub: `"tabular regression kaggle solution"`.
2. **License (WAJIB)**    apache-2.0/mit/bsd aman; cc-by-nc & gpl berisiko; sinkronkan aturan lomba.
3. **Leaderboard performance**    HF sort `downloads`+`likes`; GitHub ⭐ tinggi + aktif + weights.
4. **Verifikasi murah**    TabPFN/AutoGluon zero-shot di fold beku DULU (menit); kalah dari LGBM =
   kandidat blend (diversitas), jangan paksa.
5. **Compliance**    catat source + lisensi di decision log.

### Shortlist regresi
| Kandidat | Kapan menang | Cost |
|---|---|---|
| `Prior-Labs/TabPFN-v2-regressor` (HF) | data kecil (<10k), zero-shot | rendah |
| AutoGluon (pip) | baseline otomatis + ensemble | sedang |
| LightGBM/XGBoost/CatBoost (GitHub) | data besar, Optuna | rendah |
| FT-Transformer / SAINT (GitHub) | data besar + diversitas NN | tinggi |
| Statsmodels GLM | count/zero-inflated | rendah |

**Aturan**: data >50k baris, GBDT tuned hampir selalu menang    jangan GPU untuk NN tabular
tanpa probe frozen menang dulu.

**GATE**: kandidat + lisensi + hasil zero-shot probe (CV mean±std) tercatat.


---

## BEST PRACTICES REVERSE-ENGINEERED    KAGGLE WINNERS (REGRESI)

> Gain tipikal antar-kompetisi, BUKAN jaminan; validasi fold beku; selisih < std = SERI.

| Teknik | Kapan dipakai | Expected gain | Cost |
|---|---|---|---|
| Adversarial validation | selalu, awal | preventif (cegah gap CV↔LB) | rendah |
| Stratified K-fold di bin target (qcut) | selalu | mengecilkan std, CV jujur | ~0 |
| OOF cross-fold | selalu | estimasi jujur + bahan blend | N× |
| Target encoding K-fold | kategorikal high-card | +0.5–2% | rendah |
| Rasio/interaksi domain | tabular | +1–5% (jump terbesar) | rendah |
| Feature selection (permutation) | fitur noise | +0.3–1% | rendah |
| Transform target (log1p/Yeo-Johnson) | skew > 2 | +1–10% (metrik-relative) | ~0 |
| Loss = metrik (MAE→L1, RMSLE→log-MSE) | selalu | krusial | ~0 |
| Optuna + early stopping | selalu | +0.5–2% | sedang |
| Multi-seed ensembling | GBDT/NN variansi | +0.1–0.5% | N× |
| Blending (weighted/rank) di OOF | model beragam | +0.3–1.5% | rendah |
| Stacking (nested CV) | model beragam | +0.5–2% | sedang |
| Pseudo-labeling | train kecil | +1–5% | sedang |
| Quantile regression / two-stage | ekor berat | +1–5% | sedang |
| Clipping + rounding | selalu post-proc | kecil tapi gratis | ~0 |
| External data / transfer | aturan izin | +1–10% (variabel) | tinggi |


---

## ORIENTASI METRIK & CHECKPOINTS OPTIMASI (REGRESI)

**Ladder prioritas (gain per cost menurun):**
1. Kunci rantai metrik→loss→transform. 2. Titik nol (mean & median) + fold beku bin target.
3. Noise floor (duplikat-konflik)    tahu kapan BERHENTI.
4. Cheap wins: transform target, rasio domain, target encoding, clipping.
5. Diversitas (GBDT + linear + TabPFN) + blend OOF. 6. Pseudo-label / stacking (setelah 1–5).

**📊 METRIC CHECKPOINT tiap fase (GATE):**
- [ ] FASE 3: CV baseline mean±std (mean & median) tercatat.
- [ ] FASE 4: tiap fitur Δ metrik; gain ≥ std fold.
- [ ] FASE 5: tiap model mean±std; Ridge pembanding.
- [ ] FASE 6: error per decile    headroom vs ceiling (dari noise floor).
- [ ] FASE 9: round-trip expm1 lolos; final dari CV metrik.
