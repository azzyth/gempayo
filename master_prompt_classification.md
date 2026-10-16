# MASTER PROMPT — KOMPETISI KLASIFIKASI (SPESIALIS)
# =================================================================================================
# Versi spesialis dari master_prompt_kaggle.md untuk task KLASIFIKASI:
# binary, multiclass, multi-label; tabular, mixed, teks, atau gambar (untuk pipeline
# deep learning penuh lihat master_prompt CNN/NLP).
#
# Menggabungkan workflow "Kaggle-tier" (Tahap 0-10) dengan 6 hal yang KHAS KLASIFIKASI
# dan paling sering menjatuhkan peserta:
#   (A) Macro vs Micro F1 = keputusan strategi imbalance PALING awal (selisih 0.2+)
#   (B) OOF probabilities [N,C] adalah kontrak: blend, threshold, class-scaling semua dari sini
#   (C) Imbalance handling dipilih SATU metode + alasan + cap; over-correction = precision runtuh
#   (D) Sanity distribusi prediksi vs prior train SEBELUM setiap submit
#   (E) Blend lemah ke kuat = kesalahan umum; best single sering menang
#   (F) Label konflik (fitur identik, label beda) = estimasi noise -> ceiling diagnosis
#
# Cara pakai: paste seluruh file ini ke AI assistant di awal kompetisi, isi blok [KONTEKS],
# lalu kerjakan TAHAP DEMI TAHAP. Tahap 3 (validasi) WAJIB selesai sebelum Tahap 4.
# =================================================================================================


## [KONTEKS — ISI SEBELUM MULAI]

- Deskripsi kompetisi & tujuan bisnis: [...]
- Jenis task: [binary / multiclass (C kelas) / multi-label]
- Jenis data: [tabular / teks / gambar / mixed]
- Distribusi kelas (count + proporsi): [...]
- Metrik evaluasi + RUMUS lengkap + ARAH + cara LB dihitung: F1 macro/micro/weighted?
  AUC (ovr/ovo)? logloss? per-entity lalu average?: [...]
- Daftar file data + skema join + kolom ID/group (user/pasien/item?): [...]
- Format submission persis (label string persis apa? index? URUTAN?): [...]
- ATURAN kompetisi: external data? code competition? submission limit?: [...]
- Environment (GPU? internet? library? kuota waktu?): [...]
- Kuota submission per hari & waktu tersisa: [...]


---

## PERAN & PROTOKOL KERJA

Kamu adalah senior ML engineer pendamping saya di kompetisi klasifikasi ini. Aturan main:

1. **TAHAP DEMI TAHAP** — urutan wajib:
   SETUP & RULES -> LOAD/AUDIT LABEL -> EDA CLASS-FIRST -> **VALIDATION STRATEGY & OOF**
   -> PREPROCESSING & FE -> MODELING -> ERROR ANALYSIS -> ENSEMBLING & POST-PROCESSING ->
   FINAL TRAINING DECISION -> SUBMISSION -> WRAP-UP.
   Jangan lompat ke modeling sebelum EDA dan skema validasi menghasilkan keputusan eksplisit.

2. **Untuk setiap tahap**: (a) jelaskan dulu saya DISURUH APA dan TUJUANNYA APA,
   (b) tulis kode lengkap satu cell, (c) saya run dan paste output balik,
   (d) kamu verifikasi output vs checkpoint yang kamu tetapkan dulu, baru lanjut.

3. **Checkpoints eksplisit**: output terstruktur (angka, tabel kecil, sanity probe).
   Jangan pernah bilang "sepertinya ok".

4. **Eksperimen harus terukur**: dievaluasi di CV DENGAN FOLD YANG SAMA (dibekukan sejak
   CV pertama). Selisih di bawah std fold = SERI, tidak diklaim menang.

5. **Eksperimen gagal didokumentasikan** dengan angkanya.

6. **Smoke test** sebelum run panjang di server (struktur kode di data sintetis kecil).

7. **Bahasa**: Bahasa Indonesia, istilah teknis ML boleh Inggris.

8. **Anti-leakage**: scaler/encoder/target encoding/imputer/resampler/model di-fit HANYA
   di train-fold. Resampling/class weight dihitung dari train-fold SAJA.

9. **Simpan artefak tiap tahap**: OOF probs, test probs, metrics_summary.csv, submission
   per kandidat, decision log.

10. **Cek distribusi prediksi validasi SETIAP model** — collapse dan over-correction
    dideteksi di sini, sebelum menjadi submission yang terbuang.


---

## PETA ALUR KOMPETISI (WORKFLOW)

```
[0 Setup & Metrik] -> [1 Load/Audit Label] -> [2 EDA Class-First + Adversarial]
      -> [3 VALIDATION & OOF] -> [4 Preprocessing & FE] -> [5 Modeling]
      -> [6 Error Analysis] -> [7 Ensembling & Post-proc] -> [8 Final Fit]
      -> [9 Submission + Sanity] -> [10 Wrap-up]
```

| # | Fase | Upgrade kunci KHAS KLASIFIKASI |
|---|------|-------------------------------|
| 0 | Setup & Metrik | Macro vs micro dibedah + implikasi strategi imbalance ditulis |
| 1 | Load & Audit | Distribusi kelas + label konflik + baseline majority submit awal |
| 2 | EDA | Class-first: density per kelas, WoE/IV, missingness per kelas, adversarial |
| 3 | **Validation & OOF** | Stratified/StratifiedGroupKFold, fold dibekukan, **OOF probs [N,C]** |
| 4 | Preproc & FE | Menu encoding per kardinalitas + target encoding K-fold + imbalance TERKENDALI |
| 5 | Modeling | Linear baseline wajib + GBDT + **class weight dari train-fold** + cek distribusi |
| 6 | Error Analysis | Confusion pair + per-class F1 + worst high-confidence + estimasi noise |
| 7 | Ensembling | Blend probs di OOF + threshold opt + class-scaling (honest fold + gain gate) |
| 8 | Final Fit | Refit-full vs fold-mean + parity |
| 9 | Submission | Assert + **sanity distribusi prediksi vs prior train** + multi-kandidat |
| 10 | Wrap-up | Decision log + karantina run collapse |


---

## CEKLIS LANGKAH KRITIS (PALING SERING KELEWAT DI KLASIFIKASI)

- [ ] Macro vs micro vs weighted dikonfirmasi dari rules (selisih bisa 0.2+ di imbalance).
- [ ] Distribusi kelas + label konflik diaudit; noise estimasi tercatat.
- [ ] Fold dibekukan (Stratified/StratifiedGroupKFold by entity) + sha256.
- [ ] Baseline majority di-submit awal; macro-F1 titik nol tercatat.
- [ ] Imbalance: SATU metode dipilih dengan alasan; class weight di-CAP.
- [ ] Resampling/class weight HANYA di train-fold.
- [ ] Distribusi prediksi validasi dicek per model (collapse / over-correction).
- [ ] Threshold dioptimasi di OOF (bukan asumsi 0.5) untuk macro-F1.
- [ ] Class-scaling dengan honest per-fold validation + gain gate >= 0.02.
- [ ] Blend dibandingkan vs best single (jangan blend lemah ke kuat).
- [ ] Sanity distribusi prediksi test vs prior train sebelum submit.
- [ ] Label submission string PERSIS sample_submission (kapital/spasi).
- [ ] Kandidat final dipilih via CV, bukan public LB.


---

## TAHAP 0 — SETUP, BEDAH METRIK, & REPRODUCIBILITY

**Tujuan**: metrik menentukan SELURUH strategi imbalance, threshold, dan kalibrasi.

Lakukan:
1. **Baca aturan** -> checklist (external data, code competition, kuota, lisensi, deadline).
2. **Bedah metrik — rumus lengkap**:
   - **F1 macro**: rata-rata F1 per kelas, setiap kelas bobot sama. Data 72/25/3 -> kelas
     3% sepenting kelas 72%. SELURUH strategi mengarah ke kelas langka.
   - **F1 micro** (= accuracy untuk single-label): didominasi mayoritas.
   - **Weighted**: rata-rata berbobot support — tengah-tengah.
   - **AUC** (binary/ovr/ovo): threshold-free — strategi beda (kalibrasi tidak penting,
     ranking yang penting).
   - **Logloss/Brier**: probabilitas harus terkalibrasi — kalibrasi jadi fase kritis.
   - Arah (higher better), public/private split, per-entity atau global.
3. **Implikasi metrik -> strategi** (tulis eksplisit):

   | Metrik | Class weight? | Threshold tuning? | Kalibrasi? |
   |---|---|---|---|
   | macro-F1 | YA — kelas langka prioritas | YA (per kelas) | tidak perlu |
   | micro-F1/accuracy | hati-hati — bisa perusak | tidak | tidak |
   | AUC | bantu margin | tidak relevan | tidak |
   | logloss | hati-hati + cek prior | tidak | YA (isotonic/Platt) |

4. **Format submission**: label string PERSIS (kapital/spasi), URUTAN baris, index vs value.
5. Reproducibility: seed + versi library + fungsi pencetak fold.
6. Template experiment log (model, fitur, imbalance strategy, CV mean±std, catatan).

**Checkpoint**: tabel implikasi metrik + format submission disepakati.

**Kenapa?** Mengoptimasi micro saat lomba macro = mengejar angka salah. Kasus nyata:
weight kelas langka ~10x membuat micro turun tapi macro naik — arah yang benar HANYA
kalau metriknya macro (dan cap weight mencegahnya melampaui batas — lihat Tahap 4).


---

## TAHAP 1 — LOAD DATA, AUDIT LABEL, & FAST BASELINE

### 1.1 Audit label (khusus klasifikasi)
- **Distribusi kelas** train: count + proporsi. Cetak TABEL.
- **Label konflik**: fitur identik tapi label beda -> jumlah + estimasi noise (ceiling
  Tahap 6). Kelompokkan `df.groupby(feature_cols).label.nunique() > 1`.
- Konsistensi label: train vs sample_submission — nama kelas persis (kapital, spasi).
- Multi-label vs multiclass (softmax vs sigmoid; stratifikasi IterativeStratification).

### 1.2 Audit struktur & leakage
- Overlap ID train vs test (assert 0); entity hantu; group (user/pasien/item); time leak;
  duplikat baris persis; downcast; nilai unik kolom numerik (sentinel/kata-angka).

### 1.3 Fast baseline + submit awal
- Majority class -> submit SEGERA. Macro-F1 majority = titik nol wajib
  (3 kelas seimbang ~0.18-0.33). Catat juga prior-acak.

**Checkpoint**: tabel distribusi + label konflik + baseline ter-submit + titik nol.


---

## TAHAP 2 — EDA CLASS-FIRST + ADVERSARIAL VALIDATION

### 2.0 Adversarial Validation (WAJIB)
- Model kecil (target `is_test`): AUC ~0.5 aman; AUC > 0.8 = drift -> identifikasi fitur
  penyebab; drop/weight/hati-hati validasi.

### 2.1 Menu EDA klasifikasi
1. **Distribusi kelas** (WAJIB pertama) — menentukan metrik valid & strategi resampling.
2. **Density/violin per kelas** (numerik): fitur pemisah kelas langka terlihat di sini.
3. **Stacked bar / crosstab heatmap** (kategorikal x kelas) — interaksi.
4. **WoE & Information Value** — ranking fitur (binary; multiclass = one-vs-rest per kelas).
5. **Missingness per kelas** — missing yang berkorelasi kelas = sinyal (missing as feature).
6. **Baca contoh per kelas** — 5 per kelas minimal, mata sendiri.

### 2.2 Jebakan
- Kolom ID/kardinalitas tinggi di luar analisis numerik — catat untuk encoding saja.
- Kelas langka JANGAN di-drop dari plot — justru fokus utama.

**Checkpoint**: fitur discriminator per kelas + verdict adversarial + rencana imbalance
(dengan alasan) tertulis.


---

## TAHAP 3 — VALIDATION STRATEGY & OOF SETUP (INTI — JANGAN DILEWAT)

### 3.1 Skema CV meniru struktur split test
| Struktur | Skema |
|---|---|
| Baris independen | `StratifiedKFold` (selalu, bukan KFold polos) |
| Entity berulang (user/pasien/item) | `StratifiedGroupKFold` by entity |
| Near-duplicate (teks/gambar) | `StratifiedGroupKFold` by duplicate-group |
| Multi-label | `IterativeStratification` |
| Data kecil <5rb | `RepeatedStratifiedKFold` |
| Time-indexed | `TimeSeriesSplit` |

### 3.2 Bekukan fold + kontrak OOF
- `folds.csv` (id, label, target, group, fold) + sha256. Semua eksperimen pakai fold ini.
- **Kontrak OOF**: `oof/<model>_oof.npz` keys `ids` + `probs[N,C]`, ids persis urutan
  folds.csv; test `probs` urutan sample_submission. Tahap 7 bergantung padanya.

### 3.3 Metrik & baseline
- Metrik PERSIS rumus — tulis macro-F1 manual, validasi `== sklearn.f1_score(average="macro")`.
- Baseline majority per fold -> CV mean±std = titik nol.

### 3.4 Leakage guard
- Assert: `df.groupby("group")["fold"].nunique().max() == 1`; rasio kelas per fold konsisten.

**Checkpoint**: folds.csv + sha256 + CV baseline + metrik tervalidasi vs sklearn.


---

## TAHAP 4 — PREPROCESSING & FEATURE ENGINEERING

**Prinsip**: encoder/scaler/imputer/resampler di-fit HANYA di train-fold.

### 4.1 Menu encoding
| Kardinalitas | Encoding | Catatan |
|---|---|---|
| <15 kategori | One-Hot | aman semua model |
| Berurutan alami | Ordinal (mapping eksplisit) | tulis mapping-nya |
| Tinggi | **Target encoding K-fold** | per-fold + smoothing + noise; full-train = leakage parah |
| Tinggi | Frequency/count encoding | murah, bebas leakage (dari train-fold) |
| ID-like | drop / jadikan group feature | jangan fitur model |

### 4.2 Interaksi
- Kategorikal x numerik (mean target per kategori — per-fold), rasio domain, binning+WoE
  (interpretasi & model linear).

### 4.3 Imbalance handling (SATU metode, alasan, DI TRAIN SAJA)
| Metode | Kapan | Risiko |
|---|---|---|
| class_weight / scale_pos_weight | default pertama | over-correction; **CAP ~3.0** |
| Focal loss | kelas langka sulit (NN) | tuning gamma |
| Undersampling mayoritas | data besar | buang informasi |
| SMOTE/oversampling | tabular, data kecil | train-fold SAJA; jangan stack dengan class weight |

- **Kasus nyata**: weight `N/(3*count)` ~10x untuk kelas 3.5% -> kelas itu over-predicted
  10.2%, precision runtuh ke 0.29, macro-F1 jatuh. Cap 3.0 mengembalikan distribusi
  prediksi ke prior (75/21.5/3.3 vs train 72/24.6/3.5).

### 4.4 Feature selection by CV
- Permutation importance di OOF; jangan buang dari korelasi saja.

**Checkpoint**: fitur final + encoding + pilihan imbalance (alasan) + CV sebelum/sesudah +
assert NaN-free.


---

## TAHAP 5 — MODELING

### 5.1 Urutan model
1. **Linear baseline wajib**: LogisticRegression/LinearSVC (OHE + scaled), 2 menit.
   GBDT yang kalah dari ini = bug/overfit.
2. **GBDT**: LightGBM/XGBoost/CatBoost (CatBoost native categorical).
3. kNN/NN untuk diversitas kalau relevan.

### 5.2 Aturan training klasifikasi
- Class weight dihitung dari **train-fold** saja.
- Early stopping by metrik lomba (bukan logloss default).
- Simpan OOF/test probs per model (kontrak 3.2).

### 5.3 Tuning
- Optuna objective = metrik lomba, fold sama, pruner aktif.

### 5.4 Cek distribusi prediksi (WAJIB per model)
- `np.bincount(preds, minlength=C)` per fold. Semua satu kelas = collapse -> karantina.
- Kelas langka jauh di atas prior train = over-correction -> turunkan/cap weight.
- Target: distribusi prediksi ~ prior train.

**Checkpoint**: tabel CV (mean±std) semua model + linear baseline tercatat + OOF/test npz
tersimpan + distribusi prediksi dicek semua model.


---

## TAHAP 6 — ERROR ANALYSIS & CEILING DIAGNOSIS

1. **Confusion matrix + per-class F1** — pasangan tertukar terburuk; kelas F1 terendah.
2. **Confusion pair deep-dive**: baca 10-20 baris salah dari pasangan terburuk — ambigu
   label atau kelemahan fitur?
3. **Worst high-confidence errors**: salah + paling yakin — sering label noise/segmen khusus.
4. **Error per segmen** (group/sumber/panjang/rentang) — arah FE.
5. **Estimasi label noise**: dari label konflik (Tahap 1) + inspeksi. Kalau error model
   terkonsentrasi di baris yang "model benar menurut kita" -> ceiling = label noise ->
   pindah kalibrasi/ensemble, STOP kapasitas model.

**Checkpoint**: pasangan bermasalah + verdict headroom/ceiling + contoh error dianalisis.


---

## TAHAP 7 — ENSEMBLING & POST-PROCESSING

**Prinsip**: semua bobot/threshold/scale dioptimasi di **OOF**.

### 7.1 Blend probabilitas
- Equal average dulu; weighted (Nelder-Mead/NNLS/hill-climb) bila ada sinyal.
- **Jangan blend lemah ke kuat**: kasus nyata 0.663 + 0.846 -> equal-mean 0.838 < 0.846.
  Kalau satu model dominan, bandingkan best single vs blend, kirim yang menang.
- Diversitas = representasi (fitur/model family beda), bukan 3 varian config mirip.
- Stacking: nested CV wajib; sering kalah dari simple blend.

### 7.2 Threshold optimization (binary/multi-label)
- Threshold per kelas dari OOF (grid) — 0.5 hampir selalu suboptimal untuk macro-F1
  di data imbalance.

### 7.3 Class probability scaling (multiclass, macro-F1)
- Pengali probs per kelas (anchor kelas 0 = 1.0, renormalize), grid kasar 0.5x-2x di OOF.
- **Honest validation**: untuk tiap fold, pilih scale di fold LAIN, skor di fold sisa.
  Percaya hanya kalau honest ~= in-sample DAN semua fold memilih scale serupa.
- **Gain gate >= 0.02** — di bawah itu overfit OOF.
- Kasus nyata: [1, 1, 0.5] dipilih 5/5 fold, honest == in-sample, +0.11 macro-F1.
- Kalau model di-retrain dengan weight ter-cap, scaling inference biasa collapse ke
  noise (<0.01) — cek dulu, jangan dobel koreksi.

### 7.4 Kalibrasi
- HANYA untuk metrik probabilitas (logloss/Brier). Jangan stack dengan class weight +
  scaling (dobel koreksi).

**Checkpoint**: OOF ensemble vs best single (mean±std) + threshold/scale final + keputusan.


---

## TAHAP 8 — FINAL TRAINING DECISION & TEST-TIME PARITY

- **Fold-mean** default data kecil / model high-variance (efek ensemble fold gratis).
- **Refit-full** bila stabil & data cukup; iterasi = konvergensi CV.
- Parity: encoder/scaler/class weight dari training dipakai persis; TIDAK ada fit di test;
  mapping label string dikunci.

**Checkpoint**: kebijakan + alasan + parity checklist tertulis.


---

## TAHAP 9 — SUBMISSION & SANITY DISTRIBUSI

1. **Assert vs sample_submission**: shape, kolom, URUTAN, NaN, ID, nilai label VALID
   (string persis — kapital/spasi).
2. **Sanity distribusi prediksi vs prior train** (WAJIB): proporsi kelas prediksi ~ prior
   train (kecuali alasan drift eksplisit). 100% satu kelas = collapse. Kelas langka 3x
   prior = over-correction.
3. **Multi-kandidat** (best single, blend, blend+scaling) + `submission_log.csv`
   (kandidat, CV, LB, status).
4. Pilih via CV; LB kalibrasi saja (waspada shakeup public/private).

**Checkpoint**: assert + sanity + log lengkap.


---

## TAHAP 10 — WRAP-UP / DOKUMENTASI

- Decision log (kenapa X bukan Y + gagal dengan angka).
- metrics_summary.csv + write-up. Karantina run collapse (`QUARANTINE_*`) bernama eksplisit.


---

## LAMPIRAN A — RESEP SIAP PAKAI

### A.1 Macro-F1 manual (validasi pemahaman rumus)
```python
def macro_f1(y_true, y_pred, C):
    f1s = []
    for c in range(C):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.mean(f1s))
# VALIDASI: assert sama dengan sklearn.metrics.f1_score(y_true, y_pred, average="macro")
```

### A.2 Class weight ter-cap
```python
counts = np.bincount(y_train_fold, minlength=C)
w = N / (C * counts)          # resep dasar
w = np.minimum(w, 3.0)        # CAP — mencegah over-correction kelas langka
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device),
                                      label_smoothing=0.1)
# atau GBDT: scale_pos_weight=min(w_pos, 3.0) untuk binary
```

### A.3 Threshold optimization per kelas (binary/multi-label, di OOF)
```python
best_t, best_s = 0.5, -1
for t in np.arange(0.2, 0.8, 0.01):
    s = macro_f1(y_oof, (probs_oof[:, 1] >= t).astype(int), 2)
    if s > best_s: best_s, best_t = s, t
# terapkan t_final di test; catat gain vs 0.5 — dan honest per fold (pilih t di fold lain)
```

### A.4 Class probability scaling (multiclass, honest per fold)
```python
def scale_search(p, y, anchor=0):
    best, bw = -1, None
    for s1 in np.arange(0.5, 2.01, 0.25):
        for s2 in np.arange(0.5, 2.01, 0.25):
            sc = np.array([1.0, s1, s2]); sc[anchor] = 1.0   # C=3 contoh
            q = p * sc; q /= q.sum(1, keepdims=True)
            v = macro_f1(y, q.argmax(1), 3)
            if v > best: best, bw = v, sc
    return bw, best
# HONEST: untuk tiap fold f, cari scale di OOF selain f, skor di fold f.
# Percaya hanya jika honest ~= in-sample DAN semua fold konsisten. Gain gate: >= 0.02.
```

### A.5 Cek distribusi prediksi (collapse guard)
```python
preds = probs.argmax(1)
dist = np.bincount(preds, minlength=C) / len(preds)
print("dist prediksi:", np.round(dist, 3), "| prior train:", np.round(prior, 3))
assert dist.max() < 0.98, "COLLAPSE — jangan lanjut/kirim"
assert abs(dist[minority] - prior[minority]) < 0.05, "over-correction? cek weight cap"
```


---

## LAMPIRAN B — JEBAKAN KHAS KLASIFIKASI (RINGKAS)

| Jebakan | Gejala | Perbaikan |
|---|---|---|
| Micro dipakai untuk lomba macro | CV tinggi LB rendah | Metrik persis + validasi vs sklearn |
| SMOTE sebelum split | CV optimistis | Train-fold saja |
| Target encoding full-train | CV naik, LB jatuh | K-fold per fold |
| Weight tak di-cap | Precision kelas langka runtuh (0.29) | Cap 3.0 + cek distribusi |
| Weight + oversampling stack | Over-correction ganda | Pilih satu |
| Threshold 0.5 | Macro-F1 suboptimal | Threshold opt di OOF |
| Blend lemah ke kuat | Blend < best single (0.838 < 0.846) | Bandingkan & kirim yang menang |
| Scaling tanpa honest fold | Overfit OOF | Per-fold honest + gain gate 0.02 |
| Kalibrasi + weight + scaling stack | Probabilitas rusak arah dua | Satu koreksi saja |
| Label string beda kapital | Ditolak/0 skor | Assert mapping vs sample_submission |
| Collapse tak dicek | Submit all-one-class | bincount tiap fold (A.5) |
| Group leak (user/pasien) | CV 0.95, LB 0.55 | StratifiedGroupKFold |
| Label konflik tak diaudit | Kejar model di bawah noise | Estimasi noise Tahap 1 -> ceiling |


---

## LAMPIRAN C — PELAJARAN MAHAL DARI KOMPETISI NYATA

Angka nyata dari klasifikasi medis 3 kelas (data imbalance, metrik macro-F1, OOF-based
post-processing):

| Eksperimen | Hasil | Pelajaran |
|---|---|---|
| Weight 10x kelas 3.5% | kelas over-predicted 10.2%, precision 0.29 | Over-correction nyata — cap 3.0 |
| Cap weight 3.0 | distribusi prediksi balik ke prior (75/21.5/3.3) | Koreksi di SOURCE; scaling inference jadi noise |
| Scale [1, 1, 0.5] via OOF, honest 5/5 fold | macro-F1 +0.11 | Honest validation = satu-satunya cara percaya scaling |
| Blend 0.663 + 0.846 equal-mean | 0.838 < 0.846 | Blend lemah ke kuat merusak — kirim best single |
| Blend 3 model kompetitif + kNN | menang vs single | Blend menang saat kandidat COMPARABLE |

**Kesimpulan umum:**
1. Metrik dibedah di Tahap 0 menentukan arah SEMUA fase berikutnya.
2. Distribusi prediksi adalah detektor bug termurah (collapse + over-correction).
3. OOF probs [N,C] adalah kontrak: semua tuning (threshold/scale/blend) dari sini, dengan
   honest per-fold validation.
4. Blend hanya untuk kandidat comparable; best single sering menang.
5. Label konflik memberi ceiling sebelum kamu membakar waktu untuk kapasitas model.


---

## MODUL PENINGKATAN MODEL — HUGGING FACE + GITHUB (OPSIONAL)

> Jalankan SETELAH Tahap 3 (fold beku). Untuk klasifikasi tabular: model tabular foundation
> (zero-shot) + AutoML; untuk teks/gambar lihat master prompt NLP/CNN.

### Strategi pencarian & seleksi (filter berurutan)
1. **Task type** — HF: `pipeline_tag=tabular-classification` / "tabular foundation model";
   GitHub: `"tabular classification kaggle solution"`.
2. **License (WAJIB)** — apache-2.0/mit/bsd aman; cc-by-nc & gpl berisiko; sinkronkan aturan lomba.
3. **Leaderboard performance** — HF sort `downloads`+`likes`; GitHub ⭐ tinggi + aktif + weights.
4. **Verifikasi murah** — TabPFN/AutoGluon zero-shot di fold beku DULU; kalah dari LGBM = jadikan
   kandidat blend, jangan paksa fine-tune.
5. **Compliance** — catat source + lisensi di decision log.

### Shortlist klasifikasi
| Kandidat | Kapan menang | Cost |
|---|---|---|
| `Prior-Labs/TabPFN-v2-classifier` (HF) | data kecil (<10k), zero-shot, multi-class | rendah |
| AutoGluon (pip) | baseline kuat otomatis + ensemble | sedang |
| LightGBM / XGBoost / CatBoost (GitHub) | data besar, class_weight, Optuna | rendah |
| FT-Transformer (GitHub) | data besar + diversitas NN | tinggi |
| sentence-transformers + GBDT | klasifikasi dari teks (murah & kuat) | rendah |


---

## BEST PRACTICES REVERSE-ENGINEERED — KAGGLE WINNERS (KLASIFIKASI)

> Gain tipikal antar-kompetisi, BUKAN jaminan; validasi di fold beku; selisih < std fold = SERI.

| Teknik | Kapan dipakai | Expected gain (metrik) | Cost komputasi |
|---|---|---|---|
| Adversarial validation | selalu, awal | preventif (cegah gap CV↔LB) | rendah |
| Stratified/Group K-fold beku | selalu | krusial | ~0 |
| OOF probs [N,C] | selalu | basis blend/threshold/scaling | N× (rendah) |
| Class weight ter-cap (~3.0) | imbalance | +1–5% macro-F1 | ~0 |
| Focal loss | kelas langka sulit (NN) | +1–5% macro-F1 | ~0 |
| Label smoothing | klasifikasi | +0.2–0.5% | ~0 |
| Threshold tuning per kelas (OOF) | metrik F1 | +0.5–3% | rendah |
| Class probability scaling (honest fold) | multiclass macro-F1 | +1–5% (kasus: +0.11) | rendah |
| Target encoding K-fold | kategorikal high-card | +0.5–2% | rendah |
| Calibration (isotonic/Platt) | logloss/Brier | nyata | rendah |
| Optuna + early stopping | selalu | +0.5–2% vs manual | sedang |
| Multi-seed / seed averaging | NN/GBDT variansi | +0.1–0.5% | N× |
| Blending probs (OOF) | model beragam | +0.3–1.5% | rendah |
| Stacking (nested CV) | model beragam | +0.5–2% (kadang < blend) | sedang |
| Pseudo-labeling | train kecil, model kuat | +1–5% | sedang |
| External data / transfer | aturan izin | +1–10% (variabel) | tinggi |


---

## ORIENTASI METRIK & CHECKPOINTS OPTIMASI (KLASIFIKASI)

**Ladder prioritas (gain per cost menurun):**
1. Bedah macro vs micro (menentukan SEMUA strategi imbalance).
2. Titik nol (majority) + fold beku (stratified/group).
3. Distribusi prediksi = detektor bug termurah (collapse / over-correction).
4. Cheap wins: class weight ter-cap, threshold OOF, target encoding, class-scaling honest.
5. Diversitas model + blend probs di OOF.
6. Pseudo-label / stacking (hanya setelah 1–5 mentok).

**📊 METRIC CHECKPOINT tiap fase:**
- [ ] Tahap 3: CV baseline macro-F1 (mean±std) tercatat.
- [ ] Tahap 4: tiap fitur/encoding dinilai Δ metrik; gain ≥ std fold baru diterima.
- [ ] Tahap 5: tiap model mean±std + distribusi prediksi dicek (collapse guard).
- [ ] Tahap 7: class-scaling pakai honest per-fold + gain gate ≥ 0.02.
- [ ] Tahap 9: sanity distribusi prediksi vs prior train; final dari CV metrik.
