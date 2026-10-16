<div align="center">

# 🚨 GempaWaspada

**Earthquake risk & impact prediction for Indonesia: when the next significant quake hits, and how bad the damage will be.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-classifier-2b6cb0)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)

</div>

---

## What it does

GempaWaspada answers the two questions a disaster-early-warning tool must answer:

1. **When?** (seismicity) - the rate and expected gap between significant earthquakes, per region, from a complete USGS catalog.
2. **How big?** (severity) - the likely damage tier of an earthquake (Ringan / Sedang / Berat), from magnitude, depth, felt intensity, tsunami risk, and population exposure.

Both live in a FastAPI backend and a single-page dashboard, built for the JOINTS UGM 2026 hackathon (Sustainability theme).

## Data

| source | what it provides | size |
|---|---|---|
| BMKG catalog (`katalog_bmkg.pdf` parsed to CSV) | 307 significant or damaging earthquakes 2000-2023, with damage reports | labels + hazard features |
| USGS FDSN API | 5,722 M>=5.0 earthquakes 2000-2023, complete seismicity | the "when" signal |
| GeoNames | 258,055 Indonesian settlements + city populations | exposure features |

## How it works

### "Kapan" is a Poisson process, not a feature model

Predicting an exact earthquake time is unsolved science. The honest, useful answer is the seismicity rate: a M>=5.0 earthquake happens somewhere in Indonesia every ~0.8 days (median). Per region the rate spans ~85 events/year (Sulawesi-Papua) down to ~13/year (Java-south).

A feature-based regression on "days to next event" was trained and, honestly, did not beat the median baseline. So the shipped answer is the rate table, not a pretend forecast.

### "Seberapa besar" is a severity classifier

Damage = hazard x exposure x vulnerability. The catalog gives hazard (magnitude, depth, MMI, tsunami); GeoNames fills the exposure gap (distance to nearest settlement, settlement density within 50/100 km, population within 100 km).

LightGBM classifies three severity tiers built from reported fatalities:

| tier | definition | count |
|---|---|---|
| Ringan | 0 deaths | 237 |
| Sedang | 1-9 deaths | 47 |
| Berat | >=10 deaths | 23 |

Evaluated on a frozen stratified 80/20 split (`seed: 42`):

| model | macro-F1 | Ringan F1 | Sedang F1 | Berat F1 |
|---|---|---|---|---|
| majority baseline | 0.291 | - | - | - |
| LogisticRegression | 0.472 | 0.658 | 0.286 | 0.471 |
| **LightGBM + exposure** | **0.552** | 0.833 | 0.222 | 0.600 |

Honest caveats, also in `findings.md`:

- The "Sedang" tier stays weak because building vulnerability is not in the data.
- Exposure features are an interpretable alternative to raw lat/lon (same macro-F1, better "Sedang"), not a big accuracy jump.

## Quickstart

```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

The backend trains the LightGBM model on startup (307 rows) and serves:

| endpoint | what it returns |
|---|---|
| `GET /api/summary` | KPI + model metrics |
| `GET /api/seismicity` | USGS M>=5.0 rates + median gap per region |
| `GET /api/earthquakes` | the 307 BMKG events |
| `GET /api/confusion` | severity confusion matrix |
| `POST /api/predict` | severity prediction + live exposure readout |

## Reproduce

```bash
python data/fetch_external.py       # download USGS + GeoNames (needs internet)
python data/process_external.py     # build processed CSVs
python data/process_id_places.py    # extract 258k settlements
python eda/03_clean.py              # clean the BMKG catalog
python eda/05_exposure_features.py  # join exposure features
python modeling/09_severity_train_test.py  # severity train/test
python modeling/11_when_usgs.py     # seismicity rates
```

## Project layout

```
gempayo/
├─ extract.py                 # pdfplumber parser: BMKG PDF -> CSV
├─ eda/                       # 01 audit .. 05 exposure features
├─ modeling/                  # 05 validation .. 11 when-usgs
├─ data/                      # fetch scripts + processed CSVs (id_places, id_cities, usgs)
├─ backend/app.py             # FastAPI: LightGBM + exposure + seismicity
├─ frontend/                  # vanilla HTML/CSS/JS dashboard
├─ artifacts/                 # folds, risk tables, OOF predictions
├─ findings.md                # durable decisions + honest caveats
└─ gempabumi_exposure.csv     # clean data + exposure features
```

## What I'd do next

- Add building vulnerability or a population grid (WorldPop / GHSL) to sharpen the "Sedang" tier.
- Fine-tune severity thresholds per region.
- Turn the seismicity rate into a per-region "risk horizon" estimate.

---

*Built for the JOINTS UGM 2026 hackathon (Sustainability theme), from BMKG + USGS + GeoNames data.*
