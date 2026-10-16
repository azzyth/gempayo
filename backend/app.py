"""GempaWaspada API — prototype "kapan" & "seberapa besar" gempa Indonesia.

- Severity ("seberapa besar"): LightGBM + fitur exposure (GeoNames).
- Seismisitas ("kapan"): USGS M>=5.0 (median gap & laju per wilayah).
- Frontend statis disajikan dari /frontend.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
DATA = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")

NUM_COLS = ["Magnitude", "Depth_km", "MMI_max", "dist_nearest_settlement_km",
            "n_settlements_50km", "n_settlements_100km", "log_pop_100km"]
CAT_COLS = ["Tsunami", "Region"]
CLASS_NAMES = {0: "Ringan", 1: "Sedang", 2: "Berat"}
CLASS_ORDER = ["Ringan", "Sedang", "Berat"]
R_KM = 6371.0


def _load_frame() -> pd.DataFrame:
    exp = pd.read_csv(os.path.join(ROOT, "gempabumi_exposure.csv"), encoding="utf-8-sig")
    sev = pd.read_csv(os.path.join(ROOT, "gempabumi_model.csv"), encoding="utf-8-sig")["Severity"]
    df = exp.copy()
    df["Severity"] = sev.values
    df["Tsunami"] = df["Tsunami"].astype(str)
    df["Region"] = df["Region"].astype(str)
    return df


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    X = d[NUM_COLS + CAT_COLS].copy()
    X["MMI_max"] = X["MMI_max"].fillna(0)
    return X


# ---- state global ----
DF = _load_frame()
Y = DF["Severity"].values

PLACES = pd.read_csv(os.path.join(DATA, "id_places.csv"))
CITIES = pd.read_csv(os.path.join(DATA, "id_cities.csv"))
TREE_PLACES = BallTree(np.radians(PLACES[["latitude", "longitude"]].to_numpy()), metric="haversine")
TREE_CITIES = BallTree(np.radians(CITIES[["latitude", "longitude"]].to_numpy()), metric="haversine")
CITY_POP = CITIES["population"].to_numpy()

REGION_CENTROIDS = DF.groupby("Region")[["Latitude", "Longitude"]].mean().round(3).to_dict("index")
REGIONS = [
    {"name": r, "lat": c["Latitude"], "lon": c["Longitude"],
     "events": int((DF["Region"] == r).sum())}
    for r, c in sorted(REGION_CENTROIDS.items(), key=lambda kv: -int((DF["Region"] == kv[0]).sum()))
]

USGS = pd.read_csv(os.path.join(DATA, "usgs_earthquakes_clean.csv"))
USGS["time"] = pd.to_datetime(USGS["time"], format="ISO8601", utc=True)
USGS = USGS.sort_values("time").reset_index(drop=True)
USGS_GAP = USGS["time"].diff().dt.total_seconds() / 86400
USGS_MEDIAN_GAP = float(USGS_GAP.median())
USGS_MEAN_GAP = float(USGS_GAP.mean())
USGS_RISK = pd.read_csv(os.path.join(ARTIFACTS, "usgs_occurrence_risk.csv"), encoding="utf-8-sig")

PRE = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])


def _fit_lgbm(X, y):
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=200, learning_rate=0.05,
        num_leaves=15, min_child_samples=10, class_weight="balanced",
        random_state=42, verbosity=-1,
    )
    m.fit(X, y)
    return m


# final model pada full data (untuk prediksi live)
X_FULL = _prep(DF)
PRE.fit(X_FULL)
FINAL_MODEL = _fit_lgbm(PRE.transform(X_FULL), Y)


# ---- metrik evaluasi (stratified split, jujur) ----
def _eval_metrics():
    X_tr, X_te, y_tr, y_te = train_test_split(X_FULL, Y, test_size=0.2, stratify=Y, random_state=42)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    X_tr = pre.fit_transform(X_tr)
    X_te = pre.transform(X_te)
    m = _fit_lgbm(X_tr, y_tr)
    pred = m.predict(X_te)
    macro = float(f1_score(y_te, pred, labels=[0, 1, 2], average="macro", zero_division=0))
    per = {CLASS_NAMES[i]: round(float(f1_score(y_te, pred, labels=[i], average="macro", zero_division=0)), 3)
           for i in range(3)}
    conf = confusion_matrix(y_te, pred, labels=[0, 1, 2]).tolist()
    tp = conf[2][2]
    fn = sum(conf[2]) - tp
    recall_berat = round(tp / max(tp + fn, 1), 3)
    return macro, per, conf, recall_berat


MACRO_F1, PER_CLASS, CONFUSION, RECALL_BERAT = _eval_metrics()


def compute_exposure(lat: float, lon: float) -> dict:
    p = np.radians([[lat, lon]])
    d, _ = TREE_PLACES.query(p, k=1)
    dist_near = float(d[0, 0] * R_KM)
    n50 = int(TREE_PLACES.query_radius(p, r=50 / R_KM, count_only=True)[0])
    n100 = int(TREE_PLACES.query_radius(p, r=100 / R_KM, count_only=True)[0])
    idx = TREE_CITIES.query_radius(p, r=100 / R_KM)[0]
    pop100 = int(CITY_POP[idx].sum())
    return {
        "dist_nearest_settlement_km": round(dist_near, 2),
        "n_settlements_50km": n50,
        "n_settlements_100km": n100,
        "log_pop_100km": round(float(np.log1p(pop100)), 4),
        "pop_within_100km": pop100,
    }


app = FastAPI(title="GempaWaspada API", version="2.0.0")


class PredictRequest(BaseModel):
    magnitude: float
    depth_km: float
    mmi_max: float
    tsunami: bool
    region: str
    latitude: float | None = None
    longitude: float | None = None


@app.get("/api/summary")
def summary():
    sev = DF["Severity"].value_counts().sort_index()
    return {
        "n_events": int(len(DF)),
        "n_tsunami": int((DF["Tsunami"] == "Tsunami").sum()),
        "n_years": int(DF["Tahun"].nunique()),
        "avg_per_year": round(len(DF) / DF["Tahun"].nunique(), 1),
        "severity": {CLASS_NAMES[i]: int(sev.get(i, 0)) for i in range(3)},
        "usgs_n_events": int(len(USGS)),
        "usgs_median_gap_hours": round(USGS_MEDIAN_GAP * 24, 1),
        "usgs_mean_gap_hours": round(USGS_MEAN_GAP * 24, 1),
        "macro_f1": round(MACRO_F1, 3),
        "baseline_macro_f1": 0.291,
        "recall_berat": RECALL_BERAT,
        "per_class_f1": PER_CLASS,
    }


@app.get("/api/seismicity")
def seismicity():
    rows = []
    for _, r in USGS_RISK.iterrows():
        rows.append({
            "region": r["Region"],
            "events": int(r["n"]),
            "lambda": round(float(r["lambda_per_year"]), 2),
            "median_gap_days": round(float(r["median_gap_days"]), 2),
            "p_ge1": round(float(r["p_ge1_next_year"]), 3),
            "max_mag": round(float(r["max_mag"]), 1),
        })
    return {
        "total_events": int(len(USGS)),
        "median_gap_days": round(USGS_MEDIAN_GAP, 2),
        "mean_gap_days": round(USGS_MEAN_GAP, 2),
        "threshold": "M≥5.0",
        "regions": rows,
    }


@app.get("/api/regions")
def regions():
    return REGIONS


@app.get("/api/trend")
def trend():
    df = DF.copy()
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    yearly = df.groupby("Year").size()
    monthly = df.groupby("Month").size().reindex(range(1, 13), fill_value=0)
    return {
        "years": [int(y) for y in yearly.index],
        "year_counts": [int(c) for c in yearly.values],
        "months": [int(m) for m in monthly.index],
        "month_counts": [int(c) for c in monthly.values],
    }


@app.get("/api/earthquakes")
def earthquakes():
    rows = []
    for _, r in DF.iterrows():
        rows.append({
            "year": int(r["Tahun"]),
            "wilayah": r["Wilayah_nama"],
            "region": r["Region"],
            "magnitude": round(float(r["Magnitude"]), 1),
            "depth": round(float(r["Depth_km"]), 1),
            "mmi": None if pd.isna(r["MMI_max"]) else int(r["MMI_max"]),
            "tsunami": bool(r["Tsunami"] == "Tsunami"),
            "severity": int(r["Severity"]),
            "fatalities": None if pd.isna(r["Korban_meninggal_est"]) else int(r["Korban_meninggal_est"]),
        })
    return sorted(rows, key=lambda x: (x["year"], -x["magnitude"]))


@app.get("/api/confusion")
def confusion():
    return {"labels": CLASS_ORDER, "matrix": CONFUSION}


@app.post("/api/predict")
def predict(req: PredictRequest):
    lat = req.latitude if req.latitude is not None else REGION_CENTROIDS.get(req.region, {}).get("Latitude", 0.0)
    lon = req.longitude if req.longitude is not None else REGION_CENTROIDS.get(req.region, {}).get("Longitude", 0.0)
    exp = compute_exposure(lat, lon)
    row = pd.DataFrame([{
        "Magnitude": req.magnitude,
        "Depth_km": req.depth_km,
        "MMI_max": req.mmi_max,
        "dist_nearest_settlement_km": exp["dist_nearest_settlement_km"],
        "n_settlements_50km": exp["n_settlements_50km"],
        "n_settlements_100km": exp["n_settlements_100km"],
        "log_pop_100km": exp["log_pop_100km"],
        "Tsunami": "Tsunami" if req.tsunami else "Tidak Tsunami",
        "Region": req.region,
    }])
    X = _prep(row)
    proba = FINAL_MODEL.predict_proba(PRE.transform(X))[0]
    pred = int(np.argmax(proba))
    return {
        "class": pred,
        "label": CLASS_NAMES[pred],
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "exposure": exp,
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
