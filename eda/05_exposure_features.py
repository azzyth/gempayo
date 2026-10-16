"""TAHAP 4 - FITUR EXPOSURE (GeoNames, semua permukiman Indonesia).

Fitur exposure dari 258k permukiman (BallTree, haversine):
  - dist_nearest_settlement_km : jarak ke permukiman terdekat
  - n_settlements_{50,100}km   : jumlah permukiman dalam radius (proxy kepadatan)
  - pop_within_100km           : populasi kota (540 dgn data populasi) dalam 100km
Output: gempabumi_exposure.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

df = pd.read_csv(os.path.join(ROOT, "gempabumi_clean.csv"), encoding="utf-8-sig")
places = pd.read_csv(os.path.join(DATA, "id_places.csv"))
cities = pd.read_csv(os.path.join(DATA, "id_cities.csv"))

EQ = np.radians(df[["Latitude", "Longitude"]].to_numpy())
R = 6371.0

tree = BallTree(np.radians(places[["latitude", "longitude"]].to_numpy()), metric="haversine")

# jarak ke permukiman terdekat
dist, _ = tree.query(EQ, k=1)
df["dist_nearest_settlement_km"] = (dist[:, 0] * R).round(2)

# jumlah permukiman dalam radius (kepadatan)
for radius in (50, 100):
    df[f"n_settlements_{radius}km"] = tree.query_radius(EQ, r=radius / R, count_only=True)

# populasi kota dalam 100km
city_rad = np.radians(cities[["latitude", "longitude"]].to_numpy())
tree_c = BallTree(city_rad, metric="haversine")
idx = tree_c.query_radius(EQ, r=100 / R)
pop = cities["population"].to_numpy()
df["pop_within_100km"] = [int(pop[i].sum()) for i in idx]

df["log_n_settlements_50km"] = np.log1p(df["n_settlements_50km"])
df["log_n_settlements_100km"] = np.log1p(df["n_settlements_100km"])
df["log_pop_100km"] = np.log1p(df["pop_within_100km"])

df.to_csv(os.path.join(ROOT, "gempabumi_exposure.csv"), index=False, encoding="utf-8-sig")

print("== FITUR EXPOSURE (258k permukiman) ==")
print(df[["dist_nearest_settlement_km", "n_settlements_50km", "n_settlements_100km",
          "pop_within_100km"]].describe().round(1).to_string())

sev = pd.read_csv(os.path.join(ROOT, "gempabumi_model.csv"), encoding="utf-8-sig")["Severity"]
g = df.assign(Severity=sev.values).groupby("Severity")[
    ["dist_nearest_settlement_km", "n_settlements_50km", "log_pop_100km"]].agg(["median", "mean"]).round(1)
print("\nexposure per kelas severity (median | mean):")
print(g.to_string())
