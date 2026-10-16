"""Proses data eksternal hasil unduhan -> CSV siap pakai.

- cities1000.zip (GeoNames) -> data/id_cities.csv (kota Indonesia, populasi >0)
- usgs_earthquakes.csv -> data/usgs_earthquakes_clean.csv (time, lat, lon, depth, mag, place)
"""
import os
import zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ---- GeoNames ----
COLUMNS = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
           "feature_class", "feature_code", "country_code", "cc2", "admin1", "admin2",
           "admin3", "admin4", "population", "elevation", "dem", "timezone", "mod_date"]
with zipfile.ZipFile(os.path.join(DATA, "cities1000.zip")) as z:
    with z.open("cities1000.txt") as f:
        cities = pd.read_csv(f, sep="\t", header=None, names=COLUMNS, low_memory=False)

id_cities = cities[cities["country_code"] == "ID"].copy()
id_cities = id_cities[["name", "latitude", "longitude", "population", "admin1"]]
id_cities = id_cities[id_cities["population"] > 0]
id_cities.to_csv(os.path.join(DATA, "id_cities.csv"), index=False)

# ---- USGS ----
usgs = pd.read_csv(os.path.join(DATA, "usgs_earthquakes.csv"))
usgs_clean = usgs[["time", "latitude", "longitude", "depth", "mag", "place"]].copy()
usgs_clean["time"] = pd.to_datetime(usgs_clean["time"], errors="coerce", utc=True)
usgs_clean.to_csv(os.path.join(DATA, "usgs_earthquakes_clean.csv"), index=False)

print("== RINGKASAN ==")
print(f"USGS events  : {len(usgs_clean)} baris")
print(f"  mag range  : {usgs_clean.mag.min():.1f} .. {usgs_clean.mag.max():.1f}")
print(f"  time range : {usgs_clean.time.min()} .. {usgs_clean.time.max()}")
print(f"ID cities    : {len(id_cities)} kota/desa (populasi >0)")
print(f"  populasi   : median {id_cities.population.median():.0f}, max {id_cities.population.max():.0f}")
