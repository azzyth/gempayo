"""Ekstrak GeoNames ID.zip -> semua permukiman Indonesia (feature_class P)."""
import os
import zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

COLUMNS = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
           "feature_class", "feature_code", "country_code", "cc2", "admin1", "admin2",
           "admin3", "admin4", "population", "elevation", "dem", "timezone", "mod_date"]
with zipfile.ZipFile(os.path.join(DATA, "ID.zip")) as z:
    with z.open("ID.txt") as f:
        places = pd.read_csv(f, sep="\t", header=None, names=COLUMNS, low_memory=False)

# semua permukiman (kota/desa)
places = places[places["feature_class"] == "P"].copy()
places = places[["name", "latitude", "longitude", "population"]]
places.to_csv(os.path.join(DATA, "id_places.csv"), index=False)

n_pop = int((places["population"] > 0).sum())
print(f"permukiman Indonesia (P): {len(places)} total")
print(f"dengan data populasi    : {n_pop}")
print(f"populasi: median {places['population'].median():.0f}, max {places['population'].max():.0f}")
