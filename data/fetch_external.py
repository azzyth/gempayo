"""Unduh data eksternal untuk fitur exposure ("seberapa besar") & laju gempa ("kapan").

1. USGS earthquake catalog (M>=4.5, region Indonesia, 2000-2023) -> data/usgs_earthquakes.csv
2. GeoNames cities1000 (populasi >1000), filter Indonesia -> data/id_cities.csv
"""
import os
import io
import zipfile
import urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)


def download(url, dest):
    print(f"download {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


# ---- 1. USGS catalog ----
usgs_url = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
    "&starttime=2000-01-01&endtime=2023-12-31"
    "&minmagnitude=5.0&minlatitude=-11&maxlatitude=6"
    "&minlongitude=95&maxlongitude=141&orderby=time"
)
usgs_csv = os.path.join(DATA, "usgs_earthquakes.csv")
download(usgs_url, usgs_csv)

# ---- 2. GeoNames cities1000 ----
geo_zip = os.path.join(DATA, "cities1000.zip")
download("https://download.geonames.org/export/dump/cities1000.zip", geo_zip)

COLUMNS = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
           "feature_class", "feature_code", "country_code", "cc2", "admin1", "admin2",
           "admin3", "admin4", "population", "elevation", "dem", "timezone", "mod_date"]
with zipfile.ZipFile(geo_zip) as z:
    with z.open("cities1000.txt") as f:
        cities = pd.read_csv(f, sep="\t", header=None, names=COLUMNS, low_memory=False)

id_cities = cities[cities["country_code"] == "ID"].copy()
id_cities = id_cities[["name", "latitude", "longitude", "population", "admin1"]]
id_cities = id_cities[id_cities["population"] > 0]
id_cities.to_csv(os.path.join(DATA, "id_cities.csv"), index=False)

usgs = pd.read_csv(usgs_csv)
print("\n== RINGKASAN ==")
print(f"USGS events : {len(usgs)} baris, kolom: {list(usgs.columns)[:6]}...")
print(f"ID cities   : {len(id_cities)} kota/desa (populasi >0)")
print(f"populasi kota: median {id_cities['population'].median():.0f}, max {id_cities['population'].max():.0f}")
