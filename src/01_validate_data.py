"""Step 1: ingest + validate the Census files you downloaded (NO API calls).

How to use
  1. Download the 14 tables listed in README.md from data.census.gov (zip files).
  2. Drop the .zip files (don't unzip) into  data/downloads/
  3. Run this script. It extracts each '-Data.csv', renames it YYYY_TABLE.csv in data/raw/,
     downloads the tract shapefile, and validates everything.
"""
import re, shutil, sys, zipfile, urllib.request
from config import *
from common import read_table, income_columns

NAME_RE = re.compile(r"ACS(?:ST|DP|DT)5Y(\d{4})\.([A-Z]+\d+[A-Z]?)-Data\.csv$", re.I)


def ingest():
    n = 0
    zips = list(DOWNLOADS.rglob("*.zip")) + list(RAW.glob("*.zip"))
    for z in zips:
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            print(f"  skipped (not a valid zip): {z.name}")
            continue
        with zf:
            names = zf.namelist()
            if any(m.lower().endswith(".shp") for m in names):        # TIGER shapefile zip
                zf.extractall(RAW / "tiger_tracts")
                print(f"  extracted shapefile from {z.name}")
                continue
            for m in names:
                hit = NAME_RE.search(m)
                if hit:
                    year, table = int(hit.group(1)), hit.group(2).upper()
                    with zf.open(m) as src, open(RAW / f"{year}_{table}.csv", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"  {z.name[:45]:45s} -> {year}_{table}.csv")
                    n += 1
    for f in list(DOWNLOADS.rglob("*-Data.csv")) + list(RAW.glob("ACS*-Data.csv")):   # already unzipped
        hit = NAME_RE.search(f.name)
        if hit:
            shutil.copy(f, RAW / f"{int(hit.group(1))}_{hit.group(2).upper()}.csv")
            print(f"  {f.name} -> {hit.group(1)}_{hit.group(2).upper()}.csv")
            n += 1
    return n


def get_tiger():
    d = RAW / "tiger_tracts"
    if list(d.glob("*.shp")):
        return True
    d.mkdir(exist_ok=True)
    z = RAW / "tl_2016_06_tract.zip"
    try:
        print("Downloading TIGER tract shapefile ...")
        urllib.request.urlretrieve(TIGER_URL, z)
        zipfile.ZipFile(z).extractall(d)
        return True
    except Exception as e:
        print(f"  !! could not download shapefile automatically ({e}).\n"
              f"     Download it in your browser: {TIGER_URL}\n"
              f"     then put the zip in {DOWNLOADS} and run this script again.")
        return False


def validate():
    bad, counts = [], {}
    print("\nChecking required files ...")
    for year, table in REQUIRED:
        try:
            df, labels = read_table(year, table)
            counts[(year, table)] = len(df)
            print(f"  OK   {year}_{table:7s} tracts={len(df):5d}  estimate-cols={df.shape[1]:4d}  "
                  f"labels={'yes' if any(labels.values()) else 'no'}")
        except FileNotFoundError:
            bad.append((year, table, "missing"))
            print(f"  MISS {year}_{table}")
        except Exception as e:
            bad.append((year, table, str(e)))
            print(f"  BAD  {year}_{table}: {e}")
    if bad:
        print("\nProblems found. Open these links in your browser, download the table (Download -> CSV), "
              f"and drop the zip in {DOWNLOADS}:")
        for y, t, why in bad:
            print(f"  {y} {t:7s} ({why[:60]})\n      {census_url(y, t)}")
        return False
    # sanity checks on content
    for y in sorted({y for y, _ in REQUIRED}):
        try:
            cols = income_columns(y)
            print(f"  income bins {y}: {cols[0]} ... {cols[-1]}  ({len(cols)} bins)")
        except Exception as e:
            print(f"  !! income bins {y}: {e}"); return False
    n = list(counts.values())
    if min(n) < 6000:
        print("\nWARNING: some files have far fewer than ~8,000 California tracts. "
              "Did you select 'All Census Tracts within California' before downloading?")
    return True


if __name__ == "__main__":
    print("Ingesting downloaded files ...")
    print(f"  {ingest()} data files placed in {RAW}")
    shp_ok = get_tiger()
    ok = validate()
    if ok and shp_ok:
        print("\nAll good. Next: python src/02_build_dataset.py")
    else:
        sys.exit(1)
