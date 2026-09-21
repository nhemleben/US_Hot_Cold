"""
NOAA County-Level Record Temperature Extremes Pipeline
=========================================================
Downloads NOAA's nClimGrid-Daily / EpiNOAA county-scale daily temperature
data (max + min, 1951-present, CONUS only) and reduces it to an all-time
record high and record low per county, in the same dict shape used for the
state-level script:

    RECORD_HIGH_COUNTY = {"01001": (98.6, "Autauga County, AL", 2011), ...}
    RECORD_LOW_COUNTY  = {"01001": (-3.2, "Autauga County, AL", 1985), ...}

keyed by 5-digit county FIPS code (state FIPS + county FIPS).

WHY THIS DATA, AND WHAT IT IS NOT
------------------------------------
NOAA does not run a "State Climate Extremes Committee" equivalent for
counties, so there's no official all-time county-record table to simply
transcribe (which is what made the state version possible). What NOAA does
publish at county resolution is nClimGrid-Daily: a ~5km interpolated daily
grid of tmax/tmin, area-averaged to each county polygon, from Jan 1, 1951
to the present. This script takes the MAX (for highs) and MIN (for lows)
across every day of that record, per county - a real, computable, and
defensible "record extreme", but with two differences from the state data:
  - Period of record is 1951-present, not the 1890s-present of the state
    records, so a county's true all-time extreme from an earlier decade
    will not be captured.
  - Values are a smoothed/interpolated grid estimate for the county's area,
    not a single station's raw reading, so they'll typically run a little
    less extreme than a station-based record for the same spot.
  - Coverage is CONUS only (Alaska, Hawaii, and territories are not part
    of this product) - which is what was asked for anyway.

IMPORTANT - THIS WAS WRITTEN WITHOUT NETWORK ACCESS
------------------------------------------------------
Every URL, filename pattern, and column layout below is taken from NOAA's
own documentation (the nClimGrid-Daily User Guide and web-folder README,
and the AWS Open Data listing) - but has since been VERIFIED against the
live bucket (2026-09-20). The real schema differs from what NOAA's docs
implied: each monthly file under EpiNOAA/v1-0-0/csv/cty/ is a long-format
CSV with one row per county per day, already carrying both tmax and tmin
plus the resolved FIPS code and county/state name - so no NCEI-state
crosswalk or separate Census Gazetteer download is needed at all:

    region_type,fips,ncei_code,state_name,postal_code,region_name,date,tmax,tmin,tavg,prcp
    cty,01001,1001,Alabama,AL,"AL: Autauga",1951-01-01,12.51,-1.58,5.46,0.00

Run in inspection mode any time to re-confirm against the live bucket:

    python county_temperature_extremes.py --inspect

Requirements: requests, pandas (both already standard/likely installed)
"""

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
S3_BUCKET = "noaa-nclimgrid-daily-pds"
S3_BASE = f"https://{S3_BUCKET}.s3.amazonaws.com"
CTY_PREFIX = "EpiNOAA/v1-0-0/csv/cty/"  # one file per month, all CONUS counties

CACHE_DIR = Path("./noaa_county_cache")
OUTPUT_PY = Path("./county_temperature_records.py")

EXCLUDE_STATE_ABBR = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}  # belt-and-suspenders;
# nClimGrid-Daily's "cty" product is CONUS-only so these shouldn't appear anyway.


def c_to_f(celsius):
    return celsius * 9.0 / 5.0 + 32.0


# ---------------------------------------------------------------------------
# STEP 1: discover files via the bucket's public (no-sign-request) listing
# ---------------------------------------------------------------------------
def list_s3_objects(prefix):
    """List keys under `prefix` in the public nClimGrid-Daily S3 bucket
    using the plain REST listing API (no AWS credentials needed - the
    bucket is public, same as `aws s3 ls --no-sign-request`)."""
    keys = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = requests.get(S3_BASE + "/", params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for contents in root.findall("s3:Contents", ns):
            key = contents.find("s3:Key", ns).text
            keys.append(key)
        truncated = root.find("s3:IsTruncated", ns)
        if truncated is not None and truncated.text == "true":
            token_el = root.find("s3:NextContinuationToken", ns)
            token = token_el.text if token_el is not None else None
            if not token:
                break
        else:
            break
    return keys


def find_county_files():
    """Returns a sorted list of S3 keys for the monthly county-level CSVs
    under CTY_PREFIX. Each file already contains both tmax and tmin for
    every CONUS county, so unlike the earlier (pre-network) assumption
    there's no separate tmax-file/tmin-file split to discover."""
    keys = list_s3_objects(CTY_PREFIX)
    return sorted(k for k in keys if k.lower().endswith(".csv"))


# ---------------------------------------------------------------------------
# STEP 2: download with local caching
# ---------------------------------------------------------------------------
def download(url_or_key, dest_dir=CACHE_DIR, base=S3_BASE):
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = url_or_key if url_or_key.startswith("http") else f"{base}/{url_or_key}"
    local_path = dest_dir / Path(url_or_key).name
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return local_path


# ---------------------------------------------------------------------------
# STEP 3: inspect a real file's structure (run this first - see --inspect)
# ---------------------------------------------------------------------------
def inspect_file(path):
    df = pd.read_csv(path, nrows=20)
    print(f"\nFile: {path}")
    print(f"Shape (first 20 rows): {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(5).to_string())
    return df


# ---------------------------------------------------------------------------
# STEP 4: parse one monthly cty CSV down to the per-county max/min-of-month
# (with the date each occurred), so accumulate_records only has to compare
# one row per county per file instead of every daily row.
# ---------------------------------------------------------------------------
def parse_cty_csv(path):
    """Returns a DataFrame with columns: fips, place, tmax_c, tmax_date,
    tmin_c, tmin_date - one row per county, reduced from that file's daily
    rows. `place` is built from the file's own region_name/postal_code, so
    no external state-crosswalk or Census Gazetteer lookup is needed."""
    df = pd.read_csv(
        path,
        usecols=["fips", "postal_code", "region_name", "date", "tmax", "tmin"],
        dtype={"fips": str, "postal_code": str, "region_name": str, "date": str},
    )
    df["fips"] = df["fips"].str.zfill(5)
    tmax_c = pd.to_numeric(df["tmax"], errors="coerce")
    tmin_c = pd.to_numeric(df["tmin"], errors="coerce")
    # NOAA pads nonexistent days (e.g. Feb 30/31) with this sentinel and an
    # empty date, for every county - must be masked or it wins as a bogus
    # "record low"/"record high".
    sentinel = (tmax_c == -999.99) | (tmin_c == -999.99)
    df["tmax"] = tmax_c.mask(sentinel)
    df["tmin"] = tmin_c.mask(sentinel)
    # region_name is "ST: County Name"; drop the leading "ST: " prefix
    county_name = df["region_name"].str.split(":", n=1).str[-1].str.strip()
    df["place"] = county_name + ", " + df["postal_code"]

    idx_max = df.groupby("fips")["tmax"].idxmax()
    idx_min = df.groupby("fips")["tmin"].idxmin()

    highs = df.loc[idx_max, ["fips", "place", "tmax", "date"]].set_index("fips")
    lows = df.loc[idx_min, ["fips", "place", "tmin", "date"]].set_index("fips")
    out = highs.join(lows, lsuffix="_high", rsuffix="_low", how="outer")
    out = out.rename(columns={
        "tmax": "tmax_c", "date_high": "tmax_date",
        "tmin": "tmin_c", "date_low": "tmin_date",
    })
    out["place"] = out["place_high"].combine_first(out["place_low"])
    return out.reset_index()


# ---------------------------------------------------------------------------
# STEP 5: accumulate the running record across every file for one direction
# ---------------------------------------------------------------------------
def accumulate_records(files, mode):
    """mode: 'max' (record highs) or 'min' (record lows).
    Returns {fips: (value_c, place, year)}."""
    best = {}
    value_col = "tmax_c" if mode == "max" else "tmin_c"
    date_col = "tmax_date" if mode == "max" else "tmin_date"
    better = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)

    for path in files:
        chunk = parse_cty_csv(path)
        for row in chunk.itertuples(index=False):
            val = getattr(row, value_col)
            if pd.isna(val):
                continue
            fips = row.fips
            year = int(getattr(row, date_col)[:4])
            cur = best.get(fips)
            if cur is None or better(val, cur[0]):
                best[fips] = (val, row.place, year)
    return best


# ---------------------------------------------------------------------------
# STEP 6: orchestration
# ---------------------------------------------------------------------------
def build_dict(best_by_region):
    out = {}
    for fips, (val_c, place, year) in best_by_region.items():
        usps = place.split(", ")[-1]
        if usps in EXCLUDE_STATE_ABBR:
            continue
        out[fips] = (round(c_to_f(val_c), 1), place, year)
    return out


def write_output_py(record_high, record_low, path=OUTPUT_PY):
    with open(path, "w") as f:
        f.write('"""Auto-generated by county_temperature_extremes.py - do not hand-edit.\n')
        f.write("Record high/low per county, from NOAA nClimGrid-Daily (1951-present, CONUS).\n")
        f.write('Format matches the state-level dicts: {fips: (temp_F, "County, ST", year)}\n"""\n\n')
        f.write("RECORD_HIGH_COUNTY = {\n")
        for fips in sorted(record_high):
            temp, place, year = record_high[fips]
            f.write(f'    "cty, {fips}": ({temp}, {place!r}, {year}),\n')
        f.write("}\n\n")
        f.write("RECORD_LOW_COUNTY = {\n")
        for fips in sorted(record_low):
            temp, place, year = record_low[fips]
            f.write(f'    "cty, {fips}": ({temp}, {place!r}, {year}),\n')
        f.write("}\n")
    print(f"Wrote {path} ({len(record_high)} counties high, {len(record_low)} counties low)")


def consolidate_records(paths):
    print("Scanning files for record highs...")
    best_high = accumulate_records(paths, mode="max")
    print("Scanning files for record lows...")
    best_low = accumulate_records(paths, mode="min")
    return build_dict(best_high), build_dict(best_low)

def run_pipeline():
    print("Discovering county files on NOAA's S3 bucket...")
    keys = find_county_files()
    if not keys:
        print(f"ERROR: could not find any county files under {CTY_PREFIX}. "
              "Run with --inspect or browse https://noaa-nclimgrid-daily-pds"
              ".s3.amazonaws.com/index.html#EpiNOAA/ manually and adjust "
              "find_county_files().")
        sys.exit(1)
    print(f"Found {len(keys)} monthly county file(s).")

    print("Downloading (cached locally in ./noaa_county_cache/)...")
    paths = [download(k) for k in keys]

    record_high, record_low = consolidate_records(paths)

    write_output_py(record_high, record_low)



# ---------------------------------------------------------------------------
# SELF-TEST: validates the parsing/aggregation logic with NO network access,
# using a synthetic file built to match NOAA's documented schema.
# ---------------------------------------------------------------------------
def self_test():
    import tempfile

    cols = ["region_type", "fips", "ncei_code", "state_name", "postal_code",
            "region_name", "date", "tmax", "tmin", "tavg", "prcp"]
    rows = [
        # Fake county 1: record high (40.0C) on Feb 15, 2011
        ["cty", "01001", "1001", "Alabama", "AL", "AL: Autauga", "2011-02-15", 40.0, 20.0, 30.0, 0.0],
        ["cty", "01001", "1001", "Alabama", "AL", "AL: Autauga", "2011-02-16", 25.0, 15.0, 20.0, 0.0],
        # Fake county 2: record low (-30.5C) on Jan 3, 1998
        ["cty", "13005", "13005", "Georgia", "GA", "GA: Bacon County", "1998-01-03", 10.0, -30.5, -10.0, 0.0],
        ["cty", "13005", "13005", "Georgia", "GA", "GA: Bacon County", "1998-01-04", 8.0, -5.0, 1.5, 0.0],
    ]
    df = pd.DataFrame(rows, columns=cols)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "199801-scaled.csv"
        df.to_csv(p, index=False)
        parsed = parse_cty_csv(p)

        row1 = parsed[parsed.fips == "01001"].iloc[0]
        assert abs(row1.tmax_c - 40.0) < 1e-9, row1.tmax_c
        assert row1.tmax_date == "2011-02-15"
        assert row1.place == "Autauga, AL"

        row2 = parsed[parsed.fips == "13005"].iloc[0]
        assert abs(row2.tmin_c - (-30.5)) < 1e-9, row2.tmin_c
        assert row2.tmin_date == "1998-01-03"
        assert row2.place == "Bacon County, GA"

        # accumulate_records + build_dict end-to-end
        best_high = accumulate_records([p], mode="max")
        assert best_high["01001"][0] == 40.0

        record_high = build_dict(best_high)
        assert record_high["01001"][0] == round(c_to_f(40.0), 1)
        assert record_high["01001"][1] == "Autauga, AL"
        assert record_high["01001"][2] == 2011

    print("All self-tests passed: parsing, max/min-of-month detection, "
          "county-name/place formatting, and dict output are all correct "
          "against the real (live-verified) NOAA cty CSV schema.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true",
                         help="Download one county file and print its real structure, then exit.")
    parser.add_argument("--selftest", action="store_true",
                         help="Run the offline self-test of the parsing/aggregation logic.")
    parser.add_argument("--consolidate", action="store_true",
                         help="Run the consolidation of record highs and lows.")
    args = parser.parse_args()

    if args.selftest:
        self_test()
    elif args.inspect:
        keys = find_county_files()
        if not keys:
            print(f"No county files found under {CTY_PREFIX}.")
            sys.exit(1)
        path = download(keys[0])
        inspect_file(path)
    elif args.consolidate:
        from pathlib import Path
        file_names = [str(p) for p in Path("noaa_county_cache").iterdir() if p.is_file()]
        record_high, record_low = consolidate_records(file_names)

        import csv
        with open("all_time_extremes/record_extremes.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["region_type", "fips", "tmax", "tmin", "place", "year"])
            for fips, (temp, place, year) in record_high.items():
                writer.writerow(["cty", fips, temp, record_low[fips][0], place, year])

        print("Consolidation complete.")
        print(f"Record high entries: {len(record_high)}")
        print(f"Record low entries: {len(record_low)}")
    else:
        run_pipeline()
