
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import csv


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
USE_REAL_BORDERS = True      # True = try to download real state polygons (needs internet)
GRID_NX, GRID_NY = 260, 160  # resolution of the terrain grid
SMOOTH_SIGMA = 2.6           # gaussian blur strength (higher = smoother/blurrier)
SHOW_LABELS = False # draw state abbreviation labels on the terrain
OUTPUT_DPI = 180
GAUSS_SMOOTH = False # apply gaussian smoothing to the terrain
COUNTY_SHAPES = {}


# Continental US grid bounds (deg)
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.0, 49.5



#--------------------
# Load data
# ---

def load_county_shapes():
    """Load contiguous U.S. county polygons.

    Returns:
        dict:
            {
                fips: [
                    [(lon, lat), ...],
                    [(lon, lat), ...],
                    ...
                ]
            }

    Alaska, Hawaii, and territories are excluded.
    """

    import geopandas as gpd

    url = (
        "zip+https://www2.census.gov/geo/tiger/"
        "GENZ2025/shp/cb_2025_us_county_500k.zip"
    )

    counties = gpd.read_file(url)

    # Exclude Alaska, Hawaii, and territories.
    exclude = {
        "02",  # Alaska
        "15",  # Hawaii
        "60",  # American Samoa
        "66",  # Guam
        "69",  # Northern Mariana Islands
        "72",  # Puerto Rico
        "78",  # U.S. Virgin Islands
    }

    counties = counties[
        ~counties["STATEFP"].isin(exclude)
    ].copy()

    global COUNTY_SHAPES
    COUNTY_SHAPES = {}

    for _, row in counties.iterrows():

        fips = row["GEOID"]
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        polygons = []

        # ---------------------------------------------------------------
        # Polygon
        # ---------------------------------------------------------------
        if geom.geom_type == "Polygon":

            exterior = [
                (x, y)
                for x, y in geom.exterior.coords
            ]

            polygons.append(exterior)

        # ---------------------------------------------------------------
        # MultiPolygon
        # ---------------------------------------------------------------
        elif geom.geom_type == "MultiPolygon":

            for polygon in geom.geoms:

                exterior = [
                    (x, y)
                    for x, y in polygon.exterior.coords
                ]
                # Ensure the polygon is closed
                if exterior[0] != exterior[-1]:
                    exterior.append(exterior[0])

                polygons.append(exterior)

        COUNTY_SHAPES[fips] = polygons

    return COUNTY_SHAPES

def load_county_temperature_records(filename):
    """Load county temperature data and find all-time records.

    Returns:
        RECORD_HIGH:
            {
                fips: (highest_tmax, region_name, date),
                ...
            }

        RECORD_LOW:
            {
                fips: (lowest_tmin, region_name, date),
                ...
            }
    """

    # Temporary storage while determining records.
    high = {}
    low = {}

    with open(filename, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Only use county records.
            if row["region_type"].strip() != "cty":
                continue

            fips = row["fips"].strip()
            date = row["date"].strip()
            region_name = row["region_name"].strip()

            # Convert temperature strings to floats.
            tmax = float(row["tmax"].strip())
            tmin = float(row["tmin"].strip())

            # ---------------------------------------------------------------
            # Record high
            # ---------------------------------------------------------------
            if (
                fips not in high
                or tmax > high[fips][0]
            ):
                high[fips] = (
                    tmax,
                    region_name,
                    date,
                )

            # ---------------------------------------------------------------
            # Record low
            # ---------------------------------------------------------------
            if (
                fips not in low
                or tmin < low[fips][0]
            ):
                low[fips] = (
                    tmin,
                    region_name,
                    date,
                )

    return high, low


# ---------------------------------------------------------------------------
# TERRAIN BUILDING — COUNTY POLYGON BASED
# ---------------------------------------------------------------------------

def build_voronoi_terrain(values):
    """Nearest-centroid terrain grid + gaussian smoothing + convex-hull mask.

    `values` is a dict of {state_abbr: numeric_value} for the continental states.
    Returns (LON grid, LAT grid, Z grid [masked outside the US outline]).
    """
    lons = np.linspace(LON_MIN, LON_MAX, GRID_NX)
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_NY)
    LON, LAT = np.meshgrid(lons, lats)

    abbrs = CONTINENTAL_STATES
    centroid_lat = np.array([CENTROIDS[a][0] for a in abbrs])
    centroid_lon = np.array([CENTROIDS[a][1] for a in abbrs])
    state_vals = np.array([values[a] for a in abbrs], dtype=float)

    # correct longitude distances for the fact degrees-of-longitude shrink
    # towards the poles, so cells aren't east-west stretched
    lat_scale_ref = np.cos(np.deg2rad(np.mean(centroid_lat)))

    # nearest-centroid assignment, vectorized: for every grid point compute
    # squared distance to every state center and take the argmin
    dlat = LAT[..., None] - centroid_lat[None, None, :]
    dlon = (LON[..., None] - centroid_lon[None, None, :]) * lat_scale_ref
    dist2 = dlat ** 2 + dlon ** 2
    nearest = np.argmin(dist2, axis=-1)
    Z = state_vals[nearest]

    # smooth into continuous, rolling terrain
    if GAUSS_SMOOTH:
        Z = gaussian_filter(Z, sigma=SMOOTH_SIGMA)

    # mask to the convex hull of the state centers so the plot reads as a
    # US-shaped landmass rather than a rectangle. Hull is padded outward
    # slightly so coastal states keep their full extent.
    pts = np.column_stack([centroid_lon, centroid_lat])
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    center = hull_pts.mean(axis=0)
    hull_pts = center + (hull_pts - center) * 1.35  # pad outward ~35%
    hull_path = Path(hull_pts)

    grid_pts = np.column_stack([LON.ravel(), LAT.ravel()])
    inside = hull_path.contains_points(grid_pts).reshape(LON.shape)
    Z = np.where(inside, Z, np.nan)

    return LON, LAT, Z



def build_county_terrain(values):
    """
    Rasterize county-level values onto the geographic grid.

    values:
        {fips: numeric_value}
    """

    LON = np.linspace(LON_MIN, LON_MAX, GRID_NX)
    LAT = np.linspace(LAT_MIN, LAT_MAX, GRID_NY)

    lon_grid, lat_grid = np.meshgrid(LON, LAT)

    Z = np.full(lon_grid.shape, np.nan, dtype=float)

    matched = 0

    print(f"Processing {len(COUNTY_SHAPES)} counties.")
    for fips, polygons in COUNTY_SHAPES.items():
        if fips not in values:
            continue


        value = float(values[fips])
        matched += 1
        if matched % 100 == 0:
            print(f"Processed {matched} counties.")

        for polygon in polygons:
            exterior = np.asarray(polygon)

            path = Path(exterior)

            points = np.column_stack([
                lon_grid.ravel(),
                lat_grid.ravel(),
            ])

            inside = path.contains_points(points)

            Z.ravel()[inside] = value
        

    if matched == 0:
        raise ValueError(
            "No counties matched the supplied FIPS values."
        )

    # Smooth the county field
    valid = np.isfinite(Z)

    if not np.any(valid):
        raise ValueError("County raster contains no valid cells.")

    # Fill NaNs temporarily before Gaussian filtering
    Z_filled = np.where(valid, Z, np.nanmean(Z))

    if GAUSS_SMOOTH:
        Z_smooth = gaussian_filter(
            Z_filled,
            sigma=SMOOTH_SIGMA,
        )
    else:
        Z_smooth = Z_filled

    # Restore geographic mask
    Z_smooth[~valid] = np.nan

    return lon_grid, lat_grid, Z_smooth


def build_county_terrain_old(values):
    """Build terrain from actual U.S. county polygons.

    Each county is assigned the value of its parent state. The resulting
    county-value raster is then Gaussian-smoothed to produce continuous
    terrain.

    Alaska, Hawaii, and U.S. territories are excluded.

    Returns:
        (LON, LAT, Z)
    """

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.prepared import prep
    except ImportError:
        print(
            "geopandas/shapely not installed - using the built-in "
            "Voronoi terrain instead. (pip install geopandas shapely "
            "for county-based terrain.)"
        )
        return None

    # -----------------------------------------------------------------------
    # Load Census county boundaries
    # -----------------------------------------------------------------------
    url = (
        "zip+https://www2.census.gov/geo/tiger/GENZ2018/shp/"
        "cb_2018_us_county_20m.zip"
    )

    try:
        counties = gpd.read_file(url)
    except Exception as exc:
        print(
            f"Could not download/read Census county boundaries ({exc}); "
            "using the built-in Voronoi terrain instead."
        )
        return None

    try:
        # -------------------------------------------------------------------
        # State FIPS -> abbreviation
        # -------------------------------------------------------------------
        state_fips = {
            "01": "AL",
            "04": "AZ",
            "05": "AR",
            "06": "CA",
            "08": "CO",
            "09": "CT",
            "10": "DE",
            "12": "FL",
            "13": "GA",
            "16": "ID",
            "17": "IL",
            "18": "IN",
            "19": "IA",
            "20": "KS",
            "21": "KY",
            "22": "LA",
            "23": "ME",
            "24": "MD",
            "25": "MA",
            "26": "MI",
            "27": "MN",
            "28": "MS",
            "29": "MO",
            "30": "MT",
            "31": "NE",
            "32": "NV",
            "33": "NH",
            "34": "NJ",
            "35": "NM",
            "36": "NY",
            "37": "NC",
            "38": "ND",
            "39": "OH",
            "40": "OK",
            "41": "OR",
            "42": "PA",
            "44": "RI",
            "45": "SC",
            "46": "SD",
            "47": "TN",
            "48": "TX",
            "49": "UT",
            "50": "VT",
            "51": "VA",
            "53": "WA",
            "54": "WV",
            "55": "WI",
            "56": "WY",
        }

        counties["STUSPS"] = counties["STATEFP"].map(state_fips)

        # Keep only states for which we have data.
        counties = counties[
            counties["STUSPS"].isin(values.keys())
        ].copy()

        if counties.empty:
            raise ValueError("No counties matched the supplied state values.")

        # -------------------------------------------------------------------
        # Create plotting grid
        # -------------------------------------------------------------------
        lons = np.linspace(LON_MIN, LON_MAX, GRID_NX)
        lats = np.linspace(LAT_MIN, LAT_MAX, GRID_NY)
        LON, LAT = np.meshgrid(lons, lats)

        flat_lon = LON.ravel()
        flat_lat = LAT.ravel()

        # -------------------------------------------------------------------
        # Start with NaN everywhere.
        #
        # We will rasterize the ACTUAL COUNTY POLYGONS into this array.
        # -------------------------------------------------------------------
        result = np.full(
            flat_lon.shape,
            np.nan,
            dtype=float
        )

        # -------------------------------------------------------------------
        # Rasterize each county.
        #
        # Every point physically inside a county receives the value of the
        # county's parent state.
        # -------------------------------------------------------------------
        for _, row in counties.iterrows():

            abbr = row["STUSPS"]
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            value = float(values[abbr])

            minx, miny, maxx, maxy = geom.bounds

            # Only test grid points within the county's bounding box.
            candidates = np.where(
                (flat_lon >= minx)
                & (flat_lon <= maxx)
                & (flat_lat >= miny)
                & (flat_lat <= maxy)
            )[0]

            if len(candidates) == 0:
                continue

            prepared = prep(geom)

            for i in candidates:
                if prepared.contains(
                    Point(flat_lon[i], flat_lat[i])
                ):
                    result[i] = value

        Z = result.reshape(LON.shape)

        # -------------------------------------------------------------------
        # Actual contiguous-U.S. mask
        #
        # The county polygons themselves define the land boundary.
        # -------------------------------------------------------------------
        us_geometry = counties.geometry.union_all()

        prepared_us = prep(us_geometry)

        inside = np.array([
            prepared_us.contains(Point(lon, lat))
            for lon, lat in zip(flat_lon, flat_lat)
        ])

        inside = inside.reshape(LON.shape)

        # -------------------------------------------------------------------
        # Gaussian smoothing
        #
        # IMPORTANT:
        #
        # gaussian_filter cannot handle NaNs correctly. Fill the outside
        # region temporarily before smoothing, then restore the mask.
        # -------------------------------------------------------------------
        valid_mask = ~np.isnan(Z)

        if not np.any(valid_mask):
            raise ValueError("County rasterization produced no valid points.")

        fill_value = np.nanmean(Z)

        filled = np.where(
            valid_mask,
            Z,
            fill_value
        )

        if GAUSS_SMOOTH:
            Z = gaussian_filter(
                filled,
                sigma=SMOOTH_SIGMA
            )
        else:
            Z = filled

        # Restore the actual U.S. boundary.
        Z[~inside] = np.nan

        return LON, LAT, Z

    except Exception as exc:
        print(
            f"County terrain construction failed ({exc}); "
            "using the built-in Voronoi terrain instead."
        )
        return None


def get_terrain(values):
    """Dispatcher for terrain construction."""

    county = build_county_terrain(values)

    if county is not None:
        return county

    return build_voronoi_terrain(values)




def plot_relief(values, title, cmap_name, label_suffix, filename):
    """values: dict {abbr: (temp, place, year)} for RECORD_HIGH or RECORD_LOW."""
    numeric = {
        fips: v[0]
        for fips, v in values.items()
    }

    LON, LAT, Z = get_terrain(numeric)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    all_vals = np.array(list(numeric.values()))
    vmin, vmax = all_vals.min(), all_vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    surf = ax.plot_surface(
        LON, LAT, Z, cmap=cmap, norm=norm,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
        shade=True, vmin=vmin, vmax=vmax,
    )

    # state labels at their true centroid, height-matched to the smoothed
    # surface immediately below them
    label_outline = [pe.withStroke(linewidth=2.2, foreground="white")]
    if SHOW_LABELS:
        for abbr in CONTINENTAL_STATES:
            lat0, lon0 = CENTROIDS[abbr][0], CENTROIDS[abbr][1]
            z0 = numeric[abbr]
            ax.text(lon0, lat0, z0 + (vmax - vmin) * 0.02, abbr,
                     fontsize=6.5, ha="center", va="bottom",
                     color="black", zorder=10, path_effects=label_outline)


    ax.set_title(title, fontsize=16, weight="bold", pad=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel(f"Temperature (\u00b0F, {label_suffix})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=42, azim=-100)
    ax.set_box_aspect((LON_MAX - LON_MIN, LAT_MAX - LAT_MIN, (vmax - vmin) * 0.9))

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(f"Record {label_suffix} (\u00b0F)", fontsize=10)

    fig.text(
        0.5,
        0.02,
        "Data: NOAA National Climatic Data Center / State Climate "
        "Extremes Committee, statewide records. Terrain rasterized from "
        "U.S. county boundaries.",
        ha="center",
        fontsize=8,
        color="gray"
    )

    # fig.tight_layout() can raise a singular-matrix LinAlgError on 3D axes
    # for certain view angles (a known matplotlib mplot3d limitation) -
    # bbox_inches="tight" on savefig achieves the same trimming safely.
    try:
        fig.tight_layout()
    except np.linalg.LinAlgError:
        pass
    fig.savefig(filename, dpi=OUTPUT_DPI, facecolor="white", bbox_inches="tight")

    # also pickle the live Figure so it can be reloaded later and still be
    # rotated/zoomed interactively (a saved .png is a flat, static image)
    pickle_path = os.path.splitext(filename)[0] + ".fig.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)

    plt.show(block=True)
    #plt.close(fig)
    print(f"Saved {filename} (static) and {pickle_path} (reopen with load_figure.py to rotate)")


def main():

    COUNTY_SHAPES = load_county_shapes()
    print(f"Loaded {len(COUNTY_SHAPES)} county shapes.")

    RECORD_HIGH, RECORD_LOW = load_county_temperature_records(
    "noaa_county_cache/195102-scaled.csv"
    )
    print(f"Loaded {len(RECORD_HIGH)} record high entries.")

    plot_relief(
        RECORD_HIGH,
        title="U.S. Record-Heat Relief Map\n(peak height = all-time record high temperature)",
        cmap_name="inferno",
        label_suffix="High",
        filename="us_record_high_peaks.png",
    )
    plot_relief(
        RECORD_LOW,
        title="U.S. Record-Cold Relief Map\n(valley depth = all-time record low temperature)",
        cmap_name="Blues_r",
        label_suffix="Low",
        filename="us_record_low_valleys.png",
    )

    #"AL": (112, "Centerville", 1925), "AK": (100, "Fort Yukon", 1915),
    RECORD_RANGE = {abbr: (RECORD_HIGH[abbr][0] - RECORD_LOW[abbr][0], "no location", 0) for abbr in RECORD_HIGH}

    plot_relief(
        RECORD_RANGE,
        title="U.S. Record-Range Relief Map\n(range = all-time record high minus low temperature)",
        cmap_name="viridis",
        label_suffix="Range",
        filename="us_record_range.png",
    )


if __name__ == "__main__":
    main()
