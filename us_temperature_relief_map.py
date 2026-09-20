"""
US Temperature Relief Map
==========================
Builds two 3D "topographic style" maps of the continental United States:

  1. PEAKS   - the height of each state is its all-time RECORD HIGH temperature (deg F)
  2. VALLEYS - the depth  of each state is its all-time RECORD LOW  temperature (deg F)

Data source: NOAA National Climatic Data Center / State Climate Extremes
Committee, as compiled by Golden Gate Weather Services
(ggweather.com/climate/extremes_us.htm). Figures are the official statewide
records as of 2024.

HOW THE TERRAIN IS BUILT
-------------------------
A real "drape the temperature over the actual state polygons" map needs a
state boundary shapefile, which means either shipping a multi-megabyte file
or downloading one at runtime. To keep this script a single, dependency-light
file that works completely offline, it instead:

  1. Takes each state's geographic center (lat/lon).
  2. Builds a fine lat/lon grid over the continental US and assigns every
     grid point to its NEAREST state center (a Voronoi-style partition).
     Longitude is corrected by cos(latitude) so the cells aren't stretched.
  3. Gaussian-blurs the result so state-to-state transitions become smooth,
     rolling terrain instead of sharp mosaic tiles - i.e. "topographic style".
  4. Clips the grid to the convex hull of the state centers so it reads as
     a US-shaped landmass rather than a rectangle.

This gives a stylized, recognizable relief map with no internet connection
and no GIS libraries required. Alaska and Hawaii are shown as separate
inset columns (standard US-map cartographic convention) since including
their true coordinates would squash the lower 48 into a corner of the plot.

OPTIONAL: MORE GEOGRAPHICALLY ACCURATE VERSION
------------------------------------------------
If you `pip install geopandas shapely requests` and have internet access,
set USE_REAL_BORDERS = True below. The script will then download the US
Census Bureau's cartographic boundary shapefile and rasterize the *actual*
state polygons instead of the Voronoi approximation, giving a much crisper,
geographically exact coastline and state borders. It falls back to the
Voronoi method automatically if the download or import fails.

Requirements (core):  numpy, scipy, matplotlib
Requirements (optional, for USE_REAL_BORDERS): geopandas, shapely, requests
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
USE_REAL_BORDERS = True      # True = try to download real state polygons (needs internet)
GRID_NX, GRID_NY = 260, 160  # resolution of the terrain grid
SMOOTH_SIGMA = 2.6           # gaussian blur strength (higher = smoother/blurrier)
SHOW_LABELS = True           # draw state abbreviation labels on the terrain
OUTPUT_DPI = 180
GAUSS_SMOOTH = False # apply gaussian smoothing to the terrain

# ---------------------------------------------------------------------------
# DATA: record high / low temperature by state (deg F), NOAA NCDC records
# format: "ABBR": (temperature_F, place, year)
# ---------------------------------------------------------------------------
RECORD_HIGH = {
    "AL": (112, "Centerville", 1925), "AK": (100, "Fort Yukon", 1915),
    "AZ": (128, "Lake Havasu City", 1994), "AR": (120, "Ozark", 1936),
    "CA": (134, "Greenland Ranch (Death Valley)", 1913), "CO": (115, "John Martin Dam", 2019),
    "CT": (106, "Danbury", 1995), "DE": (110, "Millsboro", 1930),
    "FL": (109, "Monticello", 1931), "GA": (112, "Greenville", 1983),
    "HI": (100, "Pahala", 1931), "ID": (118, "Orofino", 1934),
    "IL": (117, "East St. Louis", 1954), "IN": (116, "Collegeville", 1936),
    "IA": (118, "Keokuk", 1934), "KS": (121, "Alton", 1936),
    "KY": (114, "Greensburg", 1930), "LA": (114, "Plain Dealing", 1936),
    "ME": (105, "North Bridgton", 1911), "MD": (109, "Cumberland / Frederick", 1936),
    "MA": (107, "New Bedford / Chester", 1975), "MI": (112, "Mio", 1936),
    "MN": (115, "Beardsley", 1917), "MS": (115, "Holly Springs", 1930),
    "MO": (118, "Warsaw & Union", 1954), "MT": (117, "Medicine Lake", 1937),
    "NE": (118, "Minden", 1936), "NV": (125, "Laughlin", 1994),
    "NH": (106, "Nashua", 1911), "NJ": (110, "Runyon", 1936),
    "NM": (122, "Waste Isolation Pilot Plant", 1994), "NY": (108, "Troy", 1926),
    "NC": (110, "Fayetteville", 1983), "ND": (121, "Steele", 1936),
    "OH": (113, "Gallipolis", 1934), "OK": (120, "Tipton", 1994),
    "OR": (119, "Pendleton", 1898), "PA": (111, "Phoenixville", 1936),
    "RI": (104, "Providence", 1975), "SC": (113, "Columbia", 2012),
    "SD": (120, "Gannvalley", 1936), "TN": (113, "Perryville", 1930),
    "TX": (120, "Seymour", 1936), "UT": (117, "Saint George", 1895),
    "VT": (107, "Vernon", 1912), "VA": (110, "Balcony Falls", 1954),
    "WA": (118, "Ice Harbor Dam", 1961), "WV": (112, "Martinsburg", 1936),
    "WI": (114, "Wisconsin Dells", 1936), "WY": (115, "Basin", 1900),
}

RECORD_LOW = {
    "AL": (-27, "New Market", 1966), "AK": (-80, "Prospect Creek Camp", 1971),
    "AZ": (-40, "Hawley Lake", 1971), "AR": (-29, "Pond", 1905),
    "CA": (-45, "Boca", 1937), "CO": (-61, "Maybell", 1985),
    "CT": (-32, "Falls Village", 1943), "DE": (-17, "Millsboro", 1893),
    "FL": (-2, "Tallahassee", 1899), "GA": (-17, "CCC Camp F-16", 1940),
    "HI": (12, "Mauna Kea", 1979), "ID": (-60, "Island Park Dam", 1943),
    "IL": (-38, "Mt. Carroll", 2019), "IN": (-36, "New Whiteland", 1994),
    "IA": (-47, "Elkader", 1996), "KS": (-40, "Lebanon", 1905),
    "KY": (-37, "Shelbyville", 1994), "LA": (-16, "Minden", 1899),
    "ME": (-50, "Big Black River", 2009), "MD": (-40, "Oakland", 1912),
    "MA": (-35, "Chester", 1981), "MI": (-51, "Vanderbilt", 1934),
    "MN": (-60, "Tower", 1996), "MS": (-19, "Corinth", 1966),
    "MO": (-40, "Warsaw", 1905), "MT": (-70, "Rogers Pass", 1954),
    "NE": (-47, "Camp Clarke", 1899), "NV": (-50, "San Jacinto", 1937),
    "NH": (-50, "Mt. Washington", 1885), "NJ": (-34, "River Vale", 1904),
    "NM": (-50, "Gavilan", 1951), "NY": (-52, "Old Forge", 1979),
    "NC": (-34, "Mt. Mitchell", 1985), "ND": (-60, "Parshall", 1936),
    "OH": (-39, "Milligan", 1899), "OK": (-31, "Nowata", 2011),
    "OR": (-54, "Seneca", 1933), "PA": (-42, "Smethport", 1904),
    "RI": (-28, "Wood River Junction", 1942), "SC": (-19, "Caesars Head", 1985),
    "SD": (-58, "McIntosh", 1936), "TN": (-32, "Mountain City", 1917),
    "TX": (-23, "Seminole", 1933), "UT": (-69, "Peter's Sink", 1985),
    "VT": (-50, "Bloomfield", 1933), "VA": (-30, "Mountain Lake", 1985),
    "WA": (-48, "Mazama / Winthrop", 1968), "WV": (-37, "Lewisburg", 1917),
    "WI": (-55, "Couderay", 1996), "WY": (-66, "Riverside R.S.", 1933),
}

# Geographic center of each state, deg (lat, lon). Source: Google DSPL
# canonical states.csv (public geographic reference points).
CENTROIDS = {
    "AL": (32.318231, -86.902298), "AK": (63.588753, -154.493062),
    "AZ": (34.048928, -111.093731), "AR": (35.201050, -91.831833),
    "CA": (36.778261, -119.417932), "CO": (39.550051, -105.782067),
    "CT": (41.603221, -73.087749), "DE": (38.910832, -75.527670),
    "FL": (27.664827, -81.515754), "GA": (32.157435, -82.907123),
    "HI": (19.898682, -155.665857), "ID": (44.068202, -114.742041),
    "IL": (40.633125, -89.398528), "IN": (40.551217, -85.602364),
    "IA": (41.878003, -93.097702), "KS": (39.011902, -98.484246),
    "KY": (37.839333, -84.270018), "LA": (31.244823, -92.145024),
    "ME": (45.253783, -69.445469), "MD": (39.045755, -76.641271),
    "MA": (42.407211, -71.382437), "MI": (44.314844, -85.602364),
    "MN": (46.729553, -94.685900), "MS": (32.354668, -89.398528),
    "MO": (37.964253, -91.831833), "MT": (46.879682, -110.362566),
    "NE": (41.492537, -99.901813), "NV": (38.802610, -116.419389),
    "NH": (43.193852, -71.572395), "NJ": (40.058324, -74.405661),
    "NM": (34.972730, -105.032363), "NY": (43.299428, -74.217933),
    "NC": (35.759573, -79.019300), "ND": (47.551493, -101.002012),
    "OH": (40.417287, -82.907123), "OK": (35.007752, -97.092877),
    "OR": (43.804133, -120.554201), "PA": (41.203322, -77.194525),
    "RI": (41.580095, -71.477429), "SC": (33.836081, -81.163725),
    "SD": (43.969515, -99.901813), "TN": (35.517491, -86.580447),
    "TX": (31.968599, -99.901813), "UT": (39.320980, -111.093731),
    "VT": (44.558803, -72.577841), "VA": (37.431573, -78.656894),
    "WA": (47.751074, -120.740139), "WV": (38.597626, -80.454903),
    "WI": (43.784440, -88.787868), "WY": (43.075968, -107.290284),
}

# States handled as separate insets rather than placed at their true
# (very distant) coordinates within the lower-48 grid.
INSET_STATES = ["AK", "HI"]
CONTINENTAL_STATES = [s for s in CENTROIDS if s not in INSET_STATES]

# Continental US grid bounds (deg)
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.0, 49.5


# ---------------------------------------------------------------------------
# TERRAIN BUILDING
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


def build_real_terrain(values):
    """Rasterize the *actual* state polygons instead of approximating them
    from centroids. Needs `geopandas` + `shapely` and internet access to
    download the US Census Bureau's cartographic boundary shapefile.
    Returns (LON, LAT, Z) on success, or None if anything is unavailable
    or fails - callers should fall back to build_voronoi_terrain().

    NOTE: this path was written to a well-documented, standard pattern for
    reading Census cartographic boundary files with geopandas, but could
    not be executed/tested in the environment this script was written in
    (no internet access there). It is optional and off by default; the
    tested Voronoi method above is the default terrain builder.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.prepared import prep
    except ImportError:
        print("geopandas/shapely not installed - using the built-in "
              "Voronoi terrain instead. (pip install geopandas shapely "
              "for geographically exact state borders.)")
        return None

    url = ("zip+https://www2.census.gov/geo/tiger/GENZ2018/shp/"
           "cb_2018_us_state_20m.zip")
    try:
        states = gpd.read_file(url)
    except Exception as exc:  # noqa: BLE001 - any failure should just fall back
        print(f"Could not download/read Census state boundaries ({exc}); "
              "using the built-in Voronoi terrain instead.")
        return None

    try:
        states = states.set_index("STUSPS")
        exclude = set(INSET_STATES) | {"PR", "VI", "GU", "AS", "MP"}
        states = states[~states.index.isin(exclude)]

        lons = np.linspace(LON_MIN, LON_MAX, GRID_NX)
        lats = np.linspace(LAT_MIN, LAT_MAX, GRID_NY)
        LON, LAT = np.meshgrid(lons, lats)
        flat_lon, flat_lat = LON.ravel(), LAT.ravel()
        result = np.full(flat_lon.shape, np.nan)

        for abbr, row in states.iterrows():
            if abbr not in values:
                continue
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds
            candidates = np.where(
                (flat_lon >= minx) & (flat_lon <= maxx) &
                (flat_lat >= miny) & (flat_lat <= maxy)
            )[0]
            if len(candidates) == 0:
                continue
            prepared = prep(geom)
            for i in candidates:
                if prepared.contains(Point(flat_lon[i], flat_lat[i])):
                    result[i] = values[abbr]

        Z = result.reshape(LON.shape)
        valid_mask = ~np.isnan(Z)
        filled = np.where(valid_mask, Z, np.nanmin(Z))

        if GAUSS_SMOOTH:
            Z = gaussian_filter(filled, sigma=SMOOTH_SIGMA * 0.6)
        else:
            Z = filled

        Z[~valid_mask] = np.nan
        return LON, LAT, Z
    except Exception as exc:  # noqa: BLE001
        print(f"Real-border rasterization failed ({exc}); "
              "using the built-in Voronoi terrain instead.")
        return None


def get_terrain(values):
    """Dispatcher: try the real-border method if enabled, else Voronoi."""
    if USE_REAL_BORDERS:
        real = build_real_terrain(values)
        if real is not None:
            return real
    return build_voronoi_terrain(values)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_relief(values, title, cmap_name, label_suffix, filename):
    """values: dict {abbr: (temp, place, year)} for RECORD_HIGH or RECORD_LOW."""
    numeric = {a: v[0] for a, v in values.items()}
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

    # --- Alaska / Hawaii insets, drawn as small markers at their true
    # elevation off to the side (standard convention on US maps, since
    # their true coordinates are far from the lower 48). Each is a short
    # flat "peg" centered on the state's actual value, rather than a bar
    # rising from an arbitrary baseline - that keeps its position on the
    # color/height scale directly comparable to the main terrain instead
    # of implying a misleading "distance from zero".
    inset_positions = {"AK": (-123.0, 26.5), "HI": (-118.5, 26.5)}
    peg_half_height = (vmax - vmin) * 0.02
    dx = dy = 2.4
    for abbr in INSET_STATES:
        temp = numeric[abbr]
        ilon, ilat = inset_positions[abbr]
        color = cmap(norm(temp))
        z0 = temp - peg_half_height
        ax.bar3d(ilon - dx / 2, ilat - dy / 2, z0, dx, dy, peg_half_height * 2,
                  color=color, shade=True, edgecolor="black", linewidth=0.4)
        ax.text(ilon, ilat, temp + (vmax - vmin) * 0.09, abbr,
                fontsize=9, ha="center", weight="bold",
                path_effects=label_outline)

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

    fig.text(0.5, 0.02,
              "Data: NOAA National Climatic Data Center / State Climate "
              "Extremes Committee, statewide records. AK & HI shown as insets.",
              ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(filename, dpi=OUTPUT_DPI, facecolor="white")
    plt.show(block=True)
    #plt.close(fig)
    print(f"Saved {filename}")


def main():
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


if __name__ == "__main__":
    main()
