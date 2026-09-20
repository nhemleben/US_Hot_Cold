# US Temperature Relief Map

A Python script that turns statewide U.S. temperature records into two 3D, topographic-style relief maps:

- **Record highs:** each state's height represents its all-time record high temperature.
- **Record lows:** each state's depth represents its all-time record low temperature.

The figures use statewide records from the NOAA National Climatic Data Center / State Climate Extremes Committee, as compiled by [Golden Gate Weather Services](https://ggweather.com/climate/extremes_us.htm). The records in the script are identified as official statewide records as of 2024.

## Features

- Generates a record-high peaks map and a record-low valleys map.
- Uses state geographic centers to create a stylized Voronoi-like terrain for the lower 48 states.
- Corrects longitude distances by latitude so the terrain grid is less distorted.
- Applies Gaussian smoothing to create rolling, topographic-style transitions.
- Clips the terrain to the convex hull of the state centers for a recognizable U.S. shape.
- Shows Alaska and Hawaii as inset columns so their true locations do not compress the lower-48 map.
- Saves high-resolution PNG images and keeps each figure open until its window is closed.

## Requirements

Core dependencies:

- Python 3
- NumPy
- SciPy
- Matplotlib

Optional dependencies for actual state borders:

- GeoPandas
- Shapely
- Requests

## Installation

Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
```

To enable the optional real-border mode, also install:

```bash
python -m pip install geopandas shapely requests
```

## Usage

Run the script from the repository directory:

```bash
python us_temperature_relief_map.py
```

The script creates:

- `us_record_high_peaks.png`
- `us_record_low_valleys.png`

Each plot is displayed and remains open until its figure window is closed. After closing the first figure, the script proceeds to the second one.

## Border Modes

The script is configured with:

```python
USE_REAL_BORDERS = True
```

When enabled, it attempts to download U.S. Census Bureau cartographic boundary data and rasterize the actual state polygons. This mode requires internet access and the optional dependencies listed above.

If the download or optional imports fail, the script automatically falls back to the offline Voronoi-style terrain method.

Set the value to `False` to always use the offline approximation:

```python
USE_REAL_BORDERS = False
```

## Configuration

The main settings are near the top of `us_temperature_relief_map.py`:

- `GRID_NX`, `GRID_NY`: terrain grid resolution.
- `SMOOTH_SIGMA`: Gaussian blur strength; larger values produce smoother terrain.
- `SHOW_LABELS`: enables or disables state abbreviation labels.
- `OUTPUT_DPI`: resolution of the saved PNG files.

The record data is stored in the `RECORD_HIGH` and `RECORD_LOW` dictionaries. Each entry contains:

```python
"STATE": (temperature_fahrenheit, location, year)
```

## How the Terrain Is Built

The default offline method does not require a shapefile or GIS library:

1. The script stores each state's geographic center and temperature record.
2. It creates a fine latitude/longitude grid over the continental United States.
3. Each grid point is assigned to its nearest state center.
4. The assigned temperatures are Gaussian-blurred into smooth terrain.
5. The result is clipped to the convex hull of the state centers.
6. The terrain is rendered as a 3D Matplotlib surface, with state labels and a temperature color bar.

This produces a stylized map rather than a geographically exact coastline. Real-border mode provides a crisper, more geographically accurate alternative when its data download succeeds.

## Repository Contents

- `us_temperature_relief_map.py` - main data, terrain-generation logic, and plotting script.
- `us_record_high_peaks.png` - generated record-highs figure.
- `us_record_low_valleys.png` - generated record-lows figure.
- `files/` - supporting repository files, if used by future revisions.
