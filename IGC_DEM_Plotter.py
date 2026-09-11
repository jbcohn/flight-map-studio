#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGC Flight Track DEM & Topo Map Plotter
=======================================
Plots paragliding / hang gliding IGC flight tracks over high-resolution
terrain maps, including:
  1. Shaded relief + hypsometric elevation tint (from DEM GeoTIFFs, National Geographic style)
  2. Web topographic / satellite tile basemaps (Esri WorldTopo, OpenTopo, etc.)
  3. Hybrid overlays

Includes:
  - Fast parallel IGC parsing with coordinate filtering
  - Quick-test subset sampling (e.g. 1/10th of tracks) for instant iterations
  - Multi-layer glowing track rendering for high contrast against terrain
  - Support for California Albers (EPSG:3310) and Web Mercator (EPSG:3857) projections
  - High-resolution poster / print output (PNG, PDF, TIFF)
"""

import os
import sys
import math
import glob
import json
import random
import argparse
import urllib.request
import concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyproj
from tqdm import tqdm

# Set Matplotlib cache dir if not writable
if not os.access(os.path.expanduser("~/.matplotlib"), os.W_OK):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# ============================================================
# CONFIGURATION - Edit defaults here or override via CLI
# ============================================================

# --- Data Sources ---
IGC_FOLDER_PATH = "Tracks"       # Path to directory containing .igc / .IGC files
DEM_SEARCH_PATHS = [             # Directories to search for DEM and GeoJSON files
    ".",
    "../T-Shirt Designs",
    os.path.expanduser("~/Developer/T-Shirt Designs")
]

# --- Region & Basemap Mode ---
# BASEMAP_MODE: "dem" (shaded relief), "tiles" (online topo/satellite tiles), or "none" (blank canvas)
BASEMAP_MODE = "dem"

# REGION: "california", "bay_area", or "custom" / "auto"
REGION = "california"

# --- Sampling & Testing ---
# Set SAMPLE_FRACTION to 0.1 to quickly load 1/10th of tracks for testing, or 1.0 for all tracks
SAMPLE_FRACTION = 1.0            # 0.1 = 10% sample, 1.0 = 100% (all tracks)
MAX_TRACKS = None                # Limit total number of tracks (None = unlimited)
RANDOM_SEED = 42                 # Seed for reproducible random sampling

# --- Bounding Box / Extent ---
# Set to None to auto-fit to all loaded tracks or use a region preset
# e.g., {'min_lon': -125.0, 'max_lon': -114.0, 'min_lat': 32.5, 'max_lat': 42.5}
CUSTOM_BOUNDING_BOX = None
AUTO_CROP_TO_TRACKS = False      # If True and BOUNDING_BOX is None, tight-crops around tracks with padding
AUTO_CROP_PADDING_DEG = 0.3      # Padding in degrees around tracks when auto-cropping

# --- Region Presets ---
REGION_PRESETS = {
    "california": {
        "dem_filename": "dem_california.tif",
        "target_crs": "+proj=aea +lat_0=0 +lon_0=-120 +lat_1=34 +lat_2=40.5 +x_0=0 +y_0=-4000000 +ellps=GRS80 +units=m +no_defs",   # California Albers Equal Area (EPSG:3310)
        "pixel_size_m": 150,
        "mask_to_border": True,       # Mask ocean and non-CA areas
        "feather_border": True,
        "feather_pixels": 25,
        "color_ramp": [
            (0,    (155, 186, 142)),  # Light soft sage (sea level / Central Valley - high contrast for red tracks)
            (200,  (175, 198, 142)),  # Light lime/yellow-green
            (600,  (205, 210, 138)),  # Warm foothill gold
            (1200, (222, 196, 126)),  # Mid elevation ochre
            (2000, (212, 156, 102)),  # Warm mountain copper
            (2800, (175, 118, 85)),   # Subalpine ridge
            (3400, (140, 100, 95)),   # High Sierra granite / rock
            (4000, (252, 252, 252)),  # Snow-capped peaks
        ],
        "default_bbox": None
    },
    "bay_area": {
        "dem_filename": "dem30m.tif",
        "target_crs": "+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs",   # UTM Zone 10N (EPSG:32610)
        "pixel_size_m": 30,
        "mask_to_border": False,
        "color_ramp": [
            (0,    (155, 186, 142)),  # Light soft sage (sea level / bay lowlands)
            (50,   (175, 198, 142)),
            (150,  (200, 208, 140)),  # Light golden foothills
            (300,  (218, 198, 130)),
            (500,  (215, 168, 105)),  # Mid ridges
            (700,  (180, 125, 90)),
            (1000, (145, 105, 95)),   # Mt Diablo / Mt Tam peaks
            (1400, (252, 252, 252)),
        ],
        "default_bbox": None
    }
}

# --- DEM Relief Settings ---
SUN_AZIMUTH = 315       # Sun azimuth angle in degrees (315 = NW illumination)
SUN_ALTITUDE = 45       # Sun altitude angle above horizon in degrees
AMBIENT_LIGHT = 0.38    # Ambient light fill factor (0.0 to 1.0)
TRANSPARENT_BG = False  # If True, nodata / ocean is transparent RGBA; False uses WATER_COLOR
WATER_COLOR = (0.50, 0.65, 0.72) # Ocean / water body color (RGB normalized)

# --- Online Tiles Settings (when BASEMAP_MODE = "tiles") ---
TILE_PROVIDER = "Esri.WorldTopoMap" # "Esri.WorldTopoMap", "Esri.WorldImagery", "OpenTopoMap"
TILE_ZOOM = 10

# --- Track Appearance ---
TRACK_STYLE = "glow"    # "glow" (thick glow + sharp core), "solid", or "altitude"
TRACK_COLOR = "#9E0000" # Rich ruby/crimson for high contrast on green/gold terrain
GLOW_COLOR = "#FF2222"  # Outer glow color
GLOW_ALPHA = 0.25       # Outer glow opacity
CORE_ALPHA = 0.95       # Sharp line opacity
GLOW_LINEWIDTH = 1.2    # Width of the glow underlay
CORE_LINEWIDTH = 0.12   # Width of the sharp foreground line
MAX_REASONABLE_DEGREE_JUMP = 0.5  # Max jump in degrees between consecutive fixes (cleans GPS errors)

# --- Output & Figure Settings ---
OUTPUT_FILE = "igc_flight_map.png"
OUTPUT_DPI = 300        # 150 for test previews, 300 for standard publication, 600 for poster prints
FIGURE_SIZE_IN = (20, 24) # Width x Height in inches (proportioned for California aspect ratio)
SHOW_GRID = True        # Overlay lat/lon degree grid lines
SHOW_TITLE = True       # Overlay title and flight statistics


# ============================================================
# Helper Functions: File Discovery & Coordinate Transformation
# ============================================================

def find_file(filename, search_dirs=DEM_SEARCH_PATHS):
    """Search for a file across given directories."""
    if os.path.exists(filename):
        return os.path.abspath(filename)
    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def parse_igc_lat_lon(lat_str, lon_str):
    """Parses IGC latitude and longitude strings into decimal degrees."""
    try:
        lat_deg = int(lat_str[0:2])
        lat_min = int(lat_str[2:4])
        lat_min_dec = int(lat_str[4:7]) / 1000.0
        lat_sign = 1 if lat_str[7] == 'N' else -1
        lat = lat_sign * (lat_deg + (lat_min + lat_min_dec) / 60.0)

        lon_deg = int(lon_str[0:3])
        lon_min = int(lon_str[3:5])
        lon_min_dec = int(lon_str[5:8]) / 1000.0
        lon_sign = 1 if lon_str[8] == 'E' else -1
        lon = lon_sign * (lon_deg + (lon_min + lon_min_dec) / 60.0)

        return lat, lon
    except (ValueError, IndexError):
        return None, None


def parse_igc_altitude(line):
    """Parses pressure altitude (bytes 25-30) and GPS altitude (bytes 30-35) from a B record."""
    try:
        press_alt = int(line[25:30]) if len(line) >= 30 and line[25:30].isdigit() else 0
        gps_alt = int(line[30:35]) if len(line) >= 35 and line[30:35].isdigit() else press_alt
        return gps_alt
    except (ValueError, IndexError):
        return 0


# ============================================================
# Parallel IGC Ingestion & Filtering
# ============================================================

def process_single_igc(filepath, max_jump=MAX_REASONABLE_DEGREE_JUMP):
    """Parses a single IGC file and extracts valid track coordinates and altitudes."""
    track_lats = []
    track_lons = []
    track_alts = []
    prev_coord = None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('B') and len(line) >= 35:
                    if line[24] == 'A':  # Valid 3D GPS fix
                        lat, lon = parse_igc_lat_lon(line[7:15], line[15:24])
                        if lat is not None and lon is not None:
                            curr_coord = (lat, lon)
                            if prev_coord:
                                dlat = curr_coord[0] - prev_coord[0]
                                dlon = curr_coord[1] - prev_coord[1]
                                dist = math.sqrt(dlat * dlat + dlon * dlon)
                                if dist > max_jump:
                                    continue  # Skip GPS glitch
                            track_lats.append(lat)
                            track_lons.append(lon)
                            track_alts.append(parse_igc_altitude(line))
                            prev_coord = curr_coord

        return os.path.basename(filepath), track_lats, track_lons, track_alts, None
    except Exception as e:
        return os.path.basename(filepath), None, None, None, str(e)


def load_all_igc_tracks(folder_path, sample_fraction=1.0, max_tracks=None, seed=RANDOM_SEED):
    """Loads and parses IGC files in parallel, with optional fast sampling for testing."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Tracks directory '{folder_path}' not found.")

    files = glob.glob(os.path.join(folder_path, "*.igc")) + glob.glob(os.path.join(folder_path, "*.IGC"))
    files = sorted(list(set(files)))
    total_files = len(files)

    if total_files == 0:
        raise ValueError(f"No .igc files found in '{folder_path}'.")

    # Sample subset if requested (e.g. 1/10th for test mode)
    if sample_fraction < 1.0 or max_tracks is not None:
        random.seed(seed)
        sample_size = int(total_files * sample_fraction) if sample_fraction < 1.0 else total_files
        if max_tracks is not None:
            sample_size = min(sample_size, max_tracks)
        sample_size = max(1, sample_size)
        files = sorted(random.sample(files, sample_size))
        print(f"\n[Test Mode Active] Selected {len(files)} of {total_files} tracks ({sample_fraction*100:.0f}% sample, seed={seed})")
    else:
        print(f"\nProcessing all {total_files} IGC flight tracks...")

    tracks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_igc, f): f for f in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parsing IGC files", unit="file"):
            filename, lats, lons, alts, err = future.result()
            if err:
                tqdm.write(f"  Warning: error in {filename}: {err}")
            elif lats and len(lats) > 5:
                tracks.append({
                    "filename": filename,
                    "lats": np.array(lats, dtype=np.float64),
                    "lons": np.array(lons, dtype=np.float64),
                    "alts": np.array(alts, dtype=np.float32)
                })

    print(f"Successfully loaded {len(tracks)} valid flight tracks.")
    return tracks, total_files


# ============================================================
# DEM Terrain Shading & Hypsometric Tinting Engine
# ============================================================

def download_california_dem_if_needed(output_path):
    """Downloads 3DEP elevation tiles from USGS if local DEM is missing."""
    import requests
    import rasterio
    from rasterio.merge import merge

    print(f"[{output_path}] not found. Auto-downloading California DEM tiles from USGS 3DEP...")
    bbox_grid = [
        (-124.6, 37.2, -119.2, 42.1),  # NW
        (-119.4, 37.2, -114.0, 42.1),  # NE
        (-124.6, 32.4, -119.2, 37.3),  # SW
        (-119.4, 32.4, -114.0, 37.3),  # SE
    ]

    url = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
    tmp_files = []

    for i, (w, s, e, n) in enumerate(bbox_grid):
        params = {
            'bbox': f'{w},{s},{e},{n}',
            'bboxSR': '4326',
            'imageSR': '4326',
            'size': '2500,2500',
            'format': 'tiff',
            'f': 'image'
        }
        print(f"  Downloading USGS elevation tile {i+1}/4...")
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        tmp_path = f"_ca_tile_{i}.tif"
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        tmp_files.append(tmp_path)

    print("Merging tiles into master GeoTIFF...")
    datasets = [rasterio.open(p) for p in tmp_files]
    mosaic, out_trans = merge(datasets)

    out_meta = datasets[0].meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans,
        'nodata': datasets[0].nodata
    })

    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(mosaic)

    for ds in datasets:
        ds.close()
    for p in tmp_files:
        if os.path.exists(p):
            os.remove(p)

    print(f"Saved California DEM to {output_path}")


def get_california_border_mask(src_crs, src_width, src_height, src_bounds, src_transform, dst_shape, dst_transform, target_crs):
    """Generates a raster mask of California landmass boundaries."""
    from rasterio.features import rasterize
    from rasterio.warp import reproject, Resampling

    geojson_path = find_file("california.geojson")
    if not geojson_path:
        geojson_path = "california.geojson"
        try:
            url = "https://raw.githubusercontent.com/glynnbird/usstatesgeojson/master/california.geojson"
            print("Downloading California state boundary GeoJSON...")
            urllib.request.urlretrieve(url, geojson_path)
        except Exception as e:
            print(f"Warning: Could not fetch california.geojson ({e}). Skipping border mask.")
            return np.ones(dst_shape, dtype=bool)

    with open(geojson_path, "r") as f:
        ca_geojson = json.load(f)

    if ca_geojson.get('type') == 'Feature':
        shapes = [ca_geojson['geometry']]
    elif ca_geojson.get('type') == 'FeatureCollection':
        shapes = [feat['geometry'] for feat in ca_geojson['features']]
    else:
        shapes = [ca_geojson]

    mask_orig = rasterize(shapes, out_shape=(src_height, src_width), transform=src_transform, fill=0, default_value=1)
    mask_dst = np.zeros(dst_shape, dtype=np.uint8)

    reproject(
        source=mask_orig,
        destination=mask_dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    return mask_dst > 0


def compute_hillshade(elev, azimuth_deg, altitude_deg, dx, dy):
    """Computes realistic topographic hillshading."""
    gy, gx = np.gradient(elev, dy, dx)
    slope = np.pi / 2 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = azimuth_deg * np.pi / 180
    altitude = altitude_deg * np.pi / 180
    hs = (np.sin(altitude) * np.sin(slope) +
          np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect))
    return np.clip(hs, 0, 1).astype(np.float32)


def hypsometric_color(elev, color_ramp):
    """Applies hypsometric elevation tinting along the provided color stops."""
    elevs = np.array([s[0] for s in color_ramp], dtype=np.float32)
    colors = np.array([s[1] for s in color_ramp], dtype=np.float32)
    flat = elev.ravel()
    r = np.interp(flat, elevs, colors[:, 0]).astype(np.float32)
    g = np.interp(flat, elevs, colors[:, 1]).astype(np.float32)
    b = np.interp(flat, elevs, colors[:, 2]).astype(np.float32)
    return np.stack([r, g, b], axis=-1).reshape(*elev.shape, 3) / 255.0


def render_dem_relief_raster(dem_path, target_crs, pixel_size_m, mask_to_border, feather_border, feather_pixels, color_ramp):
    """Loads DEM GeoTIFF, reprojects, computes shaded relief with hypsometric tint, and returns RGBA array + extent."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    if not os.path.exists(dem_path):
        if "california" in dem_path.lower():
            download_california_dem_if_needed(dem_path)
        else:
            raise FileNotFoundError(f"DEM GeoTIFF not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=pixel_size_m
        )
        dst_arr = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan if src.nodata is None else src.nodata,
        )
        nodata = src.nodata

        if mask_to_border and "california" in dem_path.lower():
            ca_mask = get_california_border_mask(
                src.crs, src.width, src.height, src.bounds, src.transform,
                (height, width), transform, target_crs
            )
            if feather_border:
                from scipy.ndimage import distance_transform_edt, gaussian_filter
                dist_in = distance_transform_edt(ca_mask)
                weight = np.clip(dist_in / float(feather_pixels), 0, 1)
                weight_smooth = gaussian_filter(weight, sigma=3.0)
                dst_arr = np.where(ca_mask, dst_arr * weight_smooth, np.nan)
            else:
                dst_arr[~ca_mask] = np.nan

    dx = transform.a
    dy = -transform.e
    mask = (dst_arr == nodata) | np.isnan(dst_arr) if nodata is not None else np.isnan(dst_arr)
    elev = np.where(mask, 0, dst_arr).astype(np.float32)

    # Compute hillshade and hypsometric elevation tint
    hillshade = compute_hillshade(elev, SUN_AZIMUTH, SUN_ALTITUDE, dx, dy)
    base_color = hypsometric_color(elev, color_ramp)
    shaded = base_color * (AMBIENT_LIGHT + (1 - AMBIENT_LIGHT) * hillshade[..., None])
    shaded = np.clip(shaded, 0, 1)

    # Convert to RGBA
    if TRANSPARENT_BG:
        alpha = np.where(mask, 0.0, 1.0)[..., None]
        rgba = np.dstack([shaded, alpha])
    else:
        # Fill ocean/nodata with pleasant water color
        water_bg = np.array(WATER_COLOR, dtype=np.float32)
        shaded = np.where(mask[..., None], water_bg, shaded)
        alpha = np.ones((height, width, 1), dtype=np.float32)
        rgba = np.dstack([shaded, alpha])

    # Compute bounding extent in target CRS coordinates: (left, right, bottom, top)
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e  # note transform.e is negative
    extent = (left, right, bottom, top)

    return rgba, extent, (transform, width, height)


# ============================================================
# Main Plotting Engine
# ============================================================

def plot_igc_tracks(
    tracks,
    total_found_tracks,
    basemap_mode=BASEMAP_MODE,
    region=REGION,
    output_file=OUTPUT_FILE,
    dpi=OUTPUT_DPI,
    custom_bbox=CUSTOM_BOUNDING_BOX,
    auto_crop=AUTO_CROP_TO_TRACKS,
    track_style=TRACK_STYLE,
    track_color=TRACK_COLOR,
    glow_color=GLOW_COLOR,
    glow_alpha=GLOW_ALPHA,
    core_alpha=CORE_ALPHA,
    glow_linewidth=GLOW_LINEWIDTH,
    core_linewidth=CORE_LINEWIDTH
):
    """Main rendering pipeline that combines DEM / tile basemap and flight tracks."""
    preset = REGION_PRESETS.get(region.lower(), REGION_PRESETS["california"])
    target_crs = preset["target_crs"]

    # Transformers between WGS84 GPS (EPSG:4326) and Target Projection
    transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    inv_transformer = pyproj.Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, facecolor="#F8FAFC" if not TRANSPARENT_BG else "none")
    ax.set_aspect('equal')

    # Determine Active Spatial Bounding Box
    if custom_bbox:
        bbox = custom_bbox
    elif auto_crop and tracks:
        all_lats = np.concatenate([t["lats"] for t in tracks])
        all_lons = np.concatenate([t["lons"] for t in tracks])
        pad = AUTO_CROP_PADDING_DEG
        bbox = {
            'min_lon': max(-180.0, float(np.min(all_lons)) - pad),
            'max_lon': min(180.0, float(np.max(all_lons)) + pad),
            'min_lat': max(-90.0, float(np.min(all_lats)) - pad),
            'max_lat': min(90.0, float(np.max(all_lats)) + pad),
        }
    else:
        bbox = preset.get("default_bbox")

    # 1. Render Basemap
    if basemap_mode == "dem":
        dem_file = find_file(preset["dem_filename"])
        if not dem_file:
            dem_file = preset["dem_filename"]

        print(f"\nRendering DEM Shaded Relief from: {dem_file}")
        rgba, extent, _ = render_dem_relief_raster(
            dem_file,
            target_crs=preset["target_crs"],
            pixel_size_m=preset["pixel_size_m"],
            mask_to_border=preset.get("mask_to_border", False),
            feather_border=preset.get("feather_border", True),
            feather_pixels=preset.get("feather_pixels", 25),
            color_ramp=preset["color_ramp"]
        )
        ax.imshow(rgba, extent=extent, origin='upper', interpolation='bilinear', zorder=1)

    elif basemap_mode == "tiles":
        import contextily as cx
        print(f"\nAdding Contextily online tile basemap ({TILE_PROVIDER})...")
        # For contextily tiles, Web Mercator is standard
        target_crs = "EPSG:3857"
        transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        inv_transformer = pyproj.Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

    # 2. Render Flight Tracks
    print(f"\nTransforming and plotting {len(tracks)} flight tracks...")
    for track in tqdm(tracks, desc="Plotting tracks", unit="track"):
        lats = track["lats"]
        lons = track["lons"]
        tx, ty = transformer.transform(lons, lats)
        valid = np.isfinite(tx) & np.isfinite(ty)
        if not np.any(valid):
            continue
        tx = np.array(tx)[valid]
        ty = np.array(ty)[valid]

        if track_style == "glow":
            # Layer 1: Outer soft glow for visibility against dark/light terrain
            ax.plot(tx, ty, color=glow_color, linewidth=glow_linewidth, alpha=glow_alpha, zorder=5, solid_capstyle='round')
            # Layer 2: Core high-density line for razor-sharp vector precision
            ax.plot(tx, ty, color=track_color, linewidth=core_linewidth, alpha=core_alpha, zorder=6, solid_capstyle='round')
        elif track_style == "altitude":
            # Altitude colormap gradient
            alts = np.array(track["alts"])[valid]
            ax.scatter(tx, ty, c=alts, cmap="turbo", s=0.5, alpha=0.8, zorder=6)
        else:
            ax.plot(tx, ty, color=track_color, linewidth=glow_linewidth, alpha=core_alpha, zorder=6)

    # Apply Extent / Bounding Box
    if bbox:
        corner_lons = [bbox['min_lon'], bbox['max_lon'], bbox['min_lon'], bbox['max_lon']]
        corner_lats = [bbox['min_lat'], bbox['min_lat'], bbox['max_lat'], bbox['max_lat']]
        c_x, c_y = transformer.transform(corner_lons, corner_lats)
        valid_c = np.isfinite(c_x) & np.isfinite(c_y)
        if np.any(valid_c):
            ax.set_xlim(np.min(np.array(c_x)[valid_c]), np.max(np.array(c_x)[valid_c]))
            ax.set_ylim(np.min(np.array(c_y)[valid_c]), np.max(np.array(c_y)[valid_c]))
    elif basemap_mode == "dem" and extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if basemap_mode == "tiles":
        import contextily as cx
        try:
            cx.add_basemap(ax, source=getattr(cx.providers, TILE_PROVIDER.split('.')[0])[TILE_PROVIDER.split('.')[1]], zoom=TILE_ZOOM, zorder=1)
        except Exception as e:
            print(f"Warning: Could not fetch web basemap tiles: {e}")

    # 3. Coordinate Grids & Professional Cartographic Styling
    if SHOW_GRID:
        ax.grid(True, linestyle="--", linewidth=0.5, color="#555555", alpha=0.25, zorder=10)

        # Coordinate formatters using map midpoints for valid projection inverse mapping
        def fmt_lon(x, pos):
            try:
                y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2.0
                lon, _ = inv_transformer.transform(x, y_mid)
                if np.isfinite(lon):
                    return f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'}"
            except Exception:
                pass
            return ""

        def fmt_lat(y, pos):
            try:
                x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2.0
                _, lat = inv_transformer.transform(x_mid, y)
                if np.isfinite(lat):
                    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"
            except Exception:
                pass
            return ""

        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_lat))
        ax.tick_params(axis='both', which='major', labelsize=10, colors="#333333", length=4)

    # Title & Metadata Banner
    if SHOW_TITLE:
        total_pts = sum(len(t['lats']) for t in tracks)
        title_text = f"Flight Tracks Map"
        subtitle_text = f"Rendered {len(tracks):,} Flights ({total_pts:,} GPS Fixes)"
        if len(tracks) < total_found_tracks:
            subtitle_text += f" • Test Sample ({len(tracks)/total_found_tracks*100:.0f}% of {total_found_tracks} total)"

        ax.text(
            0.02, 0.98, f"{title_text}\n{subtitle_text}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color='#1E293B',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='#CBD5E1', linewidth=1),
            zorder=20
        )

    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.06)

    # 4. Save High-Resolution Image
    print(f"\nSaving image to {output_file} at {dpi} DPI...")
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=TRANSPARENT_BG)
    plt.close()
    print(f"Successfully generated: {os.path.abspath(output_file)}")


# ============================================================
# CLI Interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot IGC flight tracks on DEM relief & topo maps.")
    parser.add_argument("--test", action="store_true", help="Quick test mode: samples 10%% of tracks and uses preview 150 DPI")
    parser.add_argument("--sample", type=float, default=None, help="Fraction of tracks to load (e.g. 0.1 for 10%%, 0.5 for 50%%)")
    parser.add_argument("--max-tracks", type=int, default=None, help="Maximum number of tracks to load")
    parser.add_argument("--region", choices=["california", "bay_area"], default=REGION, help="Preset geographic region")
    parser.add_argument("--basemap", choices=["dem", "tiles", "none"], default=BASEMAP_MODE, help="Basemap type")
    parser.add_argument("--dpi", type=int, default=None, help="Output resolution DPI (e.g. 150 for test, 300 for print, 600 for poster)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image filename (.png, .pdf, .tiff)")
    parser.add_argument("--color", type=str, default=TRACK_COLOR, help="Track line hex color (default: #9E0000)")
    parser.add_argument("--style", choices=["glow", "solid", "altitude"], default=TRACK_STYLE, help="Track visual style")
    parser.add_argument("--auto-crop", action="store_true", help="Auto-crop map bounds closely around the flight tracks")
    parser.add_argument("--no-mask", action="store_true", help="Disable state border masking (show full rectangular terrain)")
    parser.add_argument("--transparent-bg", action="store_true", help="Make ocean/nodata transparent RGBA")
    parser.add_argument("--track-width", type=float, default=None, help="Scale factor for track line width (e.g. 1.5, 2.0)")
    
    # Strip any accidental stray trailing '--' or '-' from sys.argv
    cleaned_args = [a for a in sys.argv[1:] if a not in ('--', '-')]
    args = parser.parse_args(cleaned_args)

    # Apply CLI Overrides
    sample_frac = args.sample if args.sample is not None else (0.1 if args.test else SAMPLE_FRACTION)
    max_trk = args.max_tracks if args.max_tracks is not None else (45 if args.test and args.sample is None else MAX_TRACKS)
    dpi = args.dpi if args.dpi is not None else (150 if args.test else OUTPUT_DPI)
    out_file = args.output if args.output is not None else (
        f"test_igc_map_{args.region}_{int(sample_frac*100)}pct.png" if args.test else OUTPUT_FILE
    )

    global TRANSPARENT_BG
    if args.transparent_bg:
        TRANSPARENT_BG = True

    glow_w = GLOW_LINEWIDTH * (args.track_width if args.track_width else 1.0)
    core_w = CORE_LINEWIDTH * (args.track_width if args.track_width else 1.0)

    # Allow disabling border masking
    preset = REGION_PRESETS.get(args.region.lower(), REGION_PRESETS["california"])
    if args.no_mask:
        preset["mask_to_border"] = False

    print(f"{'='*60}")
    print(f"IGC Flight Map Plotter")
    print(f"Region: {args.region} | Basemap: {args.basemap} | Sample: {sample_frac*100:.0f}% | DPI: {dpi}")
    print(f"{'='*60}")

    tracks, total_found = load_all_igc_tracks(
        IGC_FOLDER_PATH,
        sample_fraction=sample_frac,
        max_tracks=max_trk
    )

    plot_igc_tracks(
        tracks=tracks,
        total_found_tracks=total_found,
        basemap_mode=args.basemap,
        region=args.region,
        output_file=out_file,
        dpi=dpi,
        auto_crop=args.auto_crop,
        track_style=args.style,
        track_color=args.color,
        glow_linewidth=glow_w,
        core_linewidth=core_w
    )


if __name__ == "__main__":
    main()
