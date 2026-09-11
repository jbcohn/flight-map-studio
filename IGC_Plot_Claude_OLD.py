#Created on Tue Jul  8 10:39:59 2025

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#@author: joshcohn
"""
Simple IGC to PDF converter using matplotlib (no browser required)
"""

import os
import math
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import contextily as cx
import pyproj
import concurrent.futures
from tqdm import tqdm

# --- Configuration ---
IGC_FOLDER_PATH = 'Tracks'  # Change this to your folder path
OUTPUT_IMAGE_FILE = 'igc_tracks_map.png'  # Change extension to .jpg, .tiff, or .webp as needed
MAX_REASONABLE_DEGREE_JUMP = 0.5

# --- Bounding Box for Subset ---
# Set to None to plot all data, or provide a dictionary to crop the map to a subset.
BOUNDING_BOX = {'min_lon': -125.0, 'max_lon': -110.0, 'min_lat': 34.0, 'max_lat': 44.0}

# --- Map Background Resolution ---
# Increase this number for sharper maps (e.g. 10, 12, 14). Warning: High zoom levels on large bounding boxes download thousands of tiles!
MAP_ZOOM = 10

# --- Render Settings ---
OUTPUT_DPI = 600  # Decrease to 150 for rapid testing, 300 for standard, 600 for large prints

# PDF settings for high resolution
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.figsize'] = (24, 18)  # Larger figure size for better print quality
#plt.rcParams['path.simplify'] = True          # Drops invisible GPS points to speed up rendering
#plt.rcParams['path.simplify_threshold'] = 1.0 # 1.0 pixel threshold for aggressive simplification

# --- Helper Function: Parse IGC Latitude/Longitude ---
def parse_igc_lat_lon(lat_str, lon_str):
    """
    Parses IGC format latitude and longitude strings into decimal degrees.
    """
    try:
        # Latitude
        lat_deg = int(lat_str[0:2])
        lat_min = int(lat_str[2:4])
        lat_min_dec = int(lat_str[4:7]) / 1000.0
        lat_sign = 1 if lat_str[7] == 'N' else -1
        lat_decimal = lat_sign * (lat_deg + (lat_min + lat_min_dec) / 60.0)

        # Longitude
        lon_deg = int(lon_str[0:3])
        lon_min = int(lon_str[3:5])
        lon_min_dec = int(lon_str[5:8]) / 1000.0
        lon_sign = 1 if lon_str[8] == 'E' else -1
        lon_decimal = lon_sign * (lon_deg + (lon_min + lon_min_dec) / 60.0)

        return lat_decimal, lon_decimal
    except (ValueError, IndexError) as e:
        print(f"Error parsing lat/lon: {lat_str}, {lon_str} - {e}")
        return None, None

# --- Loading Bar Helper for Blocking Operations ---
class SweepingProgressBar:
    def __init__(self, desc):
        self.desc = desc
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run)

    def _run(self):
        with tqdm(total=100, desc=self.desc, bar_format="{desc}: |{bar}| [{elapsed}]") as pbar:
            step = 2
            while not self.stop_event.is_set():
                pbar.n += step
                if pbar.n >= 100:
                    pbar.n = 100
                    step = -2
                elif pbar.n <= 0:
                    pbar.n = 0
                    step = 2
                pbar.refresh()
                time.sleep(0.05)
            pbar.n = 100
            pbar.refresh()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()

# --- Main Script ---
print(f"Searching for .igc files in: {IGC_FOLDER_PATH}")

if not os.path.isdir(IGC_FOLDER_PATH):
    print(f"Error: Folder not found at '{IGC_FOLDER_PATH}'")
    exit()

igc_files = [f for f in os.listdir(IGC_FOLDER_PATH) if f.lower().endswith('.igc')]

if not igc_files:
    print(f"No .igc files found in '{IGC_FOLDER_PATH}'.")
    exit()

print(f"Found {len(igc_files)} .igc files: {', '.join(igc_files)}")

# Use a dark red color for all tracks
track_color = '#8B0000'

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(24, 18))
ax.set_aspect('equal')

# Set up coordinate transformers (WGS84 lat/lon to Web Mercator)
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
inv_transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

all_tracks = []
track_names = []

# --- Helper Function: Process Single IGC File ---
def process_igc_file(filename):
    filepath = os.path.join(IGC_FOLDER_PATH, filename)
    track_lats = []
    track_lons = []
    previous_coord = None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('B') and len(line) >= 35:
                    lat_str = line[7:15]
                    lon_str = line[15:24]
                    validity = line[24]

                    if validity == 'A':
                        lat, lon = parse_igc_lat_lon(lat_str, lon_str)
                        if lat is not None and lon is not None:
                            current_coord = (lat, lon)

                            if previous_coord:
                                dLat = current_coord[0] - previous_coord[0]
                                dLon = current_coord[1] - previous_coord[1]
                                degree_jump = math.sqrt(dLat**2 + dLon**2)

                                if degree_jump <= MAX_REASONABLE_DEGREE_JUMP:
                                    track_lats.append(lat)
                                    track_lons.append(lon)
                                    previous_coord = current_coord
                            else:
                                track_lats.append(lat)
                                track_lons.append(lon)
                                previous_coord = current_coord

        return filename, track_lats, track_lons, None
    except Exception as e:
        return filename, None, None, str(e)

# Process files in parallel
print(f"\nParsing {len(igc_files)} files in parallel...")
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_igc_file, f): f for f in igc_files}
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parsing IGCs", unit="file"):
        filename, track_lats, track_lons, error = future.result()
        if error:
            tqdm.write(f"Error processing {filename}: {error}")
        elif track_lats:
            all_tracks.append((track_lats, track_lons))
            track_names.append(filename)

if not all_tracks:
    print("No valid tracks found!")
    exit()

print("\nTransforming and plotting tracks...")
for track_lats, track_lons in tqdm(all_tracks, desc="Plotting Tracks", unit="track"):
    # Transform coordinates to Web Mercator for proper projection
    track_x, track_y = transformer.transform(track_lons, track_lats)
    
    # Plot the thick background track for weight (glow effect)
    ax.plot(track_x, track_y, color=track_color, linewidth=1.0, alpha=0.2)
    
    # Plot the thin foreground track for sharp detail
    ax.plot(track_x, track_y, color=track_color, linewidth=0.03, alpha=0.9)

# Set up the plot
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('Flight Tracks', fontsize=17, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add coordinate labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Format axis labels to translate Web Mercator meters back to lat/lon degrees
def format_x(x, pos):
    lon, _ = inv_transformer.transform(x, 0)
    return f'{lon:.1f}°'
    
def format_y(y, pos):
    _, lat = inv_transformer.transform(0, y)
    return f'{lat:.1f}°'

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

# Zoom into the requested subset of the map
if BOUNDING_BOX:
    min_x, min_y = transformer.transform(BOUNDING_BOX['min_lon'], BOUNDING_BOX['min_lat'])
    max_x, max_y = transformer.transform(BOUNDING_BOX['max_lon'], BOUNDING_BOX['max_lat'])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    print(f"\nCropping map to bounding box: {BOUNDING_BOX}")

# Force grid ticks to be exactly at whole degrees
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

lon_min, _ = inv_transformer.transform(xmin, 0)
lon_max, _ = inv_transformer.transform(xmax, 0)
_, lat_min = inv_transformer.transform(0, ymin)
_, lat_max = inv_transformer.transform(0, ymax)

# Find the nearest multiples of 5 for start and end bounds
lon_start = math.ceil((lon_min - 1e-5) / 5) * 5
lon_end = math.floor((lon_max + 1e-5) / 5) * 5
lat_start = math.ceil((lat_min - 1e-5) / 5) * 5
lat_end = math.floor((lat_max + 1e-5) / 5) * 5

lon_ticks_deg = np.arange(lon_start, lon_end + 1, 5)
lat_ticks_deg = np.arange(lat_start, lat_end + 1, 5)

x_ticks, _ = transformer.transform(lon_ticks_deg, np.zeros_like(lon_ticks_deg))
_, y_ticks = transformer.transform(np.zeros_like(lat_ticks_deg), lat_ticks_deg)

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# Adjust layout to prevent legend cutoff
print()
try:
    with SweepingProgressBar("Adding Basemap (Downloading tiles)"):
        # EPSG:3857 is native for contextily, no need to warp the basemap
        # Contextily automatically handles caching internally
        cx.add_basemap(ax, source=cx.providers.Esri.WorldTopoMap, zoom=MAP_ZOOM)
except Exception as e:
    print(f"\nWarning: Could not add basemap due to a PROJ/Contextily error:")
    print(f"  {e}")
    print("\nThis is usually caused by conflicting PROJ versions in your Conda environment.")
    print("To fix it, try running this in your terminal: conda install -c conda-forge proj")
    print("Continuing without the basemap...")

# Save as high-resolution image
print(f"\nSaving high-resolution image: {OUTPUT_IMAGE_FILE}")
with SweepingProgressBar("Rendering & Saving Image"):
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight', dpi=OUTPUT_DPI)

print(f"Image successfully generated: {os.path.abspath(OUTPUT_IMAGE_FILE)}")
print("The image contains high-resolution flight tracks ready for printing.")

plt.close()