#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:35:05 2025

@author: joshcohn
"""

import os
import folium
import re
import math
import random

# --- Configuration ---
# Set the path to the folder containing your .igc files
IGC_FOLDER_PATH = 'Tracks' # IMPORTANT: Change this to your folder path

# Output HTML map file name
OUTPUT_MAP_FILE = 'igc_tracks_map_raven.html'

# Threshold for skipping points based on simple Pythagorean distance in degrees
MAX_REASONABLE_DEGREE_JUMP = 0.5 # Max change in combined lat/lon degrees. Adjust if needed.

# --- Helper Function: Parse IGC Latitude/Longitude ---
def parse_igc_lat_lon(lat_str, lon_str):
    """
    Parses IGC format latitude and longitude strings into decimal degrees.
    Example lat_str: 4734823N
    Example lon_str: 00918345E
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

# --- Helper Function: Generate Random Hex Color ---
#def get_random_color():
#    """Generates a random hex color code."""
#    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# --- Main Script Logic ---

print(f"Searching for .igc files in: {IGC_FOLDER_PATH}")

# Check if the folder exists
if not os.path.isdir(IGC_FOLDER_PATH):
    print(f"Error: Folder not found at '{IGC_FOLDER_PATH}'")
    print("Please create the folder and place your .igc files inside, or update the IGC_FOLDER_PATH variable.")
    exit()

# Find all .igc files
igc_files = [f for f in os.listdir(IGC_FOLDER_PATH) if f.lower().endswith('.igc')]

if not igc_files:
    print(f"No .igc files found in '{IGC_FOLDER_PATH}'.")
    exit()

print(f"Found {len(igc_files)} .igc files: {', '.join(igc_files)}")

map_initialized = False
map_center = None
m = None
all_tracks_coords = [] # To store coordinates from all tracks for bounding box

# Process each IGC file
for filename in igc_files:
    filepath = os.path.join(IGC_FOLDER_PATH, filename)
    print(f"\nProcessing: {filename}")
    track_coords = []
    previous_coord = None # Keep track of the last valid coordinate added
    points_skipped = 0    # Count skipped points for this file

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith('B'):
                    if len(line) >= 35:
                        lat_str = line[7:15]
                        lon_str = line[15:24]
                        validity = line[24]

                        if validity == 'A':
                            lat, lon = parse_igc_lat_lon(lat_str, lon_str)
                            if lat is not None and lon is not None:
                                current_coord = (lat, lon) # Tuple: (latitude, longitude)

                                # Check distance if we have a previous point
                                if previous_coord:
                                    # Pythagorean distance in degrees (sqrt(dLat^2 + dLon^2))
                                    # current_coord[0] is lat, current_coord[1] is lon
                                    dLat = current_coord[0] - previous_coord[0]
                                    dLon = current_coord[1] - previous_coord[1]
                                    degree_jump = math.sqrt(dLat**2 + dLon**2)

                                    # Check if distance exceeds threshold
                                    if degree_jump > MAX_REASONABLE_DEGREE_JUMP:
                                        # print(f"  Skipping point at line {line_num}: Degree jump {degree_jump:.4f} > {MAX_REASONABLE_DEGREE_JUMP} threshold")
                                        points_skipped += 1
                                        # Do NOT update previous_coord
                                        continue # Skip adding this point
                                    else:
                                        # Distance is OK, add point and update previous_coord
                                        track_coords.append([lat, lon])
                                        previous_coord = current_coord
                                else:
                                    # This is the first valid point, add it and set previous_coord
                                    track_coords.append([lat, lon])
                                    previous_coord = current_coord

    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        continue
    except Exception as e:
        print(f"Error reading or parsing file {filename}: {e}")
        continue

    if not track_coords:
        print(f"No valid coordinates found in {filename}. Skipping.")
        continue

    print(f"Found {len(track_coords)} valid coordinates in {filename}.")
    all_tracks_coords.extend(track_coords) # Add to overall list for bounding box

    # Initialize map on the first valid track's start point if not already done
    if not map_initialized and track_coords:
        map_center = track_coords[0]
        # Use a beautiful topographic tile layer by default, similar to Raven Maps
        m = folium.Map(
            location=map_center,
            zoom_start=12,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Esri | Powered by Esri',
            name='Esri World Topo Map'
        )
        # Add a few other useful layers
        folium.TileLayer('Stamen Terrain', name='Stamen Terrain').add_to(m)
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        map_initialized = True

    # Add the track polyline to the map
    if m and track_coords:
        track_color = '#8B0000' # A deep red for good contrast on topo maps
        folium.PolyLine(
            locations=track_coords,
            color=track_color,
            weight=1.5,
            opacity=0.7,
            tooltip=f"Track: {filename}" # Show filename on hover
        ).add_to(m)

        # Add start marker (optional)
#        folium.Marker(
 #           location=track_coords[0],
  #          popup=f"Start: {filename}",
   #         icon=folium.Icon(color='green', icon='play', prefix='fa')
    #    ).add_to(m)
        # Add end marker (optional)
        # folium.Marker(
        #     location=track_coords[-1],
        #     popup=f"End: {filename}",
        #     icon=folium.Icon(color='red', icon='stop', prefix='fa')
        # ).add_to(m)

# Check if map was initialized (at least one valid track found)
if not map_initialized:
    print("\nNo valid track data found in any file. Map not generated.")
    exit()

# Adjust map bounds to fit all tracks
if all_tracks_coords:
    # Calculate bounds [ [min_lat, min_lon], [max_lat, max_lon] ]
    min_lat = min(p[0] for p in all_tracks_coords)
    max_lat = max(p[0] for p in all_tracks_coords)
    min_lon = min(p[1] for p in all_tracks_coords)
    max_lon = max(p[1] for p in all_tracks_coords)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

# Add Layer Control if multiple layers/tracks exist
folium.LayerControl().add_to(m)

# Save the map to an HTML file
try:
    m.save(OUTPUT_MAP_FILE)
    print(f"\nMap successfully generated!")
    print(f"Output file: {os.path.abspath(OUTPUT_MAP_FILE)}")
    print("Open this HTML file in your web browser to view the map.")
except Exception as e:
    print(f"\nError saving the map to {OUTPUT_MAP_FILE}: {e}")