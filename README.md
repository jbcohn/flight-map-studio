# Flight Map Studio 🗺️✈️

A fast, client-side web application and Python studio for visualizing, analyzing, and exporting paragliding and hang gliding **IGC flight tracks** over continuous 30m Digital Elevation Models (DEM), shaded relief, and topographic basemaps.

Designed to accompany [PG Race Analyzer](https://jbcohn.github.io/pg-race-analyzer/) and [XC Simulator](https://jbcohn.github.io/xc-simulator/).

---

## ✨ Features

- **Continuous 30m DEM Across the Western US**:
  - Dynamic client-side DEM tile shader rendering full 3D shaded relief on HTML5 Canvas in ~2–4ms per tile.
  - Seamless regional coverage across Washington, Oregon, California, Nevada, Idaho, Utah, Arizona, Colorado, Wyoming, Montana, and New Mexico.
  - Custom hypsometric color ramp (*soft sage $\to$ lime $\to$ warm gold $\to$ ochre $\to$ copper $\to$ subalpine sienna $\to$ granite $\to$ snow caps*) with directional NW ($315^\circ$) hillshade illumination.
- **Reference Overlay with Opacity Control**:
  - Optional place names, mountain peaks, county and state boundaries overlay.
  - Interactive opacity slider ($0\% - 100\%$) positioned beneath flight tracks so tracks always stay prominent.
- **Multiple Basemap Options**:
  - **Western US 30m DEM** (Dynamic Shaded Relief)
  - **USGS 3DEP Relief** (High-Resolution Lidar Shaded Relief)
  - **Satellite** (Esri World Imagery)
  - **OpenTopoMap** (Topographic contours)
  - **Dark Canvas** (Minimal cartography)
- **Track Styling & Analytics**:
  - **Glow**: Vibrant core track with soft outer bloom for high visibility over complex terrain.
  - **Altitude**: Continuous Turbo color gradient mapped to altitude (MSL) with an interactive legend.
  - **Vario**: Color-coded climb (emerald green) vs sink (ruby red).
  - **Distinct**: Unique harmonic color per flight for group and competition flight comparisons.
  - Real-time flight inspector: Date, pilot, glider, duration (with midnight UTC rollover handling), distance, max altitude, and interactive hover tooltips.
- **Test Mode & Sampling Slider**:
  - Live sampling slider ($5\% - 100\%$) with quick preset buttons ($10\%, 25\%, 50\%, 100\%$) to test rendering performance with hundreds of tracks.
- **High-Resolution Poster Export**:
  - Native multi-resolution canvas export ($1\times, 2\times, 3\times$ Ultra High Res Print).
  - Composites DEM relief, place names overlay, vector tracks, and a sleek title card into print-ready PNG posters.
- **Zero API Keys & 100% Client-Side**:
  - Runs directly in any modern browser without servers or API keys. Fully compatible with GitHub Pages.

---

## 🚀 Quick Start

### Running Locally
Because modern browsers restrict cross-origin file loading when using `file://`, serve the repository with any local HTTP server:

```bash
# Using Python 3
python3 -m http.server 8000
```

Then open your browser to **`http://localhost:8000`**.

### Usage
1. Drag and drop `.igc` files or an entire folder into the dropzone.
2. Or click **"Load Demo"** to automatically load sample cross-country tracks.
3. Use the sidebar controls to adjust track styling, basemaps, reference overlay opacity, and quick region viewports.
4. Click **"Export Image"** to download a high-resolution framed poster.

---

## 🐍 Python Script (`IGC_DEM_Plotter.py`)

The repository also includes the original standalone CLI plotting tool for batch generation of print posters using Matplotlib and local GeoTIFFs:

```bash
python3 IGC_DEM_Plotter.py --help
python3 IGC_DEM_Plotter.py --region california --style glow --track-width 1.5
```

---

## 📄 License
MIT License. Built for the free flight community.
