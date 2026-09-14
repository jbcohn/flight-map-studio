/**
 * Flight Map Studio - Core Application Logic
 * ===========================================
 * Handles client-side IGC parsing, Leaflet map management,
 * dynamic 30m Terrarium DEM shaded relief, place name overlays,
 * track styling (glow, altitude, climb), sampling, and poster export.
 */

// Global Application State
const state = {
    allTracks: [],          // Full array of parsed flight objects
    activeTracks: [],       // Currently displayed subset based on sample slider & visibility
    sampleFraction: 1.0,    // 0.1 to 1.0
    currentStyle: 'glow',   // 'glow' | 'altitude' | 'vario' | 'distinct'
    trackColor: '#DC2626',  // Core track color
    lineWidth: 1.5,
    glowRadius: 4,
    opacity: 0.8,
    activeBasemap: 'western-dem', // Continuous 30m DEM for the Western US
    showLabels: true,       // Place names & boundaries overlay
    labelsOpacity: 0.5,     // 0.0 to 1.0
    selectedTrackId: null,
    exportSize: '24x36',    // '24x36' | '18x24' | '20x30' | '12x18' | 'a1' | 'a2' | 'viewport'
    exportOrientation: 'landscape', // 'landscape' | 'portrait'
    exportDpi: 300,         // 300 (fine print) | 150 (draft/large format)
    showCropFrame: false,   // Only show visual poster frame overlay during export / framing
    cropFrameScale: 0.88,   // Fraction of viewport width/height covered by frame (resizable via corner handles)
};

// Standard Physical Poster Print Sizes
const POSTER_SIZES = {
    '24x36': { name: '24" × 36"', short: '24x36', wIn: 36, hIn: 24, ratio: 36 / 24, label: 'Large Wall Poster' },
    '18x24': { name: '18" × 24"', short: '18x24', wIn: 24, hIn: 18, ratio: 24 / 18, label: 'Standard Frame' },
    '20x30': { name: '20" × 30"', short: '20x30', wIn: 30, hIn: 20, ratio: 30 / 20, label: 'Photo Poster' },
    '12x18': { name: '12" × 18"', short: '12x18', wIn: 18, hIn: 12, ratio: 18 / 12, label: 'Small Print' },
    'a1':    { name: 'A1 (594 × 841 mm)', short: 'A1', wIn: 33.11, hIn: 23.39, ratio: 33.11 / 23.39, label: 'ISO A1' },
    'a2':    { name: 'A2 (420 × 594 mm)', short: 'A2', wIn: 23.39, hIn: 16.54, ratio: 23.39 / 16.54, label: 'ISO A2' },
    'viewport': { name: 'Screen Viewport', short: 'Custom', isViewport: true, label: 'Current View' }
};

// Region Presets (Center Lat, Lon, Zoom)
const REGION_PRESETS = {
    california:   { center: [36.7783, -119.4179], zoom: 6 },
    owens_valley: { center: [36.8000, -118.2500], zoom: 9 },
    tahoe:        { center: [39.0968, -120.0324], zoom: 8 },
    utah:         { center: [40.7608, -111.8910], zoom: 8 },
    colorado:     { center: [39.5501, -105.7821], zoom: 8 },
};

// ============================================================
// Terrarium 30m DEM Shader & Color Palette
// ============================================================

// Color stops matching Python DEM
const DEM_PALETTE = [
    { elev: 0,    r: 155, g: 186, b: 142 }, // Soft sage
    { elev: 400,  r: 175, g: 198, b: 142 }, // Light lime
    { elev: 800,  r: 195, g: 205, b: 138 }, // Chartreuse
    { elev: 1300, r: 215, g: 200, b: 130 }, // Warm gold
    { elev: 1900, r: 220, g: 180, b: 115 }, // Ochre
    { elev: 2500, r: 205, g: 145, b: 95  }, // Copper
    { elev: 3100, r: 165, g: 110, b: 82  }, // Subalpine
    { elev: 3600, r: 130, g: 95,  b: 90  }, // Granite
    { elev: 4200, r: 250, g: 250, b: 250 }  // Snow caps
];

const MAX_ELEV_LUT = 4500;
const lutR = new Uint8Array(MAX_ELEV_LUT + 1);
const lutG = new Uint8Array(MAX_ELEV_LUT + 1);
const lutB = new Uint8Array(MAX_ELEV_LUT + 1);

// Initialize Color LUT once
for (let e = 0; e <= MAX_ELEV_LUT; e++) {
    if (e <= DEM_PALETTE[0].elev) {
        lutR[e] = DEM_PALETTE[0].r;
        lutG[e] = DEM_PALETTE[0].g;
        lutB[e] = DEM_PALETTE[0].b;
    } else if (e >= DEM_PALETTE[DEM_PALETTE.length - 1].elev) {
        const last = DEM_PALETTE[DEM_PALETTE.length - 1];
        lutR[e] = last.r;
        lutG[e] = last.g;
        lutB[e] = last.b;
    } else {
        for (let i = 0; i < DEM_PALETTE.length - 1; i++) {
            const s0 = DEM_PALETTE[i];
            const s1 = DEM_PALETTE[i + 1];
            if (e >= s0.elev && e <= s1.elev) {
                const t = (e - s0.elev) / (s1.elev - s0.elev);
                lutR[e] = Math.round((1 - t) * s0.r + t * s1.r);
                lutG[e] = Math.round((1 - t) * s0.g + t * s1.g);
                lutB[e] = Math.round((1 - t) * s0.b + t * s1.b);
                break;
            }
        }
    }
}

// LRU Rendered Tile Cache
const demTileCache = new Map();
const MAX_DEM_CACHE = 300;

function shadeTerrariumImageData(srcData, z, y) {
    const W = 256, H = 256;
    // Calculate ground cell size at tile center latitude
    const n = Math.PI - (2 * Math.PI * (y + 0.5)) / (1 << z);
    const lat = Math.atan(Math.sinh(n));
    const cellsize = (40075016 * Math.cos(lat)) / (256 * (1 << z));
    const invCell2 = 1.0 / (2 * Math.max(1, cellsize));
    const z_factor = 1.35;

    // Allocate local elevation buffer for thread/event safe processing
    const rawElevBuffer = new Float32Array(65536);

    // Pass 1: Decode elevation in meters
    for (let i = 0; i < 65536; i++) {
        const sIdx = i << 2;
        rawElevBuffer[i] = (srcData[sIdx] * 256.0 + srcData[sIdx + 1] + srcData[sIdx + 2] / 256.0) - 32768.0;
    }

    // Pass 2: Calculate lighting and color LUT
    const outImgData = new ImageData(W, H);
    const out = outImgData.data;

    for (let py = 0; py < H; py++) {
        const yRow = py * W;
        const yAbove = (py > 0 ? py - 1 : 0) * W;
        const yBelow = (py < H - 1 ? py + 1 : H - 1) * W;

        for (let px = 0; px < W; px++) {
            const idx = yRow + px;
            const elev = rawElevBuffer[idx];

            let cr, cg, cb;
            if (elev <= 0) {
                // Slate ocean
                cr = 113; cg = 155; cb = 174;
            } else {
                const ei = elev > 4500 ? 4500 : (elev | 0);
                cr = lutR[ei]; cg = lutG[ei]; cb = lutB[ei];
            }

            const xLeft = px > 0 ? px - 1 : 0;
            const xRight = px < W - 1 ? px + 1 : W - 1;

            const dx = (rawElevBuffer[yRow + xRight] - rawElevBuffer[yRow + xLeft]) * invCell2;
            const dy = (rawElevBuffer[yBelow + px] - rawElevBuffer[yAbove + px]) * invCell2;

            const Nx = -dx * z_factor;
            const Ny = -dy * z_factor;
            const invLen = 1.0 / Math.sqrt(Nx * Nx + Ny * Ny + 1.0);
            const cosInc = (0.5 * z_factor * (dx + dy) + 0.70710678) * invLen;
            const shade = cosInc > 0 ? cosInc : 0;
            const intensity = 0.35 + 0.65 * shade;

            const outIdx = idx << 2;
            out[outIdx]     = (cr * intensity + 0.5) | 0;
            out[outIdx + 1] = (cg * intensity + 0.5) | 0;
            out[outIdx + 2] = (cb * intensity + 0.5) | 0;
            out[outIdx + 3] = 255;
        }
    }
    return outImgData;
}

// Custom Leaflet GridLayer for dynamic Terrarium DEM
L.GridLayer.TerrariumDEM = L.GridLayer.extend({
    createTile: function (coords, done) {
        const tile = document.createElement('canvas');
        tile.width = 256;
        tile.height = 256;
        const ctx = tile.getContext('2d');

        const key = `${coords.z}/${coords.x}/${coords.y}`;
        if (demTileCache.has(key)) {
            ctx.putImageData(demTileCache.get(key), 0, 0);
            setTimeout(() => done(null, tile), 0);
            return tile;
        }

        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.onload = () => {
            try {
                const c = document.createElement('canvas');
                c.width = 256;
                c.height = 256;
                const cCtx = c.getContext('2d', { willReadFrequently: true });
                cCtx.drawImage(img, 0, 0);
                const srcData = cCtx.getImageData(0, 0, 256, 256).data;
                const shaded = shadeTerrariumImageData(srcData, coords.z, coords.y);
                ctx.putImageData(shaded, 0, 0);

                if (demTileCache.size >= MAX_DEM_CACHE) {
                    const firstKey = demTileCache.keys().next().value;
                    demTileCache.delete(firstKey);
                }
                demTileCache.set(key, shaded);

                done(null, tile);
            } catch (err) {
                done(err, tile);
            }
        };
        img.onerror = () => {
            ctx.fillStyle = '#719BAE';
            ctx.fillRect(0, 0, 256, 256);
            done(null, tile);
        };
        img.src = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${coords.z}/${coords.x}/${coords.y}.png`;
        return tile;
    }
});

// Basemap Providers
const BASEMAP_LAYERS = {
    'western-dem': {
        type: 'grid',
        layer: new L.GridLayer.TerrariumDEM({ minZoom: 4, maxZoom: 15, maxNativeZoom: 14 }),
        bgColor: '#719BAE'
    },
    'usgs-relief': {
        type: 'tile',
        layer: L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'USGS The National Map 3DEP',
            maxZoom: 16
        }),
        bgColor: '#111827'
    },
    'satellite': {
        type: 'tile',
        layer: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye',
            maxZoom: 18
        }),
        bgColor: '#111827'
    },
    'opentopo': {
        type: 'tile',
        layer: L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: 'Map data: &copy; OpenStreetMap, SRTM | Map style: &copy; OpenTopoMap',
            maxZoom: 17
        }),
        bgColor: '#111827'
    },
    'dark': {
        type: 'tile',
        layer: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
            maxZoom: 16
        }),
        bgColor: '#111827'
    }
};

let map = null;
let roadsLayer = null;
let labelsLayer = null;
let tracksLayerGroup = null;

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initUIEventListeners();
    initDropzone();
    updateStatsSummary();
    updateExportMetaInfo();
    initCropFrameHandles();
});

function initMap() {
    map = L.map('map', {
        center: REGION_PRESETS.california.center,
        zoom: REGION_PRESETS.california.zoom,
        zoomControl: false,
        preferCanvas: true,
        zoomSnap: 0.1,             // Fine fractional zoom step (0.1 increments instead of 1.0)
        zoomDelta: 0.25,            // Smooth button zooming
        wheelPxPerZoomLevel: 120    // Smooth trackpad and mouse wheel zooming
    });

    // Add Zoom Control at top-right
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Reference Labels & Roads Pane (zIndex 350: sits above basemap at 200, below vector tracks at 400)
    map.createPane('labelsPane');
    map.getPane('labelsPane').style.zIndex = 350;
    map.getPane('labelsPane').style.pointerEvents = 'none';

    // High-Resolution Retina Roads & Highways Layer (Carto Voyager @2x)
    roadsLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}@2x.png', {
        attribution: 'Roads &copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        pane: 'labelsPane',
        maxZoom: 20,
        opacity: state.labelsOpacity
    }).addTo(map);

    // High-Resolution Retina Place Names & Boundaries Layer (Carto Voyager @2x)
    labelsLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}@2x.png', {
        attribution: 'Reference &copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        pane: 'labelsPane',
        maxZoom: 20,
        opacity: state.labelsOpacity
    }).addTo(map);

    // Track Layer Group (uses Canvas renderer for fast rendering of thousands of coordinates)
    tracksLayerGroup = L.layerGroup().addTo(map);

    // Set initial native DEM basemap
    setBasemap(state.activeBasemap);

    // Update poster crop frame on map pan, zoom, and container resize
    map.on('move moveend resize zoomend', updatePosterFrame);
}

function setBasemap(layerKey) {
    const prev = BASEMAP_LAYERS[state.activeBasemap];
    if (prev && map.hasLayer(prev.layer)) {
        map.removeLayer(prev.layer);
    }

    const next = BASEMAP_LAYERS[layerKey];
    if (!next) return;

    state.activeBasemap = layerKey;
    next.layer.addTo(map);

    const mapEl = document.getElementById('map');
    if (mapEl && next.bgColor) {
        mapEl.style.backgroundColor = next.bgColor;
    }
}

// ============================================================
// UI Event Listeners
// ============================================================

function initUIEventListeners() {
    // Basemap Switcher
    document.querySelectorAll('.basemap-card').forEach(card => {
        card.addEventListener('click', () => {
            const layerKey = card.dataset.layer;
            if (layerKey === state.activeBasemap) return;

            document.querySelectorAll('.basemap-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');

            setBasemap(layerKey);
        });
    });

    // Roads & Place Names Reference Overlay Controls
    const toggleLabels = document.getElementById('toggle-labels');
    const labelsSlider = document.getElementById('labels-opacity-slider');
    const labelsVal = document.getElementById('labels-opacity-val');

    if (toggleLabels) {
        toggleLabels.addEventListener('change', (e) => {
            state.showLabels = e.target.checked;
            const op = state.showLabels ? state.labelsOpacity : 0;
            if (roadsLayer) roadsLayer.setOpacity(op);
            if (labelsLayer) labelsLayer.setOpacity(op);
        });
    }

    if (labelsSlider) {
        labelsSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            state.labelsOpacity = val / 100;
            if (labelsVal) {
                labelsVal.textContent = `${val}%`;
            }
            if (state.showLabels) {
                if (roadsLayer) roadsLayer.setOpacity(state.labelsOpacity);
                if (labelsLayer) labelsLayer.setOpacity(state.labelsOpacity);
            }
        });
    }

    // Quick Region Jumps
    document.querySelectorAll('.region-btn[data-region]').forEach(btn => {
        btn.addEventListener('click', () => {
            const preset = REGION_PRESETS[btn.dataset.region];
            if (preset) {
                map.flyTo(preset.center, preset.zoom, { duration: 1.2 });
            }
        });
    });

    document.getElementById('btn-zoom-fit').addEventListener('click', fitMapToActiveTracks);

    // Test Mode & Sampling Slider
    const sampleSlider = document.getElementById('sample-slider');
    const samplePctVal = document.getElementById('sample-pct-val');

    sampleSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        state.sampleFraction = val / 100;
        samplePctVal.textContent = `${val}%`;

        // Update preset chip styling
        document.querySelectorAll('.chip-btn[data-sample]').forEach(chip => {
            chip.classList.toggle('active', parseInt(chip.dataset.sample, 10) === val);
        });

        applySamplingAndRender();
    });

    // Quick Preset Chips (10%, 25%, 50%, 100%)
    document.querySelectorAll('.chip-btn[data-sample]').forEach(chip => {
        chip.addEventListener('click', () => {
            const val = parseInt(chip.dataset.sample, 10);
            sampleSlider.value = val;
            state.sampleFraction = val / 100;
            samplePctVal.textContent = `${val}%`;

            document.querySelectorAll('.chip-btn[data-sample]').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            applySamplingAndRender();
        });
    });

    // Track Style Selector (Glow, Altitude, Vario, Distinct)
    document.querySelectorAll('.style-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentStyle = btn.dataset.style;

            // Toggle glow control visibility
            const glowCtrl = document.getElementById('glow-control');
            if (glowCtrl) {
                glowCtrl.style.display = (state.currentStyle === 'glow') ? 'block' : 'none';
            }

            // Toggle Altitude Legend
            const altLegend = document.getElementById('altitude-legend');
            if (altLegend) {
                altLegend.style.display = (state.currentStyle === 'altitude') ? 'block' : 'none';
            }

            renderTracks();
        });
    });

    // Track Styling Controls
    const colorPicker = document.getElementById('track-color-picker');
    const colorHex = document.getElementById('track-color-hex');
    colorPicker.addEventListener('input', (e) => {
        state.trackColor = e.target.value;
        colorHex.textContent = e.target.value.toUpperCase();
        renderTracks();
    });

    const lineWidthSlider = document.getElementById('line-width-slider');
    const lineWidthVal = document.getElementById('line-width-val');
    lineWidthSlider.addEventListener('input', (e) => {
        state.lineWidth = parseFloat(e.target.value);
        lineWidthVal.textContent = `${state.lineWidth}px`;
        renderTracks();
    });

    const glowRadiusSlider = document.getElementById('glow-radius-slider');
    const glowRadiusVal = document.getElementById('glow-radius-val');
    glowRadiusSlider.addEventListener('input', (e) => {
        state.glowRadius = parseInt(e.target.value, 10);
        glowRadiusVal.textContent = `${state.glowRadius}px`;
        renderTracks();
    });

    const opacitySlider = document.getElementById('opacity-slider');
    const opacityVal = document.getElementById('opacity-val');
    opacitySlider.addEventListener('input', (e) => {
        state.opacity = parseInt(e.target.value, 10) / 100;
        opacityVal.textContent = `${e.target.value}%`;
        renderTracks();
    });

    // File Input Browse Buttons
    const fileInput = document.getElementById('file-input');
    const folderInput = document.getElementById('folder-input');
    document.getElementById('btn-browse-files').addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    document.getElementById('btn-browse-folder').addEventListener('click', (e) => {
        e.stopPropagation();
        folderInput.click();
    });

    const demoBtn = document.getElementById('btn-load-demo');
    if (demoBtn) {
        demoBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            loadDemoTracks();
        });
    }

    fileInput.addEventListener('change', (e) => handleFileList(e.target.files));
    folderInput.addEventListener('change', (e) => handleFileList(e.target.files));

    // Flight List Search
    const searchInput = document.getElementById('flight-search');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        document.querySelectorAll('.flight-item').forEach(item => {
            const text = item.textContent.toLowerCase();
            item.style.display = text.includes(query) ? 'flex' : 'none';
        });
    });

    // Toggle All Visibility
    document.getElementById('btn-toggle-all').addEventListener('click', () => {
        const anyVisible = state.allTracks.some(t => t.visible);
        state.allTracks.forEach(t => t.visible = !anyVisible);
        renderFlightList();
        renderTracks();
    });

    // Clear Tracks
    document.getElementById('btn-clear-tracks').addEventListener('click', () => {
        if (state.allTracks.length === 0) return;
        if (confirm('Clear all loaded flight tracks?')) {
            state.allTracks = [];
            state.activeTracks = [];
            renderFlightList();
            renderTracks();
            updateStatsSummary();
        }
    });

    // Export Poster UI Controls
    const sizeSelect = document.getElementById('export-size-select');
    if (sizeSelect) {
        sizeSelect.addEventListener('change', (e) => {
            state.exportSize = e.target.value;
            updateExportMetaInfo();
            updatePosterFrame();
        });
    }

    const dpiSelect = document.getElementById('export-dpi-select');
    if (dpiSelect) {
        dpiSelect.addEventListener('change', (e) => {
            state.exportDpi = parseInt(e.target.value, 10) || 300;
            updateExportMetaInfo();
            updatePosterFrame();
        });
    }

    const orientGroup = document.getElementById('export-orient-group');
    if (orientGroup) {
        orientGroup.querySelectorAll('.segment-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                orientGroup.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.exportOrientation = btn.getAttribute('data-orient');
                updateExportMetaInfo();
                updatePosterFrame();
            });
        });
    }

    // Poster Framing & Export Mode Buttons
    const frameBtn = document.getElementById('btn-adjust-frame');
    if (frameBtn) {
        frameBtn.addEventListener('click', () => {
            if (state.showCropFrame) {
                exitFramingMode();
            } else {
                enterFramingMode();
            }
        });
    }

    const cancelFramingBtn = document.getElementById('btn-cancel-framing');
    if (cancelFramingBtn) {
        cancelFramingBtn.addEventListener('click', exitFramingMode);
    }

    const confirmExportBtn = document.getElementById('btn-confirm-export');
    if (confirmExportBtn) {
        confirmExportBtn.addEventListener('click', exportMapPoster);
    }

    // Export Poster Button from Sidebar
    const exportBtn = document.getElementById('btn-export-poster');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            if (!state.showCropFrame) {
                enterFramingMode();
            } else {
                exportMapPoster();
            }
        });
    }

    // Mobile Sidebar Toggle
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    const toggleBtn = document.getElementById('btn-sidebar-toggle');

    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        backdrop.classList.toggle('active');
    });

    backdrop.addEventListener('click', () => {
        sidebar.classList.remove('open');
        backdrop.classList.remove('active');
    });
}

// ============================================================
// Drag & Drop Handling
// ============================================================

function initDropzone() {
    const dropzone = document.getElementById('dropzone');

    // Prevent default browser behavior across the entire window so dropped files never open in browser tabs
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        window.addEventListener(eventName, (e) => {
            e.preventDefault();
        }, false);
        document.body.addEventListener(eventName, (e) => {
            e.preventDefault();
        }, false);
    });

    // Dropzone highlight styling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileList(dt.files);
        }
    });

    // Clicking anywhere in dropzone (except on buttons or inputs) opens file selector
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('button') || e.target.closest('input')) {
            return;
        }
        document.getElementById('file-input').click();
    });
}

// ============================================================
// IGC File Parsing Engine
// ============================================================

async function handleFileList(fileList) {
    const files = Array.from(fileList).filter(f => f.name.toLowerCase().endsWith('.igc'));
    if (files.length === 0) {
        alert('No .igc files found. Please select valid flight records.');
        return;
    }

    const progressContainer = document.getElementById('load-progress-container');
    const progressBar = document.getElementById('load-progress-bar');
    const progressText = document.getElementById('load-progress-text');
    const statusBadge = document.getElementById('status-badge');

    progressContainer.style.display = 'block';
    statusBadge.textContent = 'PARSING...';
    statusBadge.style.color = 'var(--warning)';

    let parsedCount = 0;
    const newTracks = [];

    // Parse in asynchronous chunks to keep UI responsive
    const CHUNK_SIZE = 15;
    for (let i = 0; i < files.length; i += CHUNK_SIZE) {
        const chunk = files.slice(i, i + CHUNK_SIZE);
        const chunkPromises = chunk.map(file => parseSingleIGC(file));
        const results = await Promise.all(chunkPromises);

        results.forEach(res => {
            if (res && res.points.length > 5) {
                newTracks.push(res);
            }
        });

        parsedCount += chunk.length;
        const pct = Math.round((parsedCount / files.length) * 100);
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `Parsing ${parsedCount}/${files.length} (${pct}%)`;

        // Yield to browser event loop
        await new Promise(r => setTimeout(r, 0));
    }

    progressContainer.style.display = 'none';
    statusBadge.textContent = 'READY';
    statusBadge.style.color = 'var(--success)';

    // Append and deduplicate by filename
    const existingNames = new Set(state.allTracks.map(t => t.filename));
    newTracks.forEach(t => {
        if (!existingNames.has(t.filename)) {
            state.allTracks.push(t);
        }
    });

    applySamplingAndRender();
    renderFlightList();
    updateStatsSummary();
    fitMapToActiveTracks();
}

function parseSingleIGC(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const parsed = parseIGCText(content, file.name);
            resolve(parsed);
        };
        reader.onerror = () => resolve(null);
        reader.readAsText(file);
    });
}

function parseIGCText(text, filename) {
    const lines = text.split('\n');
    const points = [];
    const latlngs = [];
    let dateStr = '';
    let pilot = '';
    let glider = '';
    let prevCoord = null;
    let totalDistKm = 0;
    let maxAlt = 0;
    let minAlt = Infinity;

    const MAX_DEG_JUMP = 0.5; // Cleans GPS glitches

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        // Header: Date
        if (line.startsWith('HFDTE') || line.startsWith('HFDTEDATE:')) {
            const dMatch = line.match(/\d{6}/);
            if (dMatch) {
                const raw = dMatch[0];
                dateStr = `20${raw.slice(4,6)}-${raw.slice(2,4)}-${raw.slice(0,2)}`;
            }
        } else if (line.startsWith('HFPLTPILOT:') || line.startsWith('HFPLT')) {
            pilot = line.split(':')[1]?.trim() || '';
        } else if (line.startsWith('HFGTYGLIDERTYPE:') || line.startsWith('HFGTY')) {
            glider = line.split(':')[1]?.trim() || '';
        }

        // B Record: B HHMMSS DDMMmmmN DDDMMmmmE A PPPPP GGGGG
        if (line[0] === 'B' && line.length >= 35) {
            const validity = line[24];
            if (validity !== 'A') continue; // Skip invalid GPS fixes

            const latStr = line.slice(7, 15);
            const lonStr = line.slice(15, 24);

            const lat = parseIgcCoord(latStr, true);
            const lon = parseIgcCoord(lonStr, false);

            if (lat === null || lon === null) continue;

            if (prevCoord) {
                const dLat = lat - prevCoord[0];
                const dLon = lon - prevCoord[1];
                if (Math.hypot(dLat, dLon) > MAX_DEG_JUMP) continue; // Skip teleport jump
                totalDistKm += vincentyDistance(prevCoord[0], prevCoord[1], lat, lon);
            }

            const pressAlt = parseInt(line.slice(25, 30), 10) || 0;
            const gpsAlt = parseInt(line.slice(30, 35), 10) || pressAlt;

            if (gpsAlt > maxAlt) maxAlt = gpsAlt;
            if (gpsAlt < minAlt) minAlt = gpsAlt;

            const timeStr = line.slice(1, 7); // HHMMSS

            points.push({
                lat,
                lon,
                alt: gpsAlt,
                time: timeStr
            });
            latlngs.push([lat, lon]);
            prevCoord = [lat, lon];
        }
    }

    if (points.length < 5) return null;

    // Calculate duration with midnight UTC rollover handling
    let durationSec = 0;
    if (points.length >= 2) {
        const t0 = parseTimeSec(points[0].time);
        const t1 = parseTimeSec(points[points.length - 1].time);
        let diff = t1 - t0;
        if (diff < 0) {
            diff += 86400; // Crossed midnight UTC
        }
        durationSec = diff;
    }

    // Calculate official XContest max distance over up to 5 points (XC Simulator algorithm)
    const xcDistKm = calculateXContest5PointDistance(points);

    return {
        id: 'trk_' + Math.random().toString(36).slice(2, 9),
        filename,
        date: dateStr || extractDateFromFilename(filename),
        pilot,
        glider,
        points,
        latlngs,
        distanceKm: xcDistKm > 0 ? xcDistKm : (Math.round(totalDistKm * 10) / 10),
        odometerKm: Math.round(totalDistKm * 10) / 10,
        maxAltM: maxAlt,
        minAltM: (minAlt === Infinity) ? 0 : minAlt,
        durationSec,
        color: generateHarmonicColor(),
        visible: true
    };
}

function parseIgcCoord(str, isLat) {
    try {
        if (isLat) {
            const deg = parseInt(str.slice(0, 2), 10);
            const min = parseInt(str.slice(2, 4), 10);
            const dec = parseInt(str.slice(4, 7), 10) / 1000.0;
            const sign = str[7] === 'N' ? 1 : -1;
            return sign * (deg + (min + dec) / 60.0);
        } else {
            const deg = parseInt(str.slice(0, 3), 10);
            const min = parseInt(str.slice(3, 5), 10);
            const dec = parseInt(str.slice(5, 8), 10) / 1000.0;
            const sign = str[8] === 'E' ? 1 : -1;
            return sign * (deg + (min + dec) / 60.0);
        }
    } catch {
        return null;
    }
}

function parseTimeSec(hhmmss) {
    if (!hhmmss || hhmmss.length < 6) return 0;
    const h = parseInt(hhmmss.slice(0, 2), 10) || 0;
    const m = parseInt(hhmmss.slice(2, 4), 10) || 0;
    const s = parseInt(hhmmss.slice(4, 6), 10) || 0;
    return h * 3600 + m * 60 + s;
}

function extractDateFromFilename(name) {
    const match = name.match(/\d{4}[-_]\d{2}[-_]\d{2}/);
    return match ? match[0].replace(/_/g, '-') : 'Unknown';
}

/**
 * Vincenty's Inverse Formula to compute geodesic distance on the WGS-84 ellipsoid.
 * Adopted directly from the XC Simulator app for millimeter-accurate geodesic distances.
 * @param {Number|Object} p1 - First point {lat, lng} or lat1
 * @param {Number|Object} p2 - Second point {lat, lng} or lon1
 * @param {Number} [lat2] - lat2 (if passing 4 numbers)
 * @param {Number} [lon2] - lon2 (if passing 4 numbers)
 * @returns {Number} Geodesic distance in kilometers
 */
function vincentyDistance(p1, p2, lat2, lon2) {
    let lat1_val, lon1_val, lat2_val, lon2_val;
    if (lat2 !== undefined && lon2 !== undefined) {
        lat1_val = p1;
        lon1_val = p2;
        lat2_val = lat2;
        lon2_val = lon2;
    } else {
        if (!p1 || !p2) return 0;
        lat1_val = (p1.lat !== undefined ? p1.lat : p1[0]);
        lon1_val = (p1.lng !== undefined ? p1.lng : (p1.lon !== undefined ? p1.lon : p1[1]));
        lat2_val = (p2.lat !== undefined ? p2.lat : p2[0]);
        lon2_val = (p2.lng !== undefined ? p2.lng : (p2.lon !== undefined ? p2.lon : p2[1]));
    }

    if (lat1_val === lat2_val && lon1_val === lon2_val) return 0;

    const lat1 = lat1_val * Math.PI / 180;
    const lon1 = lon1_val * Math.PI / 180;
    const lat2_lat = lat2_val * Math.PI / 180;
    const lon2_lon = lon2_val * Math.PI / 180;

    const a = 6378137.0;          // WGS-84 semi-major axis (meters)
    const f = 1 / 298.257223563;  // WGS-84 flattening
    const b = 6356752.314245;     // WGS-84 semi-minor axis (meters)

    const L = lon2_lon - lon1;
    const U1 = Math.atan((1 - f) * Math.tan(lat1));
    const U2 = Math.atan((1 - f) * Math.tan(lat2_lat));

    const sinU1 = Math.sin(U1), cosU1 = Math.cos(U1);
    const sinU2 = Math.sin(U2), cosU2 = Math.cos(U2);

    let lambda = L;
    let lambdaP;
    let iterLimit = 100;
    let cosSqAlpha = 0;
    let cos2SigmaM = 0;
    let sinSigma = 0;
    let cosSigma = 0;
    let sigma = 0;

    do {
        const sinLambda = Math.sin(lambda);
        const cosLambda = Math.cos(lambda);
        sinSigma = Math.sqrt((cosU2 * sinLambda) * (cosU2 * sinLambda) +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) * (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda));
        if (sinSigma === 0) return 0; // co-incident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
        sigma = Math.atan2(sinSigma, cosSigma);

        const sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
        cosSqAlpha = 1 - sinAlpha * sinAlpha;
        cos2SigmaM = (cosSqAlpha === 0) ? 0 : cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha;

        const C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha));
        lambdaP = lambda;
        lambda = L + (1 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM))
        );
    } while (Math.abs(lambda - lambdaP) > 1e-12 && --iterLimit > 0);

    if (iterLimit === 0) {
        // Fallback: Haversine on WGS-84 mean radius
        const dLat = (lat2_val - lat1_val) * Math.PI / 180;
        const dLon = (lon2_val - lon1_val) * Math.PI / 180;
        const a_h = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                    Math.cos(lat1) * Math.cos(lat2_lat) *
                    Math.sin(dLon / 2) * Math.sin(dLon / 2);
        return 6378.137 * 2 * Math.atan2(Math.sqrt(a_h), Math.sqrt(1 - a_h));
    }

    const uSq = cosSqAlpha * (a * a - b * b) / (b * b);
    const A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)));
    const B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)));
    const deltaSigma = B * sinSigma * (
        cos2SigmaM + B / 4 * (
            cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
            B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)
        )
    );

    const s = b * A * (sigma - deltaSigma);
    return s / 1000; // in kilometers
}

// Alias for backward compatibility
const haversineKm = vincentyDistance;

// ============================================================
// XContest 5-Point Distance Optimization Algorithm
// (From XC Simulator: DP turnpoint search + triangle analysis + refinement)
// ============================================================

function rdpSimplify(points, epsilon) {
    if (!points || points.length <= 2) return points;

    let sumLat = 0;
    for (let i = 0; i < points.length; i++) {
        sumLat += points[i].lat;
    }
    const avgLat = sumLat / points.length;
    const cosAvgLat = Math.cos(avgLat * Math.PI / 180);

    function distanceToSegmentFlat(p, p1, p2) {
        const x = p.lng !== undefined ? p.lng : p.lon;
        const y = p.lat;
        const x1 = p1.lng !== undefined ? p1.lng : p1.lon;
        const y1 = p1.lat;
        const x2 = p2.lng !== undefined ? p2.lng : p2.lon;
        const y2 = p2.lat;

        const A = x - x1;
        const B = y - y1;
        const C = x2 - x1;
        const D = y2 - y1;

        const dot = A * C + B * D;
        const len_sq = C * C + D * D;
        let param = len_sq !== 0 ? dot / len_sq : -1;

        let xx, yy;
        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }

        const dLat = (y - yy) * 111.12;
        const dLng = (x - xx) * 111.12 * cosAvgLat;
        return Math.sqrt(dLat * dLat + dLng * dLng);
    }

    function simplify(pts) {
        if (pts.length <= 2) return pts;

        let maxDist = 0;
        let index = 0;
        const end = pts.length - 1;

        for (let i = 1; i < end; i++) {
            const dist = distanceToSegmentFlat(pts[i], pts[0], pts[end]);
            if (dist > maxDist) {
                maxDist = dist;
                index = i;
            }
        }

        if (maxDist > epsilon) {
            const results1 = simplify(pts.slice(0, index + 1));
            const results2 = simplify(pts.slice(index));
            return results1.slice(0, results1.length - 1).concat(results2);
        } else {
            return [pts[0], pts[end]];
        }
    }

    return simplify(points);
}

function optimizeTrack(points) {
    const N = points.length;
    if (N < 2) return null;

    const dist = Array(N).fill(0).map(() => Array(N).fill(0));
    for (let i = 0; i < N; i++) {
        for (let j = i; j < N; j++) {
            const d = vincentyDistance(points[i], points[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    let bestScore = 0;
    let bestType = 'free';
    let bestIndices = [];
    let bestLegLengths = [];
    let bestGap = 0;
    let bestGapPercent = 0;
    let bestScoredDist = 0;
    let bestFreeDist = 0;

    // --- OPTION 1: FREE FLIGHT (up to 3 turnpoints, i.e., up to 4 segments) ---
    const dp = Array(5).fill(0).map(() => Array(N).fill(0));
    const parent = Array(5).fill(0).map(() => Array(N).fill(-1));

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < i; j++) {
            if (dist[j][i] > dp[1][i]) {
                dp[1][i] = dist[j][i];
                parent[1][i] = j;
            }
        }
    }

    for (let k = 2; k <= 4; k++) {
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < i; j++) {
                if (dp[k - 1][j] + dist[j][i] > dp[k][i]) {
                    dp[k][i] = dp[k - 1][j] + dist[j][i];
                    parent[k][i] = j;
                }
            }
        }
    }

    let bestFreeEnd = -1;
    let bestFreeK = 1;
    for (let k = 1; k <= 4; k++) {
        for (let i = 0; i < N; i++) {
            if (dp[k][i] > bestFreeDist) {
                bestFreeDist = dp[k][i];
                bestFreeEnd = i;
                bestFreeK = k;
            }
        }
    }

    if (bestFreeEnd !== -1) {
        let curr = bestFreeEnd;
        let k = bestFreeK;
        const indices = [];
        while (curr !== -1) {
            indices.unshift(curr);
            curr = parent[k][curr];
            k--;
        }
        bestScore = bestFreeDist * 1.0;
        bestType = 'free';
        bestIndices = indices;
        bestLegLengths = indices.slice(1).map((idx, i) => dist[indices[i]][idx]);
    }

    // --- OPTION 2: TRIANGLES (Flat / FAI) ---
    const dpGap = Array(N).fill(null).map(() => Array(N).fill(Infinity));
    const parent_is = Array(N).fill(null).map(() => Array(N).fill(-1));

    for (let hf = 0; hf < N; hf++) {
        dpGap[0][hf] = dist[0][hf];
        parent_is[0][hf] = 0;
        for (let i1 = 1; i1 < N; i1++) {
            if (dist[i1][hf] < dpGap[i1 - 1][hf]) {
                dpGap[i1][hf] = dist[i1][hf];
                parent_is[i1][hf] = i1;
            } else {
                dpGap[i1][hf] = dpGap[i1 - 1][hf];
                parent_is[i1][hf] = parent_is[i1 - 1][hf];
            }
        }
    }

    const minGap = Array(N).fill(null).map(() => Array(N).fill(null));
    for (let i1 = 0; i1 < N; i1++) {
        let minG = dpGap[i1][N - 1];
        let bestIs = parent_is[i1][N - 1];
        let bestIf = N - 1;
        minGap[i1][N - 1] = { val: minG, is: bestIs, if: bestIf };

        for (let i3 = N - 2; i3 > i1; i3--) {
            if (dpGap[i1][i3] < minG) {
                minG = dpGap[i1][i3];
                bestIs = parent_is[i1][i3];
                bestIf = i3;
            }
            minGap[i1][i3] = { val: minG, is: bestIs, if: bestIf };
        }
    }

    for (let i1 = 0; i1 < N; i1++) {
        for (let i2 = i1 + 1; i2 < N; i2++) {
            for (let i3 = i2 + 1; i3 < N; i3++) {
                const P = dist[i1][i2] + dist[i2][i3] + dist[i3][i1];
                const gapData = minGap[i1][i3];
                if (!gapData) continue;
                const g = gapData.val;

                if (g <= 0.20 * P) {
                    const scoredDist = P - g;
                    const s1 = dist[i1][i2];
                    const s2 = dist[i2][i3];
                    const s3 = dist[i3][i1];
                    const shortestLeg = Math.min(s1, s2, s3);

                    const isFai = shortestLeg >= 0.28 * P;
                    const isClosed = (g / P) < 0.05;

                    let coeff = 1.0;
                    let type = 'free_tri';

                    if (isFai && isClosed) {
                        coeff = 1.60;
                        type = 'closed_fai';
                    } else if (isFai && !isClosed) {
                        coeff = 1.40;
                        type = 'fai';
                    } else if (!isFai && isClosed) {
                        coeff = 1.40;
                        type = 'closed_free';
                    } else {
                        coeff = 1.20;
                        type = 'free_tri';
                    }

                    const score = scoredDist * coeff;

                    if (score > bestScore) {
                        bestScore = score;
                        bestType = type;
                        bestIndices = [gapData.is, i1, i2, i3, gapData.if];
                        bestLegLengths = [s1, s2, s3];
                        bestGap = g;
                        bestGapPercent = (g / P) * 100;
                        bestScoredDist = scoredDist;
                    }
                }
            }
        }
    }

    return {
        score: bestScore,
        distance: bestType === 'free' ? bestFreeDist : bestScoredDist,
        type: bestType,
        indices: bestIndices,
        legLengths: bestLegLengths,
        gap: bestGap,
        gapPercent: bestGapPercent
    };
}

function refineOptimizedFlight(rawPoints, optResult, mappingToRaw) {
    const simplifiedIndices = optResult.indices;
    if (!simplifiedIndices || simplifiedIndices.length < 2) {
        return optResult;
    }

    const M = mappingToRaw.length;
    const W = 4;

    if (optResult.type === 'free') {
        const K = simplifiedIndices.length;
        const currIndices = simplifiedIndices.map(idx => mappingToRaw[idx]);

        const minRaw = [];
        const maxRaw = [];
        for (let j = 0; j < K; j++) {
            const idx = simplifiedIndices[j];
            minRaw.push(mappingToRaw[Math.max(0, idx - W)]);
            maxRaw.push(mappingToRaw[Math.min(M - 1, idx + W)]);
        }

        for (let iter = 0; iter < 5; iter++) {
            for (let j = 0; j < K; j++) {
                let bestIdx = currIndices[j];
                let maxDistSum = -1;

                const startRange = minRaw[j];
                const endRange = maxRaw[j];

                const lowerBound = j > 0 ? currIndices[j - 1] + 1 : startRange;
                const upperBound = j < K - 1 ? currIndices[j + 1] - 1 : endRange;

                for (let i = Math.max(startRange, lowerBound); i <= Math.min(endRange, upperBound); i++) {
                    let distSum = 0;
                    if (j > 0) {
                        distSum += vincentyDistance(rawPoints[currIndices[j - 1]], rawPoints[i]);
                    }
                    if (j < K - 1) {
                        distSum += vincentyDistance(rawPoints[i], rawPoints[currIndices[j + 1]]);
                    }

                    if (distSum > maxDistSum) {
                        maxDistSum = distSum;
                        bestIdx = i;
                    }
                }
                currIndices[j] = bestIdx;
            }
        }

        const refinedLegs = [];
        let totalDist = 0;
        for (let j = 0; j < K - 1; j++) {
            const d = vincentyDistance(rawPoints[currIndices[j]], rawPoints[currIndices[j + 1]]);
            refinedLegs.push(d);
            totalDist += d;
        }

        return {
            score: totalDist * 1.0,
            distance: totalDist,
            type: 'free',
            indices: currIndices,
            legLengths: refinedLegs,
            gap: 0,
            gapPercent: 0,
            refinedPoints: currIndices.map(idx => rawPoints[idx])
        };
    } else {
        if (simplifiedIndices.length < 5) return optResult;

        const idx_s = simplifiedIndices[0];
        const idx_1 = simplifiedIndices[1];
        const idx_2 = simplifiedIndices[2];
        const idx_3 = simplifiedIndices[3];
        const idx_f = simplifiedIndices[4];

        const r_s_min = mappingToRaw[Math.max(0, idx_s - W)];
        const r_s_max = mappingToRaw[Math.min(M - 1, idx_s + W)];

        const r_1_min = mappingToRaw[Math.max(0, idx_1 - W)];
        const r_1_max = mappingToRaw[Math.min(M - 1, idx_1 + W)];

        const r_2_min = mappingToRaw[Math.max(0, idx_2 - W)];
        const r_2_max = mappingToRaw[Math.min(M - 1, idx_2 + W)];

        const r_3_min = mappingToRaw[Math.max(0, idx_3 - W)];
        const r_3_max = mappingToRaw[Math.min(M - 1, idx_3 + W)];

        const r_f_min = mappingToRaw[Math.max(0, idx_f - W)];
        const r_f_max = mappingToRaw[Math.min(M - 1, idx_f + W)];

        let curr_s = mappingToRaw[idx_s];
        let curr_1 = mappingToRaw[idx_1];
        let curr_2 = mappingToRaw[idx_2];
        let curr_3 = mappingToRaw[idx_3];
        let curr_f = mappingToRaw[idx_f];

        const getScoreForCombo = (s, i1, i2, i3, f) => {
            const d12 = vincentyDistance(rawPoints[i1], rawPoints[i2]);
            const d23 = vincentyDistance(rawPoints[i2], rawPoints[i3]);
            const d31 = vincentyDistance(rawPoints[i3], rawPoints[i1]);
            const P = d12 + d23 + d31;
            const gap = vincentyDistance(rawPoints[s], rawPoints[f]);
            const gapPercent = P > 0 ? (gap / P) * 100 : 999.0;

            const scoredDist = P - gap;
            const shortestLeg = Math.min(d12, d23, d31);
            const isFai = shortestLeg >= 0.28 * P;
            const isClosed = gapPercent < 5.0;

            let coeff = 1.0;
            if (gapPercent <= 20.0) {
                if (isFai && isClosed) {
                    coeff = 1.60;
                } else if (isFai && !isClosed) {
                    coeff = 1.40;
                } else if (!isFai && isClosed) {
                    coeff = 1.40;
                } else {
                    coeff = 1.20;
                }
            }
            return {
                score: scoredDist * coeff,
                distance: scoredDist,
                gap: gap,
                gapPercent: gapPercent,
                legs: [d12, d23, d31]
            };
        };

        for (let iter = 0; iter < 4; iter++) {
            let best_score = -1;
            let best_s = curr_s;
            for (let s = r_s_min; s <= Math.min(r_s_max, curr_1); s++) {
                const res = getScoreForCombo(s, curr_1, curr_2, curr_3, curr_f);
                if (res.score > best_score) { best_score = res.score; best_s = s; }
            }
            curr_s = best_s;

            best_score = -1;
            let best_1 = curr_1;
            for (let i1 = Math.max(r_1_min, curr_s); i1 <= Math.min(r_1_max, curr_2 - 1); i1++) {
                const res = getScoreForCombo(curr_s, i1, curr_2, curr_3, curr_f);
                if (res.score > best_score) { best_score = res.score; best_1 = i1; }
            }
            curr_1 = best_1;

            best_score = -1;
            let best_2 = curr_2;
            for (let i2 = Math.max(r_2_min, curr_1 + 1); i2 <= Math.min(r_2_max, curr_3 - 1); i2++) {
                const res = getScoreForCombo(curr_s, curr_1, i2, curr_3, curr_f);
                if (res.score > best_score) { best_score = res.score; best_2 = i2; }
            }
            curr_2 = best_2;

            best_score = -1;
            let best_3 = curr_3;
            for (let i3 = Math.max(r_3_min, curr_2 + 1); i3 <= Math.min(r_3_max, curr_f); i3++) {
                const res = getScoreForCombo(curr_s, curr_1, curr_2, i3, curr_f);
                if (res.score > best_score) { best_score = res.score; best_3 = i3; }
            }
            curr_3 = best_3;

            best_score = -1;
            let best_f = curr_f;
            for (let f = Math.max(r_f_min, curr_3); f <= r_f_max; f++) {
                const res = getScoreForCombo(curr_s, curr_1, curr_2, curr_3, f);
                if (res.score > best_score) { best_score = res.score; best_f = f; }
            }
            curr_f = best_f;
        }

        const finalCombo = getScoreForCombo(curr_s, curr_1, curr_2, curr_3, curr_f);
        return {
            score: finalCombo.score,
            distance: finalCombo.distance,
            type: optResult.type,
            indices: [curr_s, curr_1, curr_2, curr_3, curr_f],
            legLengths: finalCombo.legs,
            gap: finalCombo.gap,
            gapPercent: finalCombo.gapPercent,
            refinedPoints: [curr_s, curr_1, curr_2, curr_3, curr_f].map(idx => rawPoints[idx])
        };
    }
}

function calculateXContest5PointDistance(rawPoints) {
    if (!rawPoints || rawPoints.length < 2) return 0;
    if (rawPoints.length < 5) {
        let d = 0;
        for (let i = 0; i < rawPoints.length - 1; i++) {
            d += vincentyDistance(rawPoints[i], rawPoints[i + 1]);
        }
        return Math.round(d * 10) / 10;
    }

    let lo = 0.0001;
    let hi = 10.0;
    let simplified = [];
    for (let iter = 0; iter < 12; iter++) {
        let mid = (lo + hi) / 2;
        let testSimp = rdpSimplify(rawPoints, mid);
        if (testSimp.length > 80) {
            lo = mid;
        } else {
            hi = mid;
            simplified = testSimp;
        }
    }
    if (simplified.length < 5) {
        simplified = rdpSimplify(rawPoints, lo);
        if (simplified.length > 80) {
            const factor = Math.ceil(simplified.length / 80);
            simplified = simplified.filter((_, idx) => idx % factor === 0);
        }
    }

    const optCoarse = optimizeTrack(simplified);
    if (!optCoarse || !optCoarse.indices || optCoarse.indices.length < 2) {
        return 0;
    }

    let lastIdx = 0;
    const mappingToRaw = [];
    for (let pt of simplified) {
        let minDist = Infinity;
        let bestIdx = lastIdx;
        const ptLon = pt.lng !== undefined ? pt.lng : pt.lon;
        for (let i = lastIdx; i < rawPoints.length; i++) {
            const rpt = rawPoints[i];
            const rptLon = rpt.lng !== undefined ? rpt.lng : rpt.lon;
            const d = Math.abs(rpt.lat - pt.lat) + Math.abs(rptLon - ptLon);
            if (d < minDist) {
                minDist = d;
                bestIdx = i;
            }
            if (d === 0) break;
        }
        mappingToRaw.push(bestIdx);
        lastIdx = bestIdx;
    }

    const refined = refineOptimizedFlight(rawPoints, optCoarse, mappingToRaw);
    const finalDist = refined && refined.distance ? refined.distance : optCoarse.distance;
    return Math.round(finalDist * 10) / 10;
}

let colorIndex = 0;
function generateHarmonicColor() {
    const goldenRatio = 0.618033988749895;
    colorIndex += goldenRatio;
    colorIndex %= 1;
    const h = Math.floor(colorIndex * 360);
    return `hsl(${h}, 85%, 60%)`;
}

// ============================================================
// Track Rendering Engine
// ============================================================

function applySamplingAndRender() {
    if (state.allTracks.length === 0) {
        state.activeTracks = [];
        renderTracks();
        return;
    }

    const total = state.allTracks.length;
    const count = Math.max(1, Math.round(total * state.sampleFraction));

    // Consistently sample tracks using a pseudo-random slice
    state.activeTracks = state.allTracks.slice(0, count);

    // Update count badge
    document.getElementById('sample-count-badge').textContent = `${state.activeTracks.length} / ${total}`;

    renderTracks();
    updateMapBanner();
}

function renderTracks() {
    if (!tracksLayerGroup) return;
    tracksLayerGroup.clearLayers();

    if (currentHighlightLayer) {
        map.removeLayer(currentHighlightLayer);
        currentHighlightLayer = null;
    }

    const visibleTracks = state.activeTracks.filter(t => t.visible);

    visibleTracks.forEach(track => {
        if (state.currentStyle === 'glow') {
            // Layer 1: Soft outer glow underlay
            L.polyline(track.latlngs, {
                color: state.trackColor,
                weight: state.lineWidth + state.glowRadius * 1.8,
                opacity: state.opacity * 0.3,
                lineCap: 'round',
                lineJoin: 'round',
                interactive: false
            }).addTo(tracksLayerGroup);

            // Layer 2: Sharp foreground vector core
            L.polyline(track.latlngs, {
                color: state.trackColor,
                weight: state.lineWidth,
                opacity: state.opacity,
                lineCap: 'round',
                lineJoin: 'round',
                interactive: false
            }).addTo(tracksLayerGroup);

        } else if (state.currentStyle === 'distinct') {
            // Distinct unique pilot color
            L.polyline(track.latlngs, {
                color: track.color,
                weight: state.lineWidth + 0.5,
                opacity: state.opacity,
                lineCap: 'round',
                lineJoin: 'round',
                interactive: false
            }).addTo(tracksLayerGroup);

        } else if (state.currentStyle === 'altitude') {
            // Altitude gradient colored segments
            renderGradientTrack(track, 'alt');

        } else if (state.currentStyle === 'vario') {
            // Climb/Sink gradient colored segments
            renderGradientTrack(track, 'vario');
        }

        // Dedicated hit-zone polyline for instant, smooth hover detection across all styles
        const hitLine = L.polyline(track.latlngs, {
            color: '#000000',
            weight: Math.max(16, state.lineWidth + 12),
            opacity: 0.0001,
            lineCap: 'round',
            lineJoin: 'round',
            interactive: true
        }).addTo(tracksLayerGroup);

        attachTrackHover(hitLine, track);
    });

    updateMapBanner();
}

function renderGradientTrack(track, mode) {
    const pts = track.points;
    if (pts.length < 2) return;

    // Render grouped segments to maximize Canvas performance
    const SEG_STEP = 4;
    for (let i = 0; i < pts.length - 1; i += SEG_STEP) {
        const segEnd = Math.min(pts.length - 1, i + SEG_STEP);
        const segmentCoords = [];
        for (let j = i; j <= segEnd; j++) {
            segmentCoords.push([pts[j].lat, pts[j].lon]);
        }

        let segColor;
        if (mode === 'alt') {
            const avgAlt = (pts[i].alt + pts[segEnd].alt) / 2;
            segColor = turboColor(avgAlt / 3800.0); // 0m to 3800m scale
        } else {
            // Vario (m/s approximation)
            const dt = Math.max(1, parseTimeSec(pts[segEnd].time) - parseTimeSec(pts[i].time));
            const dz = pts[segEnd].alt - pts[i].alt;
            const climbRate = dz / dt;
            segColor = climbRate >= 0 ? '#10B981' : '#EF4444';
        }

        L.polyline(segmentCoords, {
            color: segColor,
            weight: state.lineWidth + 0.5,
            opacity: state.opacity,
            lineCap: 'round',
            lineJoin: 'round',
            interactive: false
        }).addTo(tracksLayerGroup);
    }
}

// Turbo Colormap Approximation
function turboColor(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.sin(t * Math.PI * 0.9 + 0.2));
    const g = Math.round(255 * Math.sin(t * Math.PI * 1.1 + 0.3));
    const b = Math.round(255 * Math.cos(t * Math.PI * 0.8 + 0.2));
    return `rgb(${Math.max(0, r)}, ${Math.max(0, g)}, ${Math.max(0, b)})`;
}

let currentHighlightLayer = null;

function formatDuration(sec) {
    if (!sec || isNaN(sec) || sec <= 0) return '0 min';
    const hours = Math.floor(sec / 3600);
    const mins = Math.floor((sec % 3600) / 60);
    if (hours > 0) {
        return `${hours} hr ${mins} min`;
    }
    return `${mins} min`;
}

function attachTrackHover(hitLine, track) {
    const durFormatted = formatDuration(track.durationSec);

    const tooltipContent = `
        <div class="track-hover-tooltip">
            <div class="hover-date-badge">📅 ${track.date}</div>
            <div class="hover-filename" title="${track.filename}">${track.filename}</div>
            <div class="hover-stats">
                <span>Flight Time: <strong>${durFormatted}</strong></span><br>
                <span>Distance: <strong>${track.distanceKm} km</strong></span> • <span>Max Alt: <strong>${track.maxAltM.toLocaleString()} m</strong></span>
            </div>
        </div>
    `;

    hitLine.bindTooltip(tooltipContent, {
        sticky: true,
        direction: 'top',
        offset: [0, -12],
        opacity: 1.0,
        className: 'glass-tooltip'
    });

    hitLine.on('mouseover', () => {
        if (currentHighlightLayer) {
            map.removeLayer(currentHighlightLayer);
            currentHighlightLayer = null;
        }

        currentHighlightLayer = L.polyline(track.latlngs, {
            color: '#FFFFFF',
            weight: Math.max(3.5, state.lineWidth + 2.5),
            opacity: 1.0,
            lineCap: 'round',
            lineJoin: 'round',
            interactive: false
        }).addTo(map);

        document.querySelectorAll('.flight-item').forEach(el => el.classList.remove('selected'));
        const listItem = document.getElementById(`item-${track.id}`);
        if (listItem) {
            listItem.classList.add('selected');
            listItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    });

    hitLine.on('mouseout', () => {
        if (currentHighlightLayer) {
            map.removeLayer(currentHighlightLayer);
            currentHighlightLayer = null;
        }
        const listItem = document.getElementById(`item-${track.id}`);
        if (listItem) {
            listItem.classList.remove('selected');
        }
    });

    hitLine.on('click', () => {
        map.fitBounds(hitLine.getBounds(), { padding: [50, 50] });
    });
}

function fitMapToActiveTracks() {
    const visible = state.activeTracks.filter(t => t.visible);
    if (visible.length === 0) return;

    const group = new L.featureGroup(visible.map(t => L.polyline(t.latlngs)));
    map.fitBounds(group.getBounds(), { padding: [50, 50], maxZoom: 14 });
}

// ============================================================
// UI Updates: Flight List, Stats, and Map Banner
// ============================================================

function renderFlightList() {
    const container = document.getElementById('flight-list');
    if (state.allTracks.length === 0) {
        container.innerHTML = '<div class="empty-state">No flights loaded yet. Drag & drop IGC files above.</div>';
        return;
    }

    container.innerHTML = '';
    state.allTracks.forEach(track => {
        const item = document.createElement('div');
        item.className = 'flight-item' + (track.id === state.selectedTrackId ? ' selected' : '');
        item.id = `item-${track.id}`;

        item.innerHTML = `
            <input type="checkbox" ${track.visible ? 'checked' : ''} data-id="${track.id}" style="cursor: pointer;">
            <div class="flight-meta">
                <span class="flight-name" title="${track.filename}">${track.filename}</span>
                <span class="flight-sub">${track.date} • ${track.distanceKm} km • ${formatDuration(track.durationSec)}</span>
            </div>
            <div style="width: 8px; height: 8px; border-radius: 50%; background: ${track.color}; margin-left: 6px;"></div>
        `;

        const chk = item.querySelector('input[type="checkbox"]');
        chk.addEventListener('click', (e) => {
            e.stopPropagation();
            track.visible = chk.checked;
            renderTracks();
        });

        item.addEventListener('mouseenter', () => {
            if (currentHighlightLayer) {
                map.removeLayer(currentHighlightLayer);
                currentHighlightLayer = null;
            }
            currentHighlightLayer = L.polyline(track.latlngs, {
                color: '#FFFFFF',
                weight: Math.max(3.5, state.lineWidth + 2.5),
                opacity: 1.0,
                lineCap: 'round',
                lineJoin: 'round',
                interactive: false
            }).addTo(map);
        });

        item.addEventListener('mouseleave', () => {
            if (currentHighlightLayer) {
                map.removeLayer(currentHighlightLayer);
                currentHighlightLayer = null;
            }
        });

        item.addEventListener('click', () => {
            map.fitBounds(L.polyline(track.latlngs).getBounds(), { padding: [60, 60] });
        });

        container.appendChild(item);
    });
}

function updateStatsSummary() {
    const count = state.allTracks.length;
    const totalPts = state.allTracks.reduce((acc, t) => acc + t.points.length, 0);
    const totalDist = Math.round(state.allTracks.reduce((acc, t) => acc + t.distanceKm, 0));
    const maxAlt = state.allTracks.reduce((acc, t) => Math.max(acc, t.maxAltM), 0);

    document.getElementById('stat-flight-count').textContent = count.toLocaleString();
    document.getElementById('stat-points-count').textContent = totalPts.toLocaleString();
    document.getElementById('stat-total-dist').textContent = `${totalDist.toLocaleString()} km`;
    document.getElementById('stat-max-alt').textContent = `${maxAlt.toLocaleString()} m`;

    updateMapBanner();
}

function updateMapBanner() {
    const banner = document.getElementById('map-banner');
    const title = document.getElementById('banner-title');
    const sub = document.getElementById('banner-sub');

    const activeCount = state.activeTracks.filter(t => t.visible).length;
    const totalCount = state.allTracks.length;

    if (totalCount === 0) {
        title.textContent = 'Flight Tracks Map';
        sub.textContent = 'Drop IGC files to start visualizing';
        return;
    }

    const titleInput = document.getElementById('export-title-input');
    title.textContent = titleInput.value || 'Flight Tracks Map';

    let subText = `${activeCount.toLocaleString()} Flights Rendered`;
    if (activeCount < totalCount) {
        subText += ` (${Math.round(state.sampleFraction * 100)}% Test Sample of ${totalCount})`;
    }
    const totalDist = Math.round(state.activeTracks.reduce((acc, t) => acc + t.distanceKm, 0));
    subText += ` • ${totalDist.toLocaleString()} km Distance`;

    sub.textContent = subText;
}

// ============================================================
// High-Resolution Poster Export
// ============================================================

// Helper: Load an image asynchronously with crossOrigin anonymous
function loadImageAsync(url) {
    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = url;
    });
}

// Helper: Get cached or freshly rendered Terrarium DEM tile ImageData
async function getOrRenderTerrariumTile(x, y, z) {
    const key = `${z}/${x}/${y}`;
    if (demTileCache.has(key)) {
        return demTileCache.get(key);
    }
    const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
    const img = await loadImageAsync(url);
    if (!img) return null;

    try {
        const c = document.createElement('canvas');
        c.width = 256;
        c.height = 256;
        const cCtx = c.getContext('2d', { willReadFrequently: true });
        cCtx.drawImage(img, 0, 0);
        const srcData = cCtx.getImageData(0, 0, 256, 256).data;
        const shaded = shadeTerrariumImageData(srcData, z, y);
        if (demTileCache.size >= MAX_DEM_CACHE) {
            const firstKey = demTileCache.keys().next().value;
            demTileCache.delete(firstKey);
        }
        demTileCache.set(key, shaded);
        return shaded;
    } catch (err) {
        console.warn('Terrarium shading error:', err);
        return null;
    }
}

// Helper: Construct exact tile URL for any layer at arbitrary zoom z (avoiding Leaflet screen-zoom override)
function getLayerTileUrl(layer, x, y, z) {
    if (!layer) return null;
    const urlTemplate = layer._url;
    if (urlTemplate) {
        let sub = 'a';
        if (layer.options && layer.options.subdomains && layer.options.subdomains.length > 0) {
            const subs = layer.options.subdomains;
            sub = subs[Math.abs(x + y) % subs.length];
        }
        return urlTemplate
            .replace('{s}', sub)
            .replace('{z}', z)
            .replace('{x}', x)
            .replace('{y}', y)
            .replace('{r}', '');
    }
    if (typeof layer.getTileUrl === 'function') {
        return layer.getTileUrl({ x, y, z });
    }
    return null;
}

// ============================================================
// Poster Sizing, Framing & High-Resolution Export
// ============================================================

// Precomputed CRC32 Table for PNG chunk checksums
const CRC32_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) {
            c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        }
        table[i] = c;
    }
    return table;
})();

function computeCrc32(buf, start, len) {
    let crc = 0xFFFFFFFF;
    for (let i = start; i < start + len; i++) {
        crc = CRC32_TABLE[(crc ^ buf[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
}

// Embeds standard PNG pHYs chunk so image viewers (e.g. macOS Preview, Photoshop) recognize exact physical inches and print DPI
function embedPngDpi(blob, dpi) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const buffer = reader.result;
                const view = new DataView(buffer);
                const bytes = new Uint8Array(buffer);

                // Check PNG signature: 89 50 4E 47 0D 0A 1A 0A
                if (view.getUint32(0) !== 0x89504E47 || view.getUint32(4) !== 0x0D0A1A0A) {
                    return resolve(blob);
                }

                // IHDR chunk ends at byte offset 33 (8 bytes sig + 25 bytes IHDR chunk)
                const ihdrEnd = 33;
                const ppm = Math.round(dpi / 0.0254); // Pixels per meter

                // 21-byte pHYs chunk:
                // 4 bytes: length (9)
                // 4 bytes: chunk type ('pHYs')
                // 4 bytes: X pixels per unit
                // 4 bytes: Y pixels per unit
                // 1 byte:  unit specifier (1 = meter)
                // 4 bytes: CRC32
                const physChunk = new Uint8Array(21);
                const pView = new DataView(physChunk.buffer);
                pView.setUint32(0, 9);
                physChunk[4] = 0x70; // 'p'
                physChunk[5] = 0x48; // 'H'
                physChunk[6] = 0x59; // 'Y'
                physChunk[7] = 0x73; // 's'
                pView.setUint32(8, ppm);
                pView.setUint32(12, ppm);
                physChunk[16] = 1; // 1 = meter

                const crc = computeCrc32(physChunk, 4, 13);
                pView.setUint32(17, crc);

                const newBytes = new Uint8Array(bytes.length + 21);
                newBytes.set(bytes.subarray(0, ihdrEnd), 0);
                newBytes.set(physChunk, ihdrEnd);
                newBytes.set(bytes.subarray(ihdrEnd), ihdrEnd + 21);

                resolve(new Blob([newBytes.buffer], { type: 'image/png' }));
            } catch (err) {
                console.warn('Could not inject PNG pHYs metadata:', err);
                resolve(blob);
            }
        };
        reader.onerror = () => resolve(blob);
        reader.readAsArrayBuffer(blob);
    });
}

function getPosterConfig() {
    const sizeKey = document.getElementById('export-size-select')?.value || state.exportSize || '24x36';
    const orient = state.exportOrientation || 'landscape';
    const dpi = parseInt(document.getElementById('export-dpi-select')?.value, 10) || state.exportDpi || 300;
    const preset = POSTER_SIZES[sizeKey] || POSTER_SIZES['24x36'];

    let wIn, hIn;
    if (preset.isViewport) {
        const mapSize = map ? map.getSize() : { x: 1200, y: 900 };
        const baseW = mapSize.x / 96;
        const baseH = mapSize.y / 96;
        if (orient === 'landscape') {
            wIn = Math.max(baseW, baseH);
            hIn = Math.min(baseW, baseH);
        } else {
            wIn = Math.min(baseW, baseH);
            hIn = Math.max(baseW, baseH);
        }
        // Scale so long edge is at least 18 inches for a standard poster
        const longEdge = Math.max(wIn, hIn);
        if (longEdge < 18) {
            const factor = 18 / longEdge;
            wIn *= factor;
            hIn *= factor;
        }
    } else {
        if (orient === 'landscape') {
            wIn = Math.max(preset.wIn, preset.hIn);
            hIn = Math.min(preset.wIn, preset.hIn);
        } else {
            wIn = Math.min(preset.wIn, preset.hIn);
            hIn = Math.max(preset.wIn, preset.hIn);
        }
    }

    const widthPx = Math.round(wIn * dpi);
    const heightPx = Math.round(hIn * dpi);

    return {
        sizeKey,
        preset,
        orient,
        dpi,
        wIn: parseFloat(wIn.toFixed(2)),
        hIn: parseFloat(hIn.toFixed(2)),
        widthPx,
        heightPx,
        ratio: wIn / hIn
    };
}

function getPosterScreenFrame() {
    if (!map) return null;
    const mapSize = map.getSize();
    const config = getPosterConfig();
    const targetRatio = config.ratio; // W / H

    // Fit frame inside cropFrameScale fraction of visible screen viewport (resizable by dragging corner handles)
    const scale = state.cropFrameScale || 0.88;
    const maxW = mapSize.x * scale;
    const maxH = mapSize.y * scale;

    let frameW, frameH;
    if (maxW / maxH > targetRatio) {
        frameH = maxH;
        frameW = maxH * targetRatio;
    } else {
        frameW = maxW;
        frameH = maxW / targetRatio;
    }

    const frameX = (mapSize.x - frameW) / 2;
    const frameY = (mapSize.y - frameH) / 2;

    const nwPt = L.point(frameX, frameY);
    const sePt = L.point(frameX + frameW, frameY + frameH);
    const nePt = L.point(frameX + frameW, frameY);
    const swPt = L.point(frameX, frameY + frameH);

    return {
        screen: { x: frameX, y: frameY, w: frameW, h: frameH },
        points: { nwPt, sePt, nePt, swPt },
        geo: {
            nwLatLng: map.containerPointToLatLng(nwPt),
            seLatLng: map.containerPointToLatLng(sePt),
            neLatLng: map.containerPointToLatLng(nePt),
            swLatLng: map.containerPointToLatLng(swPt),
        },
        config
    };
}

function updateExportMetaInfo() {
    const infoEl = document.getElementById('export-dimensions-info');
    if (!infoEl) return;
    const config = getPosterConfig();
    const wFmt = config.widthPx.toLocaleString();
    const hFmt = config.heightPx.toLocaleString();
    infoEl.textContent = `${wFmt} × ${hFmt} px • ${config.wIn}" × ${config.hIn}" @ ${config.dpi} DPI`;
}

function updatePosterFrame() {
    const overlay = document.getElementById('poster-frame-overlay');
    const box = document.getElementById('poster-frame-box');
    const badge = document.getElementById('poster-frame-badge');
    const framingBadge = document.getElementById('framing-bar-badge');
    if (!overlay || !box || !map) return;

    if (!state.showCropFrame) {
        overlay.classList.add('hidden');
        return;
    }
    overlay.classList.remove('hidden');

    const frameData = getPosterScreenFrame();
    if (!frameData) return;

    const { x, y, w, h } = frameData.screen;
    box.style.left = `${Math.round(x)}px`;
    box.style.top = `${Math.round(y)}px`;
    box.style.width = `${Math.round(w)}px`;
    box.style.height = `${Math.round(h)}px`;

    const config = frameData.config;
    const orientLabel = config.orient.charAt(0).toUpperCase() + config.orient.slice(1);

    if (badge) {
        badge.textContent = `${config.wIn}" × ${config.hIn}" ${orientLabel} (${config.widthPx.toLocaleString()} × ${config.heightPx.toLocaleString()} px)`;
    }
    if (framingBadge) {
        framingBadge.textContent = `${config.wIn}" × ${config.hIn}" ${orientLabel}`;
    }
}

function enterFramingMode() {
    state.showCropFrame = true;
    const overlay = document.getElementById('poster-frame-overlay');
    const framingBar = document.getElementById('export-framing-bar');
    if (overlay) overlay.classList.remove('hidden');
    if (framingBar) framingBar.classList.remove('hidden');

    const frameBtn = document.getElementById('btn-adjust-frame');
    if (frameBtn) {
        frameBtn.classList.add('active');
        frameBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M6 18L18 6M6 6l12 12"/>
            </svg>
            Close Frame
        `;
    }

    updatePosterFrame();
}

function exitFramingMode() {
    state.showCropFrame = false;
    const overlay = document.getElementById('poster-frame-overlay');
    const framingBar = document.getElementById('export-framing-bar');
    if (overlay) overlay.classList.add('hidden');
    if (framingBar) framingBar.classList.add('hidden');

    const frameBtn = document.getElementById('btn-adjust-frame');
    if (frameBtn) {
        frameBtn.classList.remove('active');
        frameBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 8V4m0 0h4M4 4l5 5m11-5h-4m4 0v4m0-4l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5h-4m4 0v-4m0 4l-5-5"/>
            </svg>
            Frame Poster
        `;
    }
}

function initCropFrameHandles() {
    const box = document.getElementById('poster-frame-box');
    const overlay = document.getElementById('poster-frame-overlay');
    if (!box || !overlay) return;

    box.querySelectorAll('.crop-handle').forEach(handle => {
        handle.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            handle.setPointerCapture(e.pointerId);

            const corner = handle.dataset.corner; // 'tl', 'tr', 'bl', 'br'

            const onPointerMove = (ev) => {
                if (!map) return;
                const rect = overlay.getBoundingClientRect();
                const config = getPosterConfig();
                const targetRatio = config.ratio; // W / H

                // Cursor position in container coordinates
                const mouseX = ev.clientX - rect.left;
                const mouseY = ev.clientY - rect.top;

                // Center of container
                const cx = rect.width / 2;
                const cy = rect.height / 2;

                // Distance from center along X and Y according to which corner is dragged
                let dx = 0;
                let dy = 0;

                if (corner === 'br') {
                    dx = mouseX - cx;
                    dy = mouseY - cy;
                } else if (corner === 'tl') {
                    dx = cx - mouseX;
                    dy = cy - mouseY;
                } else if (corner === 'tr') {
                    dx = mouseX - cx;
                    dy = cy - mouseY;
                } else if (corner === 'bl') {
                    dx = cx - mouseX;
                    dy = mouseY - cy;
                }

                if (dx <= 10 && dy <= 10) return;

                // Project (dx, dy) onto the aspect-ratio diagonal line so corner tracks mouse cursor exactly
                // Aspect ratio: wHalf / hHalf = targetRatio => wHalf = targetRatio * hHalf
                const A = targetRatio;
                const wHalf = Math.max(30, (A * (dx * A + dy)) / (A * A + 1));

                // Max possible half-width that fits in container
                const maxHalfW = 0.5 * Math.min(rect.width, rect.height * targetRatio);
                if (maxHalfW <= 0) return;

                const newScale = wHalf / maxHalfW;
                state.cropFrameScale = Math.min(0.96, Math.max(0.15, newScale));
                updatePosterFrame();
            };

            const onPointerUp = (ev) => {
                try {
                    handle.releasePointerCapture(ev.pointerId);
                } catch (_) {}
                handle.removeEventListener('pointermove', onPointerMove);
                handle.removeEventListener('pointerup', onPointerUp);
                handle.removeEventListener('pointercancel', onPointerUp);
            };

            handle.addEventListener('pointermove', onPointerMove);
            handle.addEventListener('pointerup', onPointerUp);
            handle.addEventListener('pointercancel', onPointerUp);
        });
    });
}

async function exportMapPoster() {
    const title = document.getElementById('export-title-input').value || 'Western US Flight Tracks';
    const config = getPosterConfig();
    const frameData = getPosterScreenFrame();
    if (!frameData) {
        alert('Map frame could not be determined.');
        return;
    }

    const exportBtn = document.getElementById('btn-export-poster');
    const confirmBtn = document.getElementById('btn-confirm-export');
    const origExportHTML = exportBtn ? exportBtn.innerHTML : '';
    const origConfirmHTML = confirmBtn ? confirmBtn.innerHTML : '';

    const setExportStatus = (msg) => {
        if (exportBtn) {
            exportBtn.disabled = true;
            exportBtn.innerHTML = msg;
        }
        if (confirmBtn) {
            confirmBtn.disabled = true;
            confirmBtn.innerHTML = msg;
        }
    };

    setExportStatus('Rendering Poster...');

    try {
        const exportW = config.widthPx;
        const exportH = config.heightPx;

        const poster = document.createElement('canvas');
        poster.width = exportW;
        poster.height = exportH;
        const ctx = poster.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // 1. Fill base background
        const currentBg = BASEMAP_LAYERS[state.activeBasemap]?.bgColor || '#111827';
        ctx.fillStyle = currentBg;
        ctx.fillRect(0, 0, exportW, exportH);

        // 2. Determine optimal export tile zoom level
        const baseZ = map.getZoom();
        const maxNativeZoom = BASEMAP_LAYERS[state.activeBasemap]?.layer?.options?.maxZoom || 16;
        
        const nwLatLng = frameData.geo.nwLatLng;
        const seLatLng = frameData.geo.seLatLng;
        const deltaLng = Math.max(0.001, Math.abs(seLatLng.lng - nwLatLng.lng));

        // Target zoom where tile resolution naturally matches export pixel width
        // World width at zoom Z is 256 * 2^Z. Span deltaLng is 256 * 2^Z * (deltaLng / 360).
        // Setting span = exportW gives 2^Z = (exportW * 360) / (256 * deltaLng).
        const idealZ = Math.round(Math.log2((exportW * 360) / (256 * deltaLng)));
        let exportZ = Math.min(maxNativeZoom, Math.max(Math.floor(baseZ), idealZ));

        const tileSize = 256;
        let pNW = map.project(nwLatLng, exportZ);
        let pSE = map.project(seLatLng, exportZ);

        let minPx = Math.min(pNW.x, pSE.x);
        let maxPx = Math.max(pNW.x, pSE.x);
        let minPy = Math.min(pNW.y, pSE.y);
        let maxPy = Math.max(pNW.y, pSE.y);

        let maxTiles = 1 << exportZ;
        let minX = Math.floor(minPx / tileSize) - 1;
        let maxX = Math.ceil(maxPx / tileSize) + 1;
        let minY = Math.max(0, Math.floor(minPy / tileSize) - 1);
        let maxY = Math.min(maxTiles - 1, Math.ceil(maxPy / tileSize) + 1);

        let totalTiles = (maxX - minX + 1) * (maxY - minY + 1);

        // Clamp if tile count exceeds 550 to preserve memory and fast export
        while (totalTiles > 550 && exportZ > Math.floor(baseZ)) {
            exportZ--;
            pNW = map.project(nwLatLng, exportZ);
            pSE = map.project(seLatLng, exportZ);
            minPx = Math.min(pNW.x, pSE.x);
            maxPx = Math.max(pNW.x, pSE.x);
            minPy = Math.min(pNW.y, pSE.y);
            maxPy = Math.max(pNW.y, pSE.y);
            maxTiles = 1 << exportZ;
            minX = Math.floor(minPx / tileSize) - 1;
            maxX = Math.ceil(maxPx / tileSize) + 1;
            minY = Math.max(0, Math.floor(minPy / tileSize) - 1);
            maxY = Math.min(maxTiles - 1, Math.ceil(maxPy / tileSize) + 1);
            totalTiles = (maxX - minX + 1) * (maxY - minY + 1);
        }

        const projW = pSE.x - pNW.x;
        const projH = pSE.y - pNW.y;
        const scaleX = exportW / projW;
        const scaleY = exportH / projH;

        const tileCoordsList = [];
        for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) {
                const tileX_proj = x * tileSize;
                const tileY_proj = y * tileSize;

                const drawX = Math.floor((tileX_proj - pNW.x) * scaleX);
                const drawY = Math.floor((tileY_proj - pNW.y) * scaleY);
                const drawW = Math.ceil(tileSize * scaleX) + 1;
                const drawH = Math.ceil(tileSize * scaleY) + 1;

                const wrappedX = ((x % maxTiles) + maxTiles) % maxTiles;

                tileCoordsList.push({
                    x,
                    y,
                    z: exportZ,
                    tileX: wrappedX,
                    drawX,
                    drawY,
                    drawW,
                    drawH
                });
            }
        }

        // 3. Render Basemap Tiles
        let loadedTiles = 0;
        const updateProgress = () => {
            loadedTiles++;
            setExportStatus(`Rendering Basemap (${loadedTiles}/${tileCoordsList.length})...`);
        };

        if (state.activeBasemap === 'western-dem') {
            await Promise.all(tileCoordsList.map(async (t) => {
                const tileImgData = await getOrRenderTerrariumTile(t.tileX, t.y, t.z);
                if (tileImgData) {
                    const tCanvas = document.createElement('canvas');
                    tCanvas.width = 256;
                    tCanvas.height = 256;
                    tCanvas.getContext('2d').putImageData(tileImgData, 0, 0);
                    ctx.drawImage(tCanvas, t.drawX, t.drawY, t.drawW, t.drawH);
                }
                updateProgress();
            }));
        } else {
            const layerObj = BASEMAP_LAYERS[state.activeBasemap];
            if (layerObj && layerObj.layer) {
                await Promise.all(tileCoordsList.map(async (t) => {
                    try {
                        const url = getLayerTileUrl(layerObj.layer, t.tileX, t.y, t.z);
                        if (url) {
                            const img = await loadImageAsync(url);
                            if (img) {
                                ctx.drawImage(img, t.drawX, t.drawY, t.drawW, t.drawH);
                            }
                        }
                    } catch (e) {
                        console.warn('Basemap tile fetch error:', e);
                    }
                    updateProgress();
                }));
            }
        }

        // 4. Render Roads & Place Names Reference Overlays
        if (state.showLabels && state.labelsOpacity > 0) {
            ctx.save();
            ctx.globalAlpha = state.labelsOpacity;

            // Roads
            if (roadsLayer) {
                setExportStatus('Rendering Roads...');
                await Promise.all(tileCoordsList.map(async (t) => {
                    try {
                        const url = getLayerTileUrl(roadsLayer, t.tileX, t.y, t.z);
                        if (url) {
                            const img = await loadImageAsync(url);
                            if (img) ctx.drawImage(img, t.drawX, t.drawY, t.drawW, t.drawH);
                        }
                    } catch (e) {}
                }));
            }

            // Place Names
            if (labelsLayer) {
                setExportStatus('Rendering Place Names...');
                await Promise.all(tileCoordsList.map(async (t) => {
                    try {
                        const url = getLayerTileUrl(labelsLayer, t.tileX, t.y, t.z);
                        if (url) {
                            const img = await loadImageAsync(url);
                            if (img) ctx.drawImage(img, t.drawX, t.drawY, t.drawW, t.drawH);
                        }
                    } catch (e) {}
                }));
            }
            ctx.restore();
        }

        // 5. Draw Flight Tracks with Subpixel Precision
        setExportStatus('Rendering Flight Tracks...');
        const dpiScale = config.dpi / 96;
        const visibleTracks = state.activeTracks.filter(t => t.visible);

        visibleTracks.forEach(track => {
            const pts = track.points;
            if (pts.length < 2) return;

            const screenCoords = pts.map(p => {
                const ptProj = map.project([p.lat, p.lon], exportZ);
                const exX = (ptProj.x - pNW.x) * scaleX;
                const exY = (ptProj.y - pNW.y) * scaleY;
                return [exX, exY, p.alt, p.time];
            });

            if (state.currentStyle === 'glow') {
                // Outer Glow underlay
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(screenCoords[0][0], screenCoords[0][1]);
                for (let i = 1; i < screenCoords.length; i++) {
                    ctx.lineTo(screenCoords[i][0], screenCoords[i][1]);
                }
                ctx.strokeStyle = state.trackColor;
                ctx.lineWidth = (state.lineWidth + state.glowRadius * 1.8) * dpiScale;
                ctx.globalAlpha = state.opacity * 0.35;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.restore();

                // Core track
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(screenCoords[0][0], screenCoords[0][1]);
                for (let i = 1; i < screenCoords.length; i++) {
                    ctx.lineTo(screenCoords[i][0], screenCoords[i][1]);
                }
                ctx.strokeStyle = state.trackColor;
                ctx.lineWidth = state.lineWidth * dpiScale;
                ctx.globalAlpha = state.opacity;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.restore();

            } else if (state.currentStyle === 'distinct') {
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(screenCoords[0][0], screenCoords[0][1]);
                for (let i = 1; i < screenCoords.length; i++) {
                    ctx.lineTo(screenCoords[i][0], screenCoords[i][1]);
                }
                ctx.strokeStyle = track.color;
                ctx.lineWidth = (state.lineWidth + 0.5) * dpiScale;
                ctx.globalAlpha = state.opacity;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.restore();

            } else if (state.currentStyle === 'altitude' || state.currentStyle === 'vario') {
                ctx.save();
                ctx.lineWidth = (state.lineWidth + 0.5) * dpiScale;
                ctx.globalAlpha = state.opacity;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                for (let i = 0; i < screenCoords.length - 1; i++) {
                    let segColor;
                    if (state.currentStyle === 'altitude') {
                        const avgAlt = (screenCoords[i][2] + screenCoords[i + 1][2]) / 2;
                        segColor = turboColor(avgAlt / 3800.0);
                    } else {
                        const dt = Math.max(1, parseTimeSec(screenCoords[i + 1][3]) - parseTimeSec(screenCoords[i][3]));
                        const dz = screenCoords[i + 1][2] - screenCoords[i][2];
                        segColor = (dz / dt) >= 0 ? '#10B981' : '#EF4444';
                    }
                    ctx.strokeStyle = segColor;
                    ctx.beginPath();
                    ctx.moveTo(screenCoords[i][0], screenCoords[i][1]);
                    ctx.lineTo(screenCoords[i + 1][0], screenCoords[i + 1][1]);
                    ctx.stroke();
                }
                ctx.restore();
            }
        });

        // 6. Draw Elegant Poster Title Card (Scaled to Physical Inches)
        const dpi = config.dpi;
        const pad = Math.round(0.4 * dpi);
        const boxW = Math.min(Math.round(4.8 * dpi), Math.round(exportW * 0.45));
        const boxH = Math.round(1.15 * dpi);
        const boxX = exportW - boxW - pad;
        const boxY = pad;
        const cornerR = Math.round(0.12 * dpi);
        const borderW = Math.max(1.5, Math.round(0.015 * dpi));

        ctx.save();
        ctx.fillStyle = 'rgba(11, 15, 25, 0.90)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
        ctx.lineWidth = borderW;

        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(boxX, boxY, boxW, boxH, cornerR);
        } else {
            ctx.rect(boxX, boxY, boxW, boxH);
        }
        ctx.fill();
        ctx.stroke();

        // Title text
        const titleFontSize = Math.round(0.24 * dpi);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = `bold ${titleFontSize}px Inter, -apple-system, sans-serif`;
        ctx.fillText(title, boxX + Math.round(0.25 * dpi), boxY + Math.round(0.46 * dpi));

        // Subtitle text
        const totalDist = Math.round(state.activeTracks.reduce((acc, t) => acc + t.distanceKm, 0));
        const subFontSize = Math.round(0.135 * dpi);
        ctx.fillStyle = '#94A3B8';
        ctx.font = `500 ${subFontSize}px Inter, -apple-system, sans-serif`;
        ctx.fillText(
            `${state.activeTracks.length} Flights • ${totalDist.toLocaleString()} km Total Distance`,
            boxX + Math.round(0.25 * dpi),
            boxY + Math.round(0.82 * dpi)
        );
        ctx.restore();

        // 7. Generate PNG Blob, Embed pHYs Metadata (300 DPI), and Trigger Download
        setExportStatus('Encoding PNG & Embedding Print Metadata...');
        poster.toBlob(async (blob) => {
            if (!blob) {
                alert('Export failed to generate PNG image.');
                if (exportBtn) {
                    exportBtn.disabled = false;
                    exportBtn.innerHTML = origExportHTML;
                }
                if (confirmBtn) {
                    confirmBtn.disabled = false;
                    confirmBtn.innerHTML = origConfirmHTML;
                }
                return;
            }

            // Inject PNG pHYs chunk with physical DPI
            const finalBlob = await embedPngDpi(blob, config.dpi);

            const url = URL.createObjectURL(finalBlob);
            const link = document.createElement('a');
            const safeTitle = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
            link.download = `${safeTitle}_${config.preset.short || config.sizeKey}_${config.orient}_${config.dpi}dpi.png`;
            link.href = url;
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 20000);

            // Hide bounding box once download starts
            exitFramingMode();

            if (exportBtn) {
                exportBtn.disabled = false;
                exportBtn.innerHTML = origExportHTML;
            }
            if (confirmBtn) {
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = origConfirmHTML;
            }
        }, 'image/png');

    } catch (err) {
        console.error('Poster export error:', err);
        alert('Could not export poster: ' + err.message);
        if (exportBtn) {
            exportBtn.disabled = false;
            exportBtn.innerHTML = origExportHTML;
        }
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.innerHTML = origConfirmHTML;
        }
    }
}

// ============================================================
// Demo Flights Loader
// ============================================================

async function loadDemoTracks() {
    if (window.location.protocol === 'file:') {
        alert('Browser security blocks automatic HTTP loading when running directly as a file://.\n\nTo load demo files automatically, open the web app via a local server (e.g. run "python3 -m http.server 8000" and open http://localhost:8000).\n\nOpening the file picker now so you can select any files from the Tracks folder directly!');
        document.getElementById('file-input').click();
        return;
    }

    const demoFiles = [
        'Tracks/2008-07-26-XGD-538-01.igc',
        'Tracks/2011-06-11-XCS-AAA-08.igc',
        'Tracks/2013-05-30-XCS-AAA-01.igc',
        'Tracks/2014-07-10-XCS-AAA-01.igc',
        'Tracks/Dunlap%20to%20Los%20Banos%202011-05-30_18-17.igc',
        'Tracks/Josh%20Cohn%20Potato%20Hill%202009-07-18_19-07.igc',
        'Tracks/2012-09-30-XCS-AAA-01.igc',
        'Tracks/joshcohn.2017-05-28.18-04-47.IGC',
        'Tracks/joshcohn.2024-08-31.18-31-11.IGC',
        'Tracks/joshcohn.2025-07-12.18-12-11.IGC'
    ];

    const progressContainer = document.getElementById('load-progress-container');
    const progressBar = document.getElementById('load-progress-bar');
    const progressText = document.getElementById('load-progress-text');
    const statusBadge = document.getElementById('status-badge');

    progressContainer.style.display = 'block';
    statusBadge.textContent = 'FETCHING DEMO...';
    statusBadge.style.color = 'var(--warning)';

    let loaded = 0;
    for (const url of demoFiles) {
        try {
            const resp = await fetch(url);
            if (resp.ok) {
                const text = await resp.text();
                const filename = decodeURIComponent(url.split('/').pop());
                const track = parseIGCText(text, filename);
                if (track && track.points.length > 5) {
                    if (!state.allTracks.some(t => t.filename === track.filename)) {
                        state.allTracks.push(track);
                    }
                }
            }
        } catch (err) {
            console.warn('Could not load demo track:', url, err);
        }
        loaded++;
        const pct = Math.round((loaded / demoFiles.length) * 100);
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `Loading demo ${loaded}/${demoFiles.length}`;
    }

    progressContainer.style.display = 'none';
    statusBadge.textContent = 'READY';
    statusBadge.style.color = 'var(--success)';

    applySamplingAndRender();
    renderFlightList();
    updateStatsSummary();
    fitMapToActiveTracks();
}
