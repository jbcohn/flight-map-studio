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

// Shared Offscreen Canvas for reading terrarium RGB
const offCanvas = document.createElement('canvas');
offCanvas.width = 256;
offCanvas.height = 256;
const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

const rawElevBuffer = new Float32Array(256 * 256);

function shadeTerrariumImageData(srcData, z, y) {
    const W = 256, H = 256;
    // Calculate ground cell size at tile center latitude
    const n = Math.PI - (2 * Math.PI * (y + 0.5)) / (1 << z);
    const lat = Math.atan(Math.sinh(n));
    const cellsize = (40075016 * Math.cos(lat)) / (256 * (1 << z));
    const invCell2 = 1.0 / (2 * Math.max(1, cellsize));
    const z_factor = 1.35;

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
                offCtx.drawImage(img, 0, 0);
                const srcData = offCtx.getImageData(0, 0, 256, 256).data;
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
        img.onerror = (err) => {
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
});

function initMap() {
    map = L.map('map', {
        center: REGION_PRESETS.california.center,
        zoom: REGION_PRESETS.california.zoom,
        zoomControl: false,
        preferCanvas: true
    });

    // Add Zoom Control at top-right
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Reference Labels Pane (zIndex 350: sits above basemap at 200, below vector tracks at 400)
    map.createPane('labelsPane');
    map.getPane('labelsPane').style.zIndex = 350;
    map.getPane('labelsPane').style.pointerEvents = 'none';

    // Place Names & Boundaries Reference Layer
    labelsLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Reference &copy; Esri',
        pane: 'labelsPane',
        maxZoom: 18,
        opacity: state.labelsOpacity
    }).addTo(map);

    // Track Layer Group (uses Canvas renderer for fast rendering of thousands of coordinates)
    tracksLayerGroup = L.layerGroup().addTo(map);

    // Set initial native DEM basemap
    setBasemap(state.activeBasemap);
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

    // Place Names & Boundaries Reference Overlay Controls
    const toggleLabels = document.getElementById('toggle-labels');
    const labelsSlider = document.getElementById('labels-opacity-slider');
    const labelsVal = document.getElementById('labels-opacity-val');

    if (toggleLabels) {
        toggleLabels.addEventListener('change', (e) => {
            state.showLabels = e.target.checked;
            if (labelsLayer) {
                labelsLayer.setOpacity(state.showLabels ? state.labelsOpacity : 0);
            }
        });
    }

    if (labelsSlider) {
        labelsSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            state.labelsOpacity = val / 100;
            if (labelsVal) {
                labelsVal.textContent = `${val}%`;
            }
            if (state.showLabels && labelsLayer) {
                labelsLayer.setOpacity(state.labelsOpacity);
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

    // Export Poster
    document.getElementById('btn-export-poster').addEventListener('click', exportMapPoster);

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
                totalDistKm += haversineKm(prevCoord[0], prevCoord[1], lat, lon);
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

    return {
        id: 'trk_' + Math.random().toString(36).slice(2, 9),
        filename,
        date: dateStr || extractDateFromFilename(filename),
        pilot,
        glider,
        points,
        latlngs,
        distanceKm: Math.round(totalDistKm * 10) / 10,
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

function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371.0;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
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
        offCtx.drawImage(img, 0, 0);
        const srcData = offCtx.getImageData(0, 0, 256, 256).data;
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

async function exportMapPoster() {
    const title = document.getElementById('export-title-input').value || 'Western US Flight Tracks';
    const resMultiplier = parseInt(document.getElementById('export-res-select').value, 10) || 1;

    const exportBtn = document.getElementById('btn-export-poster');
    const origHTML = exportBtn.innerHTML;
    exportBtn.disabled = true;
    exportBtn.innerHTML = 'Rendering Poster...';

    try {
        const mapSize = map.getSize();
        const exportW = mapSize.x * resMultiplier;
        const exportH = mapSize.y * resMultiplier;

        const poster = document.createElement('canvas');
        poster.width = exportW;
        poster.height = exportH;
        const ctx = poster.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // 1. Fill base background (slate ocean or dark base)
        const currentBg = BASEMAP_LAYERS[state.activeBasemap]?.bgColor || '#111827';
        ctx.fillStyle = currentBg;
        ctx.fillRect(0, 0, exportW, exportH);

        // 2. Compute visible tile grid at current map view
        const bounds = map.getPixelBounds();
        const origin = map.getPixelOrigin();
        const z = map.getZoom();
        const tileSize = 256;
        const minX = Math.floor(bounds.min.x / tileSize);
        const maxX = Math.floor(bounds.max.x / tileSize);
        const minY = Math.floor(bounds.min.y / tileSize);
        const maxY = Math.floor(bounds.max.y / tileSize);

        const tileCoordsList = [];
        for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) {
                tileCoordsList.push({
                    x,
                    y,
                    z,
                    screenX: x * tileSize - origin.x,
                    screenY: y * tileSize - origin.y
                });
            }
        }

        // 3. Render Basemap Tiles
        if (state.activeBasemap === 'western-dem') {
            await Promise.all(tileCoordsList.map(async (t) => {
                const tileImgData = await getOrRenderTerrariumTile(t.x, t.y, t.z);
                if (tileImgData) {
                    const tCanvas = document.createElement('canvas');
                    tCanvas.width = 256;
                    tCanvas.height = 256;
                    tCanvas.getContext('2d').putImageData(tileImgData, 0, 0);
                    ctx.drawImage(
                        tCanvas,
                        t.screenX * resMultiplier,
                        t.screenY * resMultiplier,
                        tileSize * resMultiplier,
                        tileSize * resMultiplier
                    );
                }
            }));
        } else {
            // Tile-based basemap (USGS 3DEP, Satellite, OpenTopo, Dark Canvas)
            const layerObj = BASEMAP_LAYERS[state.activeBasemap];
            if (layerObj && layerObj.layer && layerObj.layer.getTileUrl) {
                await Promise.all(tileCoordsList.map(async (t) => {
                    try {
                        const url = layerObj.layer.getTileUrl({ x: t.x, y: t.y, z: t.z });
                        const img = await loadImageAsync(url);
                        if (img) {
                            ctx.drawImage(
                                img,
                                t.screenX * resMultiplier,
                                t.screenY * resMultiplier,
                                tileSize * resMultiplier,
                                tileSize * resMultiplier
                            );
                        }
                    } catch (e) {
                        console.warn('Basemap tile fetch error:', e);
                    }
                }));
            }
        }

        // 4. Render Place Names & Boundaries Reference Overlay (if enabled)
        if (state.showLabels && state.labelsOpacity > 0 && labelsLayer) {
            ctx.save();
            ctx.globalAlpha = state.labelsOpacity;
            await Promise.all(tileCoordsList.map(async (t) => {
                try {
                    const url = labelsLayer.getTileUrl({ x: t.x, y: t.y, z: t.z });
                    const img = await loadImageAsync(url);
                    if (img) {
                        ctx.drawImage(
                            img,
                            t.screenX * resMultiplier,
                            t.screenY * resMultiplier,
                            tileSize * resMultiplier,
                            tileSize * resMultiplier
                        );
                    }
                } catch (e) {
                    // Ignore missing label tiles
                }
            }));
            ctx.restore();
        }

        // 5. Draw Flight Tracks (Vector paths scaled to export resolution)
        const visibleTracks = state.activeTracks.filter(t => t.visible);
        visibleTracks.forEach(track => {
            const pts = track.points;
            if (pts.length < 2) return;

            // Project all coordinates to container pixel space
            const screenCoords = pts.map(p => {
                const pt = map.latLngToContainerPoint([p.lat, p.lon]);
                return [pt.x * resMultiplier, pt.y * resMultiplier, p.alt, p.time];
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
                ctx.lineWidth = (state.lineWidth + state.glowRadius * 1.8) * resMultiplier;
                ctx.globalAlpha = state.opacity * 0.35;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.restore();

                // Core sharp track
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(screenCoords[0][0], screenCoords[0][1]);
                for (let i = 1; i < screenCoords.length; i++) {
                    ctx.lineTo(screenCoords[i][0], screenCoords[i][1]);
                }
                ctx.strokeStyle = state.trackColor;
                ctx.lineWidth = state.lineWidth * resMultiplier;
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
                ctx.lineWidth = (state.lineWidth + 0.5) * resMultiplier;
                ctx.globalAlpha = state.opacity;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.restore();

            } else if (state.currentStyle === 'altitude' || state.currentStyle === 'vario') {
                ctx.save();
                ctx.lineWidth = (state.lineWidth + 0.5) * resMultiplier;
                ctx.globalAlpha = state.opacity;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                const SEG_STEP = 3;
                for (let i = 0; i < screenCoords.length - 1; i += SEG_STEP) {
                    const end = Math.min(screenCoords.length - 1, i + SEG_STEP);
                    let segColor;
                    if (state.currentStyle === 'altitude') {
                        const avgAlt = (screenCoords[i][2] + screenCoords[end][2]) / 2;
                        segColor = turboColor(avgAlt / 3800.0);
                    } else {
                        const dt = Math.max(1, parseTimeSec(screenCoords[end][3]) - parseTimeSec(screenCoords[i][3]));
                        const dz = screenCoords[end][2] - screenCoords[i][2];
                        segColor = (dz / dt) >= 0 ? '#10B981' : '#EF4444';
                    }
                    ctx.strokeStyle = segColor;
                    ctx.beginPath();
                    ctx.moveTo(screenCoords[i][0], screenCoords[i][1]);
                    for (let j = i + 1; j <= end; j++) {
                        ctx.lineTo(screenCoords[j][0], screenCoords[j][1]);
                    }
                    ctx.stroke();
                }
                ctx.restore();
            }
        });

        // 6. Draw Elegant Title Card Overlay (Top Right)
        const pad = 24 * resMultiplier;
        const boxW = 340 * resMultiplier;
        const boxH = 76 * resMultiplier;
        const boxX = exportW - boxW - pad;
        const boxY = pad;

        ctx.save();
        ctx.fillStyle = 'rgba(11, 15, 25, 0.88)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1.5 * resMultiplier;

        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(boxX, boxY, boxW, boxH, 10 * resMultiplier);
        } else {
            ctx.rect(boxX, boxY, boxW, boxH);
        }
        ctx.fill();
        ctx.stroke();

        // Title text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = `bold ${16 * resMultiplier}px Inter, -apple-system, sans-serif`;
        ctx.fillText(title, boxX + 18 * resMultiplier, boxY + 30 * resMultiplier);

        // Subtitle text
        const totalDist = Math.round(state.activeTracks.reduce((acc, t) => acc + t.distanceKm, 0));
        ctx.fillStyle = '#94A3B8';
        ctx.font = `500 ${11 * resMultiplier}px Inter, -apple-system, sans-serif`;
        ctx.fillText(
            `${state.activeTracks.length} Flights • ${totalDist.toLocaleString()} km Total Distance`,
            boxX + 18 * resMultiplier,
            boxY + 54 * resMultiplier
        );
        ctx.restore();

        // 7. Trigger PNG Download
        poster.toBlob((blob) => {
            if (!blob) {
                alert('Export failed to generate PNG blob.');
                return;
            }
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `${title.toLowerCase().replace(/\s+/g, '_')}_poster.png`;
            link.href = url;
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 15000);
        }, 'image/png');

    } catch (err) {
        console.error('Poster export error:', err);
        alert('Could not export poster: ' + err.message);
    } finally {
        exportBtn.disabled = false;
        exportBtn.innerHTML = origHTML;
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
        'Tracks/joshcohn.2016-11-13.09-49-57.IGC',
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
