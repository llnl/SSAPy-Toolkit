# flake8: noqa: E501
# (The embedded HTML/CSS template below contains inherently long lines that
#  cannot be wrapped without corrupting the generated markup.)
"""Build the self-contained satellite viewer HTML.

Run:  python assemble.py

Inlines scene.js, the vendored libraries (three.min.js, satellite.min.js) and
the four Earth textures into a single standalone HTML file. The texture images
come from the external SSAPy-Data package, not from this source repository.

Layout-agnostic on purpose: inputs are looked up next to this script first,
then in an optional assets/ subfolder, then one directory up. That means the
files can sit flat in an existing package folder or in their own subdirectory,
whichever suits the repo -- no particular directory structure is required.
"""

import base64
import os
from io import BytesIO

from ssapy_toolkit.data import DataPackageNotFoundError, DataResourceNotFoundError, read_data_binary

HERE = os.path.dirname(os.path.abspath(__file__))          # .../ssapy_toolkit/plots
PKG_ROOT = os.path.dirname(HERE)                           # .../ssapy_toolkit
REPO_ROOT = os.path.dirname(PKG_ROOT)                      # repository root

# Candidate directories for inputs, in priority order. The code files are
# expected to sit beside this script; the texture archive may live in a
# dedicated data directory instead, so several conventional locations are
# checked. Set SSAPY_VIEWER_DATA to override with an explicit path.
SEARCH_DIRS = [
    d for d in [
        os.environ.get("SSAPY_VIEWER_DATA"),   # explicit override
        HERE,                                  # ssapy_toolkit/plots/
        os.path.join(PKG_ROOT, "data"),        # ssapy_toolkit/data/
        os.path.join(REPO_ROOT, "data"),       # <repo>/data/
        os.path.join(HERE, "assets"),          # ssapy_toolkit/plots/assets/
        PKG_ROOT,                              # ssapy_toolkit/
        REPO_ROOT,                             # <repo>/
    ] if d
]


def find_input(filename):
    """Locate an input file, or fail saying where we looked."""
    for d in SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    looked = "\n  ".join(SEARCH_DIRS)
    raise FileNotFoundError(
        "ERROR: could not find required input '{}'.\n"
        "Looked in:\n  {}\n"
        "Place it in one of those directories and re-run."
        .format(filename, looked)
    )


def text(filename):
    with open(find_input(filename), "r", encoding="utf-8") as f:
        return f.read()


def load_textures():
    """Return base64 strings for the four Earth textures.

    Textures ship as individual files in SSAPy-Data. This avoids committing a
    duplicated texture archive to SSAPy-Toolkit while
    still allowing installed users to build a self-contained HTML viewer.
    """
    files = {
        "day": "earth_day_2048.jpg",
        "night": "earth_night_2048.jpg",
        "specular": "earth_specular_2048.jpg",
        "clouds": "earth_clouds_2048.png",
    }
    return {
        key: base64.b64encode(_read_texture_binary(filename, key)).decode("ascii")
        for key, filename in files.items()
    }


def _read_texture_binary(filename, kind):
    """Read a texture from SSAPy-Data or return a generated placeholder."""
    try:
        return read_data_binary(filename)
    except (DataPackageNotFoundError, DataResourceNotFoundError):
        pass

    try:
        from ssapy_toolkit.plots.starfield import find_data_file
        path = find_data_file(filename)
        if path is not None:
            return path.read_bytes()
    except Exception:
        pass

    return _placeholder_texture(kind)


def _placeholder_texture(kind):
    """Small deterministic texture used when optional Earth assets are absent."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        # 1x1 black PNG; acceptable for every channel if Pillow is absent.
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

    mode = "RGB" if kind != "clouds" else "RGBA"
    image = Image.new(mode, (64, 32), (7, 25, 58, 255) if mode == "RGBA" else (7, 25, 58))
    draw = ImageDraw.Draw(image)
    if kind == "day":
        draw.rectangle((0, 0, 63, 31), fill=(20, 82, 142))
        draw.ellipse((4, 6, 25, 22), fill=(50, 130, 70))
        draw.ellipse((32, 4, 58, 24), fill=(60, 145, 75))
    elif kind == "night":
        draw.rectangle((0, 0, 63, 31), fill=(2, 8, 24))
        for x, y in [(8, 9), (15, 14), (35, 8), (48, 20), (55, 12)]:
            draw.point((x, y), fill=(255, 210, 120))
    elif kind == "specular":
        draw.rectangle((0, 0, 63, 31), fill=(45, 45, 45))
        draw.ellipse((0, 0, 63, 31), fill=(160, 160, 160))
    else:
        image = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 26, 18), fill=(255, 255, 255, 80))
        draw.ellipse((30, 5, 58, 19), fill=(255, 255, 255, 65))

    buffer = BytesIO()
    if kind == "clouds":
        image.save(buffer, format="PNG")
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# NOTE: asset loading, template substitution and the file write all used to
# happen here at module level. Because ssapy_toolkit/plots/__init__.py
# auto-imports every .py in this folder, that meant a 4.1 MB HTML file was
# read, base64-encoded, assembled and written to disk on EVERY
# `import ssapy_toolkit.plots` -- every GUI start, every pytest run, every CI
# job -- and it printed "wrote ... (4.11 MB)" each time. That work now lives
# in build() below and only runs when this file is executed directly.

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Satellite 3D Scene -- Real Earth texture + TLE-driven satellites</title>
<style>
  html, body { margin: 0; padding: 0; background: #000; overflow: hidden; height: 100%; }
  #scene-container { position: absolute; inset: 0; }
  #hint {
    position: absolute; top: 14px; left: 18px; color: #cfd8e3; font: 13px/1.4 -apple-system, sans-serif;
    background: rgba(0,0,0,0.35); padding: 8px 12px; border-radius: 8px; pointer-events: none;
  }
  #label-layer {
    position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 5;
  }
  .sat-label {
    position: absolute; transform: translate(10px, -50%); white-space: nowrap;
    color: #eef3fa; font: 11px/1.25 -apple-system, sans-serif; letter-spacing: 0.02em;
    background: rgba(8,12,20,0.66); padding: 3px 7px; border-radius: 5px;
    border: 1px solid rgba(140,170,220,0.28); pointer-events: none;
    text-shadow: 0 1px 2px rgba(0,0,0,0.6); will-change: left, top;
  }
  .sat-label .sat-label-sub {
    display: block; font-size: 9.5px; opacity: 0.72; font-variant-numeric: tabular-nums;
    letter-spacing: 0;
  }
  .sat-label::before {
    content: ''; position: absolute; left: -7px; top: 50%; width: 4px; height: 4px;
    margin-top: -2px; border-radius: 50%; background: #ffe066;
    box-shadow: 0 0 5px 1px rgba(255,224,102,0.7);
  }
  #sat-panel {
    position: absolute; top: 14px; right: 18px; color: #e8edf4; font: 13px/1.5 -apple-system, sans-serif;
    background: rgba(10,14,22,0.72); padding: 12px 14px; border-radius: 10px; width: 280px;
    border: 1px solid rgba(255,255,255,0.08); max-height: 90vh; display: flex; flex-direction: column;
  }

  #time-section {
    flex-shrink: 0; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #333c4a;
  }
  #time-clock-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  #time-clock { font-size: 11px; opacity: 0.75; font-variant-numeric: tabular-nums; }
  #time-scale-select {
    background: #171c26; color: #e8edf4; border: 1px solid #333c4a; border-radius: 6px;
    padding: 4px 6px; font: 12px -apple-system, sans-serif;
  }
  #time-pause-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 4px 10px; font: 12px -apple-system, sans-serif; cursor: pointer;
  }
  #time-pause-btn:hover { background: #2b3c5c; }
  #time-step-row { display: flex; gap: 4px; margin-bottom: 6px; }
  .time-step-btn, #time-now-btn {
    flex: 1; background: #171c26; color: #e8edf4; border: 1px solid #333c4a; border-radius: 5px;
    padding: 4px 2px; font: 11px -apple-system, sans-serif; cursor: pointer;
  }
  .time-step-btn:hover, #time-now-btn:hover { background: #1e2532; }
  #time-jump-row { display: flex; gap: 4px; }
  #time-jump-input {
    flex: 1; min-width: 0; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 5px; padding: 3px 4px; font: 11px -apple-system, sans-serif;
  }
  #time-jump-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 5px;
    padding: 3px 8px; font: 11px -apple-system, sans-serif; cursor: pointer; flex-shrink: 0;
  }
  #time-jump-btn:hover { background: #2b3c5c; }

  #db-section { flex-shrink: 0; margin-bottom: 10px; }
  #db-load-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  #db-load-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 5px 10px; font: 12px -apple-system, sans-serif; cursor: pointer; flex-shrink: 0;
  }
  #db-load-btn:hover { background: #2b3c5c; }
  #db-status { font-size: 11px; opacity: 0.65; overflow-wrap: anywhere; }
  #db-search-input {
    width: 100%; box-sizing: border-box; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 6px; padding: 6px 8px; font: 13px -apple-system, sans-serif; margin-bottom: 6px;
  }
  #db-search-input:disabled { opacity: 0.5; }
  #db-search-results { max-height: 260px; overflow-y: auto; border: 1px solid #262d3d; border-radius: 6px; }
  .db-hint { padding: 8px; font-size: 11px; opacity: 0.6; }
  .db-result-row {
    padding: 5px 8px; font-size: 12px; cursor: pointer; display: flex; gap: 6px;
    border-bottom: 1px solid #1c212e;
  }
  .db-result-row:last-child { border-bottom: none; }
  .db-result-row:hover { background: #1a2333; }
  .db-result-row.active { background: #1c3050; }
  .db-result-check { width: 12px; color: #6fd68a; flex-shrink: 0; }
  .db-result-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  #analysis-section {
    flex-shrink: 0; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #333c4a;
  }
  .analysis-toggle {
    display: flex; align-items: center; gap: 7px; font-size: 12px; cursor: pointer; user-select: none;
  }
  .analysis-toggle input { accent-color: #5fd6e6; cursor: pointer; margin: 0; }
  .analysis-swatch {
    width: 18px; height: 3px; border-radius: 2px; background: #5fd6e6; flex-shrink: 0;
    box-shadow: 0 0 4px 0 rgba(95,214,230,0.7);
  }
  .analysis-hint { font-size: 10.5px; opacity: 0.55; }

  #conj-controls { margin-top: 9px; }
  #conj-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 7px; }
  #conj-screen-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 5px 10px; font: 12px -apple-system, sans-serif; cursor: pointer;
  }
  #conj-screen-btn:hover:not(:disabled) { background: #2b3c5c; }
  #conj-screen-btn:disabled { opacity: 0.6; cursor: default; }
  .conj-param { font-size: 11px; opacity: 0.8; display: flex; align-items: center; gap: 3px; }
  .conj-param input {
    width: 46px; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 5px; padding: 3px 4px; font: 11px -apple-system, sans-serif;
  }
  #conj-results { max-height: 168px; overflow-y: auto; font-size: 11.5px; }
  .conj-event {
    padding: 5px 8px; border: 1px solid #2a3346; border-left-width: 3px; border-radius: 6px;
    margin-bottom: 5px; cursor: pointer;
  }
  .conj-event:hover { background: #1a2333; }
  .conj-pair { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conj-meta { opacity: 0.72; font-variant-numeric: tabular-nums; margin-top: 1px; }
  .conj-empty { font-size: 11px; opacity: 0.6; padding: 2px 0; }

  #pass-controls { margin-top: 10px; padding-top: 9px; border-top: 1px solid #333c4a; }
  .pass-title { font-size: 12px; font-weight: 600; margin-bottom: 6px; }
  .pass-site-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
  .pass-site-row input[type=number] { width: 58px; }
  #pass-target {
    flex: 1 1 auto; min-width: 0; background: #171c26; color: #e8edf4; border: 1px solid #333c4a;
    border-radius: 5px; padding: 4px 5px; font: 11px -apple-system, sans-serif;
  }
  #pass-compute-btn {
    background: #223049; color: #e8edf4; border: 1px solid #3a4a6b; border-radius: 6px;
    padding: 4px 10px; font: 12px -apple-system, sans-serif; cursor: pointer; flex-shrink: 0;
  }
  #pass-compute-btn:hover:not(:disabled) { background: #2b3c5c; }
  #pass-compute-btn:disabled { opacity: 0.6; cursor: default; }
  #pass-results { max-height: 168px; overflow-y: auto; font-size: 11.5px; }
  .cloud-toggle-row { margin-top: 4px; margin-bottom: 4px; }
  .cloud-status { font-size: 10px; opacity: 0.6; margin-bottom: 6px; min-height: 12px; overflow-wrap: anywhere; }
  .vis-badge { font-size: 10px; opacity: 0.9; white-space: nowrap; }
  .vis-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }
  .csv-btn {
    margin-top: 6px; background: #1a2230; color: #cfe0f0; border: 1px solid #33465e;
    border-radius: 5px; padding: 4px 10px; font: 11px -apple-system, sans-serif; cursor: pointer;
  }
  .csv-btn:hover:not(:disabled) { background: #223049; }
  .csv-btn:disabled { opacity: 0.4; cursor: default; }

  #legend-details { margin-top: 10px; padding-top: 9px; border-top: 1px solid #333c4a; }
  #legend-details summary { font-size: 12px; cursor: pointer; opacity: 0.85; }
  .legend-grid {
    display: grid; grid-template-columns: auto 1fr; gap: 4px 8px; align-items: center;
    margin-top: 7px; font-size: 11px; opacity: 0.85;
  }
  .lg-swatch { width: 16px; height: 4px; border-radius: 2px; display: inline-block; }
  .legend-note { font-size: 10px; opacity: 0.55; margin-top: 8px; line-height: 1.4; }

  #sat-info { font-size: 12px; line-height: 1.5; overflow-y: auto; flex: 1 1 auto; border-top: 1px solid #333c4a; padding-top: 8px; }
  #sat-info b { color: #ffe066; }
  .sat-info-block { padding: 2px 0; }
  .sat-info-line { padding: 1px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sat-info-sep { border: none; border-top: 1px solid #333c4a; opacity: 0.6; margin: 8px 0; }
</style>
</head>
<body>
<div id="scene-container"></div>
<div id="label-layer"></div>
<div id="hint">Drag to rotate &middot; scroll / pinch to zoom</div>
<div id="sat-panel">
  <div id="time-section">
    <div id="time-clock-row">
      <span id="time-clock">--</span>
      <select id="time-scale-select">
        <option value="1" selected>Real-time</option>
        <option value="60">60x (1 min/s)</option>
        <option value="3600">3,600x (1 hr/s)</option>
        <option value="86400">86,400x (1 day/s)</option>
      </select>
    </div>
    <div id="time-step-row">
      <button class="time-step-btn" type="button" data-step="-86400000">-1d</button>
      <button class="time-step-btn" type="button" data-step="-3600000">-1h</button>
      <button id="time-now-btn" type="button">Now</button>
      <button class="time-step-btn" type="button" data-step="3600000">+1h</button>
      <button class="time-step-btn" type="button" data-step="86400000">+1d</button>
      <button id="time-pause-btn" type="button">Pause</button>
    </div>
    <div id="time-jump-row">
      <input type="datetime-local" id="time-jump-input" step="1">
      <button id="time-jump-btn" type="button">Go (UTC)</button>
    </div>
  </div>
  <div id="db-section">
    <div id="db-load-row">
      <button id="db-load-btn" type="button">Load database</button>
      <span id="db-status" style="opacity:0.5">No file loaded -- load a satellite JSON (CelesTrak, UDL elset, or ssapy_satellites.json)</span>
    </div>
    <input type="file" id="db-file-input" accept=".json" style="display:none">
    <input type="text" id="db-search-input" placeholder="Load a database first..." disabled>
    <div id="db-search-results"></div>
  </div>
  <div id="analysis-section">
    <label class="analysis-toggle">
      <input type="checkbox" id="ground-track-toggle">
      <span class="analysis-swatch"></span>
      Ground tracks
      <span class="analysis-hint">sub-satellite path</span>
    </label>
    <div id="conj-controls">
      <div id="conj-row">
        <button id="conj-screen-btn" type="button">Screen conjunctions</button>
        <label class="conj-param">window <input type="number" id="conj-window" value="24" min="1" max="168" step="1">h</label>
        <label class="conj-param">&le; <input type="number" id="conj-threshold" value="10" min="0.1" max="500" step="0.5">km</label>
      </div>
      <div id="conj-results"><div class="conj-empty">Screen to compute closest approaches.</div></div>
      <button id="conj-export-btn" class="csv-btn" type="button" disabled>Export CSV</button>
    </div>
    <div id="pass-controls">
      <div class="pass-title">Passes over a ground site</div>
      <div class="pass-site-row">
        <label class="conj-param" title="Observer latitude in degrees (north positive)">lat <input type="number" id="pass-lat" value="37.68" step="0.01"></label>
        <label class="conj-param" title="Observer longitude in degrees (east positive)">lon <input type="number" id="pass-lon" value="-121.77" step="0.01"></label>
        <label class="conj-param" title="Minimum elevation above the horizon to count as a pass (0 deg = horizon, 90 deg = straight overhead). Below ~10 deg is usually blocked by terrain/buildings and hazy.">min el <input type="number" id="pass-minel" value="10" min="0" max="89" step="1">&deg;</label>
      </div>
      <div class="pass-site-row">
        <select id="pass-target"></select>
        <label class="conj-param">win <input type="number" id="pass-window" value="24" min="1" max="168" step="1">h</label>
        <button id="pass-compute-btn" type="button">Compute</button>
      </div>
      <label class="analysis-toggle cloud-toggle-row">
        <input type="checkbox" id="cloud-toggle">
        Check sky <span class="analysis-hint">Open-Meteo forecast &middot; external request</span>
      </label>
      <div id="cloud-status" class="cloud-status"></div>
      <div id="pass-results"><div class="conj-empty">Set a site and target, then Compute.</div></div>
      <button id="pass-export-btn" class="csv-btn" type="button" disabled>Export CSV</button>
    </div>
    <details id="legend-details">
      <summary>Legend</summary>
      <div class="legend-grid">
        <span class="lg-swatch" style="background:#ffffff"></span><span>orbit path</span>
        <span class="lg-swatch" style="background:#5fd6e6"></span><span>ground track / nadir</span>
        <span class="lg-swatch" style="background:#66ff99"></span><span>ground site / in view</span>
        <span class="lg-swatch" style="background:#6fd68a; border-radius:50%; width:8px; height:8px"></span><span>pass: optically visible</span>
        <span class="lg-swatch" style="background:#7d8796; border-radius:50%; width:8px; height:8px"></span><span>pass: daylight (not visible)</span>
        <span class="lg-swatch" style="background:#4a5568; border-radius:50%; width:8px; height:8px"></span><span>pass: in Earth's shadow</span>
        <span class="lg-swatch" style="background:#ffe066"></span><span>conjunction &lt; threshold</span>
        <span class="lg-swatch" style="background:#ffa53d"></span><span>conjunction &lt; 5 km</span>
        <span class="lg-swatch" style="background:#ff4d4d"></span><span>conjunction &lt; 1 km</span>
      </div>
      <div class="legend-note">Motion is real propagation at the selected time (use the speed control to see it). Positions are TLE-accuracy; conjunctions are geometric miss distance, not collision probability.</div>
    </details>
  </div>
  <div id="sat-info"></div>
</div>
<!-- Three.js r128 (MIT) inlined for full offline operation -- no CDN / network
     dependency. See THIRD_PARTY_NOTICES.md. -->
<script>
__THREE_JS__
</script>
<script>
__SATELLITE_JS__
</script>
<script>
const DAY_TEXTURE_DATAURI = "data:image/jpeg;base64,__DAY_B64__";
const NIGHT_TEXTURE_DATAURI = "data:image/jpeg;base64,__NIGHT_B64__";
const SPECULAR_TEXTURE_DATAURI = "data:image/jpeg;base64,__SPEC_B64__";
const CLOUDS_TEXTURE_DATAURI = "data:image/png;base64,__CLOUDS_B64__";
</script>
<script>
__SCENE_JS__
</script>
</body>
</html>
"""

def build(out_path=None, verbose=True):
    """Assemble the self-contained Three.js viewer and write it to disk.

    Everything expensive lives here rather than at module level so that
    importing this module -- which the plots package does automatically --
    costs nothing. Call it explicitly, or run this file as a script.

    Parameters
    ----------
    out_path : str or None
        Destination HTML file. Defaults to the standard SSATK output directory
        under ``~/ssatk_output/figures``.
    verbose : bool
        Print the written path and size, as the old module-level code did.

    Returns
    -------
    str
        The path written.
    """
    _tex = load_textures()
    day_b64 = _tex["day"]
    night_b64 = _tex["night"]
    spec_b64 = _tex["specular"]
    clouds_b64 = _tex["clouds"]

    satellite_js = text("satellite.min.js")
    three_js = text("three.min.js")
    scene_js = text("satellite_viewer_scene.js")

    # Work on a local copy: `html` is the module-level template and must stay
    # un-substituted so repeated build() calls don't compound replacements.
    doc = html
    doc = doc.replace("__SATELLITE_JS__", satellite_js)
    doc = doc.replace("__DAY_B64__", day_b64)
    doc = doc.replace("__NIGHT_B64__", night_b64)
    doc = doc.replace("__SPEC_B64__", spec_b64)
    doc = doc.replace("__CLOUDS_B64__", clouds_b64)
    doc = doc.replace("__SCENE_JS__", scene_js)
    # Three.js last: it's the largest blob, and doing it after the others
    # avoids any chance of a placeholder-looking substring inside it being
    # re-substituted.
    doc = doc.replace("__THREE_JS__", three_js)

    if out_path is None:
        from ssapy_toolkit.plots.figpath import figpath
        out_path = figpath("figures/satellite_3d_scene_threejs.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    if verbose:
        print("wrote {} ({:.2f} MB)".format(out_path, len(doc) / 1e6))
    return out_path


if __name__ == "__main__":
    build()
