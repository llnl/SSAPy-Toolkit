"""
moon_webgl.py — per-fragment WebGL Moon for lunar plots

Drop into:
  ~/SSAPy-Toolkit/ssapy_toolkit/plots/moon_webgl.py

Why this exists
---------------
Plotly colours 3D surfaces per VERTEX, so on moon_render.moon_mesh_plotly
the visible detail is capped by the mesh, not the data: at 180x360 the
whole Moon gets ~65k colour samples regardless of how good the textures
are. moon_raycast.py removes that ceiling by ray-casting per pixel, but it
makes an image, not a scene you can spin.

WebGL shades per FRAGMENT, which removes the same ceiling while staying
interactive. It also fixes something moon_raycast cannot: the GPU filters
textures anisotropically, so the per-pixel surface footprint is averaged
rather than point-sampled. That is the direct cure for the limb speckle in
the ray-cast renders, where one screen pixel spans ~10 DEM cells near the
limb and nearest-neighbour sampling picks one of them arbitrarily.

The photometry is the same as moon_raycast: Lommel-Seeliger from real
incidence and emission angles, with the same opposition surge. See
_FRAG below; the shader is a line-for-line port and is regression-tested
against the ray-caster.

What it needs
-------------
The baked textures from scripts/bake_moon_maps.py, in the cache directory
(~/.ssapy_toolkit/moon by default, or $SSAPY_TOOLKIT_CACHE). Nothing is
read from SSAPy-Data and nothing large is committed.

Why it serves over HTTP
-----------------------
WebGL refuses to upload textures fetched over file:// -- Chrome treats
each file as an opaque origin and Firefox has since 68 -- so a
double-click HTML pointing at sibling textures fails with a security
error. Inlining ~36 MB of textures as base64 would work but produces a
~48 MB document. Instead show() serves the page and the cache together on
a loopback port.

Note this viewer does NOT set logarithmicDepthBuffer. The scene spans a
1737 km Moon and an orbit a few thousand km out, so ordinary depth has
ample precision. That deliberately avoids the trap in the satellite
viewer, where built-in materials write logarithmic depth and hand-written
ShaderMaterials write ordinary depth.

Usage
-----
    from ssapy_toolkit.plots.moon_webgl import moon_webgl, show

    path = moon_webgl(r=r_m, t=t_gps, save_path="~/ssatk_figures/moon.html")
    show(path)                       # serves and opens a browser

    python -m ssapy_toolkit.plots.moon_webgl        # demo, no arguments

r is in metres in the GCRF frame, matching moon_plot_3d and
moon_orbit_plotly, and is converted to the lunar-fixed frame internally.
Pass r_frame="moon_centered" for data already Moon-centred.
"""

from __future__ import annotations

import http.server
import urllib.request
import json
import os
import socketserver
import threading
import webbrowser

import numpy as np

R_MOON_KM = 1737.4

_ASSETS = ("moon_albedo.jpg", "moon_normal.png", "moon_horizon_meta.json")


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def cache_dir():
    """Where bake_moon_maps.py writes. Must match that script."""
    env = os.environ.get("SSAPY_TOOLKIT_CACHE")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".ssapy_toolkit", "moon")


# three.js r128 is fetched once into the texture cache rather than vendored
# in the package. The eclipse renderers ship a vendored r185 ES module pair;
# this page uses the r128 UMD build plus OrbitControls, which is a different
# major with a different module system, so sharing one copy would mean porting
# the shaders. Keeping it in the cache means the repository carries one three.js
# rather than two, and the page still runs offline once the cache is warm --
# the same rule the baked textures follow.
_THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0"
_THREE_FILES = {
    "three.min.js": "/build/three.min.js",
    "OrbitControls.js": "/examples/js/controls/OrbitControls.js",
}


def ensure_three(directory=None, download=True):
    """Return {name: url} for the three.js runtime the page should load.

    Serves from the cache when the files are present, and falls back to the
    CDN when they are absent and cannot be fetched, so an online first run
    still works. Set SSAPY_TOOLKIT_NO_DOWNLOAD=1 to never reach the network.
    """
    d = os.path.expanduser(directory) if directory else cache_dir()
    if os.environ.get("SSAPY_TOOLKIT_NO_DOWNLOAD"):
        download = False
    urls = {}
    for name, remote in _THREE_FILES.items():
        local = os.path.join(d, name)
        if not os.path.exists(local) and download:
            try:
                os.makedirs(d, exist_ok=True)
                with urllib.request.urlopen(_THREE_CDN + remote, timeout=30) as r:
                    data = r.read()
                with open(local, "wb") as f:
                    f.write(data)
                print(f"[moon_webgl] cached {name} ({len(data) / 1024:.0f} KB) in {d}")
            except Exception as exc:
                print(f"[moon_webgl] could not cache {name}: {exc}")
        urls[name] = f"/cache/{name}" if os.path.exists(local) else _THREE_CDN + remote
    return urls


def find_moon_cache(path=None):
    """
    Locate the baked textures, or say precisely how to make them.

    Returns (directory, metadata dict).
    """
    d = os.path.expanduser(path) if path else cache_dir()
    missing = [f for f in _ASSETS if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise SystemExit(
            f"moon_webgl: missing baked textures in {d}\n"
            f"  absent: {', '.join(missing)}\n\n"
            "Build them once with:\n"
            "  python scripts/bake_moon_maps.py --dem <ldem_16_uint.tif> "
            "--albedo <lroc_color_poles_8k.tif>")

    with open(os.path.join(d, "moon_horizon_meta.json")) as f:
        meta = json.load(f)

    n_files = meta["n_az"] // 4
    for k in range(n_files):
        p = os.path.join(d, f"moon_horizon_{k}.png")
        if not os.path.exists(p):
            raise SystemExit(f"moon_webgl: {p} missing; re-run the bake")
    return d, meta


# --------------------------------------------------------------------------
# shaders
# --------------------------------------------------------------------------

_VERT = """
varying vec3 vPos;
varying vec2 vUv;
void main() {
    vPos = position;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

# Port of moon_raycast.render(). Kept deliberately close to the numpy so the
# two can be diffed by eye:
#     ls    = mu0 / (mu0 + mu)
#     B     = 1 / (1 + tan(g/2) / 0.06)
#     shade = ls * (1 + B) / (1 + B * 0.5)
_FRAG = """
precision highp float;

uniform sampler2D uAlbedo;
uniform sampler2D uNormal;
uniform sampler2D uHorizon0;
uniform sampler2D uHorizon1;
uniform sampler2D uHorizon2;
uniform sampler2D uHorizon3;
uniform vec3  uSunDir;          // unit, lunar-fixed
uniform float uExposure;
uniform float uSlopeGain;
uniform float uNAz;
uniform float uAngLo;           // horizon decode, degrees
uniform float uAngHi;
uniform float uShadows;         // 1 enable, 0 disable

varying vec3 vPos;
varying vec2 vUv;

const float PI = 3.141592653589793;

// One horizon bearing. Bearings are packed four to an RGBA texture, so
// bearing k lives in file k/4, channel k%4. GLSL ES 1.0 forbids dynamic
// indexing of samplers, hence the explicit chain.
float horizonAt(vec2 uv, int idx) {
    int tex = idx / 4;
    int ch  = idx - tex * 4;
    vec4 v;
    if      (tex == 0) v = texture2D(uHorizon0, uv);
    else if (tex == 1) v = texture2D(uHorizon1, uv);
    else if (tex == 2) v = texture2D(uHorizon2, uv);
    else               v = texture2D(uHorizon3, uv);
    if      (ch == 0) return v.r;
    else if (ch == 1) return v.g;
    else if (ch == 2) return v.b;
    return v.a;
}

void main() {
    vec3 up = normalize(vPos);

    // local tangent frame; east = z_hat x up, north = up x east.
    // Matches moon_raycast's e_lon and e_lat exactly.
    vec3 zAxis = vec3(0.0, 0.0, 1.0);
    vec3 east  = cross(zAxis, up);
    float eLen = length(east);
    east  = eLen > 1e-6 ? east / eLen : vec3(1.0, 0.0, 0.0);
    vec3 north = cross(up, east);

    // XY normal map; z is reconstructed because storing it wastes the
    // encoding -- near flat ground one 8-bit level of z spans 7.2 degrees.
    vec2 nxy = texture2D(uNormal, vUv).rg * 2.0 - 1.0;
    nxy *= uSlopeGain;
    float nz = sqrt(max(1.0 - dot(nxy, nxy), 0.0));
    vec3 N = normalize(nxy.x * east + nxy.y * north + nz * up);

    vec3 viewDir = normalize(cameraPosition - vPos);

    float mu0 = max(dot(N, uSunDir), 0.0);
    float mu  = max(dot(N, viewDir), 0.0);

    // cast shadow: one horizon lookup replaces moon_raycast's 174-step
    // 3D ray-march. Bearing 0 is north, increasing east.
    if (uShadows > 0.5 && mu0 > 0.0) {
        float sunElev = asin(clamp(dot(uSunDir, up), -1.0, 1.0));
        float sunAz   = atan(dot(uSunDir, east), dot(uSunDir, north));
        if (sunAz < 0.0) sunAz += 2.0 * PI;

        float f  = sunAz / (2.0 * PI) * uNAz;
        float i0 = floor(f);
        float wa = f - i0;
        int a0 = int(mod(i0, uNAz));
        int a1 = int(mod(i0 + 1.0, uNAz));

        float h = mix(horizonAt(vUv, a0), horizonAt(vUv, a1), wa);
        float hAng = radians(uAngLo + h * (uAngHi - uAngLo));
        if (sunElev < hAng) mu0 = 0.0;
    }

    // Lommel-Seeliger with opposition surge, as in moon_raycast
    float ls = (mu0 + mu > 0.0) ? mu0 / (mu0 + mu + 1e-9) : 0.0;
    float g  = acos(clamp(dot(uSunDir, viewDir), -1.0, 1.0));
    float B  = 1.0 / (1.0 + tan(min(g, PI * 0.5 - 1e-3) * 0.5) / 0.06);
    float shade = ls * (1.0 + B) / (1.0 + B * 0.5);

    vec3 alb = texture2D(uAlbedo, vUv).rgb;
    gl_FragColor = vec4(clamp(alb * shade * uExposure, 0.0, 1.0), 1.0);
}
"""


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;
            font-family:system-ui,sans-serif;color:#EAEAEA}
  #c{display:block;width:100%;height:100%}
  #hud{position:absolute;left:14px;top:12px;font-size:13px;line-height:1.5;
       text-shadow:0 1px 3px #000;pointer-events:none}
  #hud b{font-size:15px}
  #load{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
        font-size:14px;color:#9aa}
  #panel{position:absolute;right:14px;top:12px;font-size:12px;
         background:rgba(12,14,18,.72);padding:10px 12px;border-radius:6px}
  #panel label{display:block;margin:6px 0 2px}
  #panel input{width:150px}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><b>{title}</b><br><span id="sub">{subtitle}</span></div>
<div id="load">loading textures…</div>
<div id="panel">
  <label>Sun azimuth <span id="azv"></span></label>
  <input id="az" type="range" min="0" max="360" step="1" value="{sun_az}">
  <label>Sun elevation <span id="elv"></span></label>
  <input id="el" type="range" min="-90" max="90" step="1" value="{sun_el}">
  <label>Exposure <span id="exv"></span></label>
  <input id="ex" type="range" min="0.2" max="3" step="0.05" value="{exposure}">
  <label><input id="sh" type="checkbox" checked style="width:auto"> cast shadows</label>
</div>

<script src="{three_src}"></script>
<script src="{orbit_src}"></script>
<script>
const META   = {meta};
const ORBIT  = {orbit};
const R      = {radius};
const VERT   = {vert};
const FRAG   = {frag};

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, innerWidth/innerHeight, 1, 1e7);
camera.position.set(R*3.2, -R*1.6, R*1.1);
camera.up.set(0,0,1);

const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),
                                          antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = R*1.05;
controls.maxDistance = R*40;

const loader = new THREE.TextureLoader();
const maxAniso = renderer.capabilities.getMaxAnisotropy();

// Anisotropy is the point of moving to WebGL: it averages the surface
// footprint of each pixel instead of point-sampling it, which is what
// removes the limb speckle. Horizon textures are exempt -- they store a
// max over a bearing, and mip-averaging a maximum is not a maximum.
function tex(name, aniso) {
  const t = loader.load('cache/' + name);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.minFilter = aniso ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = aniso;
  if (aniso) t.anisotropy = maxAniso;
  return t;
}

const uniforms = {
  uAlbedo:  {value: tex('moon_albedo.jpg', true)},
  uNormal:  {value: tex('moon_normal.png', true)},
  uHorizon0:{value: tex('moon_horizon_0.png', false)},
  uHorizon1:{value: tex('moon_horizon_1.png', false)},
  uHorizon2:{value: tex('moon_horizon_2.png', false)},
  uHorizon3:{value: tex('moon_horizon_3.png', false)},
  uSunDir:  {value: new THREE.Vector3(1,0,0)},
  uExposure:{value: {exposure}},
  uSlopeGain:{value: 1.0},
  uNAz:     {value: META.n_az},
  uAngLo:   {value: META.angle_lo_deg},
  uAngHi:   {value: META.angle_hi_deg},
  uShadows: {value: 1.0}
};

// SphereGeometry is Y-up; rotate so +Z is lunar north, matching the frame
// moon_raycast and the bake use. Its own UVs are used rather than deriving
// them from position, because a derived longitude wraps discontinuously and
// the mip selection then shows a seam.
const geom = new THREE.SphereGeometry(R, 256, 128);
geom.rotateX(Math.PI/2);
const moon = new THREE.Mesh(geom, new THREE.ShaderMaterial({
  uniforms: uniforms, vertexShader: VERT, fragmentShader: FRAG
}));
scene.add(moon);

if (ORBIT && ORBIT.length > 2) {
  const pos = new Float32Array(ORBIT);
  const col = new Float32Array(pos.length);
  const n = pos.length/3;
  for (let i=0;i<n;i++){
    const f = i/(n-1);
    col[3*i] = 0.25+0.75*f; col[3*i+1] = 0.55+0.25*f; col[3*i+2] = 1.0-0.55*f;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos,3));
  g.setAttribute('color',    new THREE.BufferAttribute(col,3));
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({vertexColors:true})));
}

function setSun() {
  const az = +document.getElementById('az').value;
  const el = +document.getElementById('el').value;
  const a = az*Math.PI/180, e = el*Math.PI/180;
  uniforms.uSunDir.value.set(Math.cos(e)*Math.cos(a),
                             Math.cos(e)*Math.sin(a),
                             Math.sin(e)).normalize();
  document.getElementById('azv').textContent = az + '\\u00B0';
  document.getElementById('elv').textContent = el + '\\u00B0';
}
for (const id of ['az','el']) document.getElementById(id).oninput = setSun;
document.getElementById('ex').oninput = e => {
  uniforms.uExposure.value = +e.target.value;
  document.getElementById('exv').textContent = (+e.target.value).toFixed(2);
};
document.getElementById('sh').onchange =
  e => uniforms.uShadows.value = e.target.checked ? 1.0 : 0.0;
setSun();
document.getElementById('exv').textContent = uniforms.uExposure.value.toFixed(2);

THREE.DefaultLoadingManager.onLoad = () =>
  document.getElementById('load').style.display = 'none';

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

(function loop(){
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
})();
</script></body></html>
"""


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _as_km(arr):
    """moon_plot_3d passes metres. A lunar orbit is a few thousand km."""
    arr = np.asarray(arr, dtype=float)
    return arr / 1e3 if np.nanmax(np.abs(arr)) > 1e6 else arr


def _orbit_xyz(r, t, r_frame):
    if r is None:
        return []
    if r_frame == "moon_centered":
        xyz = _as_km(r)
    else:
        try:
            from ..coordinates import gcrf_to_lunar_fixed
        except ImportError:
            from ssapy_toolkit.coordinates import gcrf_to_lunar_fixed
        xyz = _as_km(gcrf_to_lunar_fixed(np.asarray(r, dtype=float), t))
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return [round(float(v), 3) for v in xyz.ravel()]


def moon_webgl(r=None, t=None, r_frame="gcrf",
               title="Moon — lunar-fixed frame", subtitle=None,
               sun_azimuth_deg=35.0, sun_elevation_deg=8.0,
               exposure=1.4, cache=None, save_path=None):
    """
    Write the viewer page. Returns its path.

    r, t              trajectory and times, as moon_plot_3d takes them
    sun_*_deg         initial Sun direction in the lunar-fixed frame; both
                      are live sliders in the page. A low elevation shows
                      the relief, which is the whole point of the bake
    exposure          display gain, not radiometry
    save_path         defaults beside the cache
    """
    d, meta = find_moon_cache(cache)
    orbit = _orbit_xyz(r, t, r_frame)

    if subtitle is None:
        if orbit:
            a = np.linalg.norm(np.array(orbit).reshape(-1, 3), axis=1) - R_MOON_KM
            subtitle = (f"orbit altitude {a.min():.0f}–{a.max():.0f} km · "
                        f"LOLA relief · Lommel-Seeliger")
        else:
            subtitle = ("LOLA relief · Lommel-Seeliger · "
                        f"{meta['n_az']} bearing horizon shadows")

    three = ensure_three(d)
    doc = _HTML
    for key, val in (("{title}", title),
                     ("{three_src}", three["three.min.js"]),
                     ("{orbit_src}", three["OrbitControls.js"]),
                     ("{subtitle}", subtitle),
                     ("{meta}", json.dumps(meta)),
                     ("{orbit}", json.dumps(orbit)),
                     ("{radius}", repr(R_MOON_KM)),
                     ("{vert}", json.dumps(_VERT)),
                     ("{frag}", json.dumps(_FRAG)),
                     ("{sun_az}", repr(float(sun_azimuth_deg))),
                     ("{sun_el}", repr(float(sun_elevation_deg))),
                     ("{exposure}", repr(float(exposure)))):
        doc = doc.replace(key, val)

    out = os.path.expanduser(save_path) if save_path \
        else os.path.join(d, "moon_webgl.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[moon_webgl] wrote {out}  ({len(doc) / 1024:.0f} KB)")
    return out


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------

def _handler(html_path, cache_path):
    class H(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            p = path.split("?", 1)[0].split("#", 1)[0]
            if p in ("/", "/index.html"):
                return html_path
            if p.startswith("/cache/"):
                # basename only: never let a request walk out of the cache
                return os.path.join(cache_path, os.path.basename(p))
            return super().translate_path(path)

        def log_message(self, *a):
            pass
    return H


def show(html_path=None, cache=None, port=0, open_browser=True):
    """
    Serve the page and its textures on loopback, and open a browser.

    Textures cannot be loaded over file:// -- WebGL rejects them as
    cross-origin -- so the page needs an HTTP origin even locally.
    Blocks until interrupted.
    """
    d, _ = find_moon_cache(cache)
    html_path = html_path or os.path.join(d, "moon_webgl.html")
    if not os.path.exists(html_path):
        html_path = moon_webgl(cache=cache)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port),
                                _handler(html_path, d)) as srv:
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        print(f"[moon_webgl] serving {url}   (ctrl-C to stop)")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[moon_webgl] stopped")


if __name__ == "__main__":
    show(moon_webgl())