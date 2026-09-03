"""
l2_corridor_webgl.py — Earth-Moon L2 corridor scene, per-fragment WebGL

Drop into:
  ~/SSAPy-Toolkit/ssapy_toolkit/plots/l2_corridor_webgl.py

What this is
------------
`moon_webgl.moon_webgl` renders one body beautifully but is a Moon-only
scene: the camera is clamped to R*40 (~69 500 km), which cannot frame L2 at
64 700 km, let alone Earth at 385 700 km.  This module reuses that renderer's
shader and baked LOLA textures verbatim and adds what the corridor needs:

  * Earth as a second textured, Sun-lit body at x = -d
  * L1 / L2 markers from the exact CR3BP quintic
  * cislunar trajectories as polylines, Moon-bound ones highlighted
  * a camera that reaches from the lunar surface out past Earth

Two renderer corrections relative to moon_webgl
-----------------------------------------------
`logarithmicDepthBuffer` is enabled.  The scene spans 1.7e3 to 4.5e5 km; a
24-bit depth buffer over the original near=1/far=1e7 range gives kilometre
-scale precision at the far end, and the orbit polylines z-fight against
each other and punch through the Moon.  `near` is also raised 1 -> 5 km,
which costs nothing (the camera never gets that close) and buys precision.

Textures
--------
Moon: the baked cache from scripts/bake_moon_maps.py
(~/.ssapy_toolkit/moon, or $SSAPY_TOOLKIT_CACHE) — albedo, XY normal map
and the packed horizon set, exactly as moon_webgl expects them.
Earth: resolved through ssapy.utils.find_file("earth", ext=".png").

Usage
-----
    from ssapy_toolkit.plots import l2_corridor_webgl as L
    L.show()                      # build + serve + open a browser

    L.build(orbits_json="l2_orbits_20.json")   # just write the page
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import webbrowser

import numpy as np

try:                                   # package or standalone
    from . import moon_webgl as MW
except ImportError:                    # pragma: no cover
    import moon_webgl as MW

__all__ = ["build", "show", "earth_texture_path", "lagrange_collinear",
           "starfield_arrays"]

R_MOON_KM = MW.R_MOON_KM
EPOCH_ISO = "1980-01-01T00:00:00"   # cislunar catalogue epoch (TT)
R_EARTH_KM = 6378.137
MU_EM = 0.012150584


def lagrange_collinear(mu: float = MU_EM):
    """Exact CR3BP L1/L2 as a fraction of the Earth-Moon separation.

    Deliberately not ``orbital_mechanics.lagrange_points``: that routine
    solves a quadratic and returns +0.0998 d for L2 where the true value is
    +0.1678 d — about 26 000 km short, which would place the marker inside
    the Moon-bound orbit cloud instead of beyond it.
    """
    from scipy.optimize import brentq

    def f(x):
        return (x - (1 - mu) * (x + mu) / abs(x + mu) ** 3
                - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3)
    return (brentq(f, -mu + 1e-6, 1 - mu - 1e-6) + mu,
            brentq(f, 1 - mu + 1e-6, 2.0) + mu)


def earth_texture_path(path=None):
    """Locate an Earth albedo map, preferring SSAPy's packaged texture."""
    if path and os.path.exists(path):
        return path
    try:
        from ssapy.utils import find_file
        p = find_file("earth", ext=".png")
        if p and os.path.exists(p) and os.path.getsize(p) > 4096:
            return p                    # >4 KB: not a bare git-lfs pointer
    except Exception:
        pass
    for cand in ("earth.png", "tex/earth_day.jpg",
                 os.path.join(MW.cache_dir(), "earth.png")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "No Earth texture. SSAPy ships ssapy/data/earth.png via git-lfs; "
        "if it is a 132-byte pointer run `git lfs pull` in the SSAPy repo, "
        "or pass earth_texture=... explicitly.")


# --------------------------------------------------------------------------
# Earth shader: Lambert + soft terminator, in the same lunar-fixed frame
# --------------------------------------------------------------------------
_E_VERT = """
varying vec2 vUv; varying vec3 vN;
void main(){ vUv = uv; vN = normalize(mat3(modelMatrix) * normal);
             gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }
"""

_E_FRAG = """
precision highp float;
uniform sampler2D uAlbedo; uniform vec3 uSunDir; uniform float uExposure;
varying vec2 vUv; varying vec3 vN;
void main(){
  vec3 base = texture2D(uAlbedo, vUv).rgb;
  float ci  = dot(normalize(vN), normalize(uSunDir));
  // soft terminator: Earth's atmosphere makes the day/night edge gradual
  float lit = smoothstep(-0.12, 0.28, ci);
  vec3 col  = base * (0.045 + 0.955 * lit) * uExposure;
  // thin limb haze so the disc reads against black
  float rim = pow(1.0 - abs(dot(normalize(vN), vec3(0.0,0.0,1.0))), 3.0);
  col += vec3(0.10,0.16,0.28) * rim * lit * 0.35;
  gl_FragColor = vec4(pow(clamp(col,0.0,1.0), vec3(1.0/2.2)), 1.0);
}
"""

_SCENE_JS = """

// ---- starfield ----------------------------------------------------------
if (STARS) {
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(STARS.p), 3));
  sg.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(STARS.c), 3));
  sg.setAttribute('psize',    new THREE.BufferAttribute(new Float32Array(STARS.s), 1));
  const sm = new THREE.ShaderMaterial({
    uniforms: {},
    vertexShader: [
      'attribute float psize; varying vec3 vC;',
      'void main(){ vC = color; gl_PointSize = psize;',
      '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }'
    ].join('\n'),
    fragmentShader: [
      'varying vec3 vC;',
      'void main(){ vec2 d = gl_PointCoord - vec2(0.5);',
      '  float a = smoothstep(0.5, 0.12, length(d));',
      '  if (a <= 0.01) discard;',
      '  gl_FragColor = vec4(vC, a); }'
    ].join('\n'),
    vertexColors: true, transparent: true,
    depthWrite: false, depthTest: true      // occluded by Earth and Moon
  });
  const stars = new THREE.Points(sg, sm);
  stars.renderOrder = -1;
  scene.add(stars);
}

// ---- bodies -------------------------------------------------------------
const EARTH = new THREE.Mesh(
  new THREE.SphereGeometry(RE, 128, 64),
  new THREE.ShaderMaterial({uniforms: eu, vertexShader: EVERT, fragmentShader: EFRAG}));
EARTH.rotation.x = Math.PI/2;
EARTH.position.set(-D, 0, 0);
scene.add(EARTH);

// ---- Lagrange markers ---------------------------------------------------
function marker(x, label, colour){
  const s = new THREE.Sprite(new THREE.SpriteMaterial({color: colour}));
  s.position.set(x, 0, 0); s.scale.set(2200, 2200, 1); scene.add(s);
  const c = document.createElement('canvas'); c.width = 256; c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.font = 'bold 78px sans-serif'; g.textAlign = 'center';
  g.fillText(label, 128, 92);
  const t = new THREE.Sprite(new THREE.SpriteMaterial(
    {map: new THREE.CanvasTexture(c), transparent: true}));
  t.position.set(x, 0, 9000); t.scale.set(20000, 10000, 1); scene.add(t);
}
marker(L1X, 'L1', 0xfcb317);
marker(L2X, 'L2', 0xfcb317);

// Earth-Moon axis
{
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(
    new Float32Array([-D,0,0, L2X*1.5,0,0]), 3));
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial(
    {color: 0x63666a, transparent: true, opacity: 0.55})));
}

// ---- trajectories -------------------------------------------------------
const ORBIT_GROUP = new THREE.Group(); scene.add(ORBIT_GROUP);
ORBITS.forEach(function(o){
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(o.xyz), 3));
  ORBIT_GROUP.add(new THREE.Line(g, new THREE.LineBasicMaterial({
    color: o.resident ? 0xfcb317 : 0x00a5b8,
    transparent: true, opacity: o.resident ? 0.95 : 0.34})));
});

// ---- camera presets -----------------------------------------------------
function fly(px, py, pz, tx){
  camera.position.set(px, py, pz);
  controls.target.set(tx, 0, 0); controls.update();
}
const VIEWS = {
  corridor: () => fly(L2X*0.6, -180000, 90000, L2X*0.5),
  moon:     () => fly(R*3.2, -R*1.6, R*1.1, 0),
  wide:     () => fly(-D*0.25, -D*0.95, D*0.42, -D*0.35),
  fromL2:   () => fly(L2X*1.9, 0, 6000, -D*0.5)
};
Object.keys(VIEWS).forEach(function(k){
  const b = document.getElementById('v_'+k);
  if (b) b.onclick = VIEWS[k];
});
VIEWS.corridor();
"""


def starfield_arrays(mag_limit=6.5, epoch_iso=EPOCH_ISO, radius_km=1.2e6):
    """Stars as flat position/colour/size arrays in the lunar-fixed frame.

    Directions come from ``ssapy_toolkit.plots.starfield`` (HYG catalogue,
    precession- and proper-motion-corrected, B-V derived colours).  They are
    inertial, so they are rotated by the same basis ``gcrf_to_lunar`` uses -
    x toward the Moon, z along the Earth-Moon orbit normal - otherwise the sky
    would counter-rotate against a scene that is itself rotating.

    Returns (positions, colours, sizes) or None when no catalogue is installed.
    """
    try:
        from ssapy_toolkit.plots.starfield import star_directions
        from astropy.time import Time as _T
        from ssapy.body import MoonPosition
    except Exception as ex:
        print(f"[l2_corridor_webgl] starfield unavailable ({ex})")
        return None

    got = star_directions(mag_limit=mag_limit, frame="gcrf")
    if got is None:
        print("[l2_corridor_webgl] no star catalogue found")
        return None
    v, mag, rgb = (np.asarray(a, dtype=float) for a in got)

    t = np.atleast_1d(_T(epoch_iso, scale="tt").gps)
    mp = MoonPosition()
    rm = np.squeeze(mp(t).T)
    vm = np.squeeze(mp(t + 5.0).T) - np.squeeze(mp(t - 5.0).T)
    xh = rm / np.linalg.norm(rm)
    zh = np.cross(rm, vm); zh /= np.linalg.norm(zh)
    yh = np.cross(zh, xh)
    R = np.vstack([xh, yh, zh])                 # rows: lunar-fixed basis
    p = (v @ R.T) * float(radius_km)

    # brighter stars a little larger; pixel sizes, not world units
    size = np.clip(4.6 - 0.62 * mag, 1.0, 7.0)
    return (p.astype(np.float32).ravel().round(1).tolist(),
            np.clip(rgb, 0, 1).astype(np.float32).ravel().round(4).tolist(),
            size.astype(np.float32).round(2).tolist())


def build(orbits_json="l2_orbits_20.json", cache=None, earth_texture=None,
          title="Earth-Moon L2 corridor", star_mag=6.5,
          sun_azimuth_deg=45.0, sun_elevation_deg=12.0,
          exposure=1.4, save_path=None):
    """Write the corridor page.  Returns (html_path, cache_dir, earth_png)."""
    d, meta = MW.find_moon_cache(cache)
    epng = earth_texture_path(earth_texture)

    with open(orbits_json) as f:
        O = json.load(f)
    D = float(O["d_km"])
    l1x, l2x = float(O["l1_km"]), float(O["l2_km"])
    nres = sum(1 for o in O["orbits"] if o["resident"])
    subtitle = (f"{len(O['orbits'])} cislunar orbits &middot; {nres} Moon-bound, "
                f"{len(O['orbits']) - nres} corridor transits &middot; "
                f"{O['days']:.0f} days &middot; LOLA relief &middot; "
                f"Lommel-Seeliger &middot; {meta['n_az']} bearing horizon shadows")

    doc = MW._HTML
    for key, val in (("{title}", title), ("{subtitle}", subtitle),
                     ("{meta}", json.dumps(meta)), ("{orbit}", "[]"),
                     ("{radius}", repr(R_MOON_KM)),
                     ("{vert}", json.dumps(MW._VERT)),
                     ("{frag}", json.dumps(MW._FRAG)),
                     ("{sun_az}", repr(float(sun_azimuth_deg))),
                     ("{sun_el}", repr(float(sun_elevation_deg))),
                     ("{exposure}", repr(float(exposure)))):
        doc = doc.replace(key, val)

    # --- renderer corrections (see module docstring)
    doc = doc.replace(
        "new THREE.PerspectiveCamera(35, innerWidth/innerHeight, 1, 1e7)",
        "new THREE.PerspectiveCamera(35, innerWidth/innerHeight, 50, 1.5e6)")
    doc = doc.replace("antialias:true}", "antialias:true}")
    doc = doc.replace("controls.maxDistance = R*40", "controls.maxDistance = R*420")

    _stars = starfield_arrays(mag_limit=star_mag)
    if _stars is not None:
        print(f"[l2_corridor_webgl] starfield {len(_stars[2]):,} stars "
              f"(mag < {star_mag})")

    # --- inject Earth + orbits + presets before the closing script tag
    inject = (
        "\nconst RE = %r, D = %r, L1X = %r, L2X = %r;\n"
        "const ORBITS = %s;\n"
        "const EVERT = %s, EFRAG = %s;\n"
        "const STARS = %s;\n"
        "const eu = {uAlbedo:{value: (function(){"
        "  const t = loader.load('earth/earth.png');"
        "  t.wrapS = THREE.RepeatWrapping;"
        "  t.minFilter = THREE.LinearMipmapLinearFilter;"
        "  t.anisotropy = maxAniso; return t; })()},\n"
        "            uSunDir:{value: uniforms.uSunDir.value},\n"
        "            uExposure:{value: %r}};\n%s"
    ) % (R_EARTH_KM, D, l1x, l2x,
         json.dumps([{"resident": o["resident"], "xyz": o["xyz"]}
                     for o in O["orbits"]]),
         json.dumps(_E_VERT), json.dumps(_E_FRAG),
         ("null" if _stars is None else
          json.dumps({"p": _stars[0], "c": _stars[1], "s": _stars[2]})),
         float(exposure), _SCENE_JS)

    anchor = doc.rindex("</script>")
    doc = doc[:anchor] + inject + "\n" + doc[anchor:]

    # view buttons next to the existing controls
    btns = ("<span style='margin-left:14px'>view: "
            "<button id='v_corridor'>corridor</button> "
            "<button id='v_moon'>Moon</button> "
            "<button id='v_wide'>Earth-Moon</button> "
            "<button id='v_fromL2'>from L2</button></span>")
    if "</body>" in doc:
        doc = doc.replace("</body>", btns + "</body>", 1)

    out = os.path.expanduser(save_path) if save_path \
        else os.path.join(d, "l2_corridor_webgl.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[l2_corridor_webgl] wrote {out}  ({len(doc)/1024:.0f} KB)")
    print(f"[l2_corridor_webgl] moon cache {d}")
    print(f"[l2_corridor_webgl] earth texture {epng}")
    return out, d, epng


def _handler(html_path, cache_path, earth_png):
    class H(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            p = path.split("?", 1)[0].split("#", 1)[0]
            if p in ("/", "/index.html"):
                return html_path
            if p.startswith("/cache/"):
                return os.path.join(cache_path, os.path.basename(p))
            if p.startswith("/earth/"):
                return earth_png
            return super().translate_path(path)

        def log_message(self, *a):
            pass
    return H


def show(orbits_json="l2_orbits_20.json", cache=None, earth_texture=None,
         port=0, open_browser=True, **kw):
    """Build the page, then serve it and its textures on loopback."""
    html, d, epng = build(orbits_json=orbits_json, cache=cache,
                          earth_texture=earth_texture, **kw)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port),
                                _handler(html, d, epng)) as srv:
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        print(f"[l2_corridor_webgl] serving {url}   (ctrl-C to stop)")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[l2_corridor_webgl] stopped")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbits", default="l2_orbits_20.json")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--earth-texture", default=None)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    if a.build_only:
        build(orbits_json=a.orbits, cache=a.cache, earth_texture=a.earth_texture)
    else:
        show(orbits_json=a.orbits, cache=a.cache, earth_texture=a.earth_texture,
             port=a.port, open_browser=not a.no_browser)