"""
cislunar_webgl.py — Self-contained WebGL cislunar orbit scene
=============================================================
Renders cislunar orbits in the Earth-Moon pulsating synodic frame as a single
self-contained HTML file: three.js, orbit geometry, star catalogue and both
body textures are all embedded, so the output is one file you can double-click.
No local server, no CORS workaround, no loose assets.

Why WebGL rather than matplotlib: ``plot_surface`` flat-shades one colour per
quad and has no UV mapping, so body detail is capped by mesh density and costs
O(n^2) re-projection per frame. Here the GPU samples the full-resolution
albedo per fragment and detail is effectively free.

Drop into:
  ~/SSAPy-Toolkit/ssapy_toolkit/plots/cislunar_webgl.py

Usage:
    from ssapy_toolkit.plots.cislunar_webgl import cislunar_webgl
    cislunar_webgl(n_orbits=100, save_path="cislunar_l2.html")

    python -m ssapy_toolkit.plots.cislunar_webgl --n 100
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np

__all__ = ["cislunar_webgl", "collect_orbits", "load_tracks", "save_tracks",
           "collect_stars",
           "encode_texture", "lagrange_collinear", "FAMILY_COLOR"]

from ..constants import EARTH_RADIUS, MOON_RADIUS, EARTH_MU, MOON_MU
from ..coordinates import gcrf_to_lunar_fixed
from .starfield import star_directions
from .figpath import ssatk_path

EPOCH_ISO = "1980-01-01T00:00:00"          # cislunar catalogue epoch (TT)
MU_EM = MOON_MU / (EARTH_MU + MOON_MU)
R_EARTH_KM, R_MOON_KM = EARTH_RADIUS / 1e3, MOON_RADIUS / 1e3
YELLOW, TURQ = 0xFCB317, 0x00A5B8
FAMILY_COLOR = {
    "Moon-line libration": (0x00A5B8, "Moon-line librator (L1-L2 corridor)"),
    "L4 tadpole":          (0x84C342, "L4 tadpole"),
    "L5 tadpole":          (0xFF7900, "L5 tadpole"),
    "horseshoe":           (0xB40F64, "Horseshoe"),
    "near-Moon":           (0xFCB317, "Moon-bound"),
}


# --------------------------------------------------------------------------
def lagrange_collinear(mu: float = MU_EM):
    """Exact CR3BP L1/L2 as a fraction of the Earth-Moon separation.

    Not ``orbital_mechanics.lagrange_points``: that solves a quadratic and puts
    L2 at +0.0998 d instead of +0.1678 d, ~26 000 km short.
    """
    from scipy.optimize import brentq

    def f(x):
        return (x - (1 - mu) * (x + mu) / abs(x + mu) ** 3
                - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3)
    return (brentq(f, -mu + 1e-6, 1 - mu - 1e-6) + mu,
            brentq(f, 1 - mu + 1e-6, 2.0) + mu)


def earth_range_km(n_samples, stride_hours=12, epoch_iso=EPOCH_ISO):
    """True Earth-Moon separation on the track sampling grid.

    In the lunar-fixed frame the Earth sits at ``x = -|r_moon(t)|``, which
    breathes by about 50 000 km over a lunar month.  Pinning it at the mean
    misplaces the Earth by up to 7.4% of a lunar distance.
    """
    from astropy.time import Time
    from ssapy.body import MoonPosition
    t = (Time(epoch_iso, scale="tt")
         + np.arange(n_samples) * stride_hours * 3600.0 / 86400.0).gps
    return np.linalg.norm(MoonPosition()(t).T, axis=1) / 1e3


def save_tracks(orbits, d_km, l1, l2, path="cislunar_tracks_100.npz",
                stride_hours=12):
    """Bundle transformed tracks so the scene does not need the orbit cache."""
    xyz = np.stack([np.asarray(o["p"], dtype=np.int32).reshape(-1, 3)
                    for o in orbits])
    rng = earth_range_km(xyz.shape[1], stride_hours)
    np.savez_compressed(path, xyz=xyz, earth_range=rng.astype(np.float32),
                        ids=np.array([o["id"] for o in orbits]),
                        fam=np.array([o["f"] for o in orbits]),
                        meta=np.array([d_km, l1, l2]))
    print(f"[cislunar_webgl] wrote {path} "
          f"({os.path.getsize(path)/1e6:.1f} MB, {xyz.shape[0]} orbits)")
    return path


def load_tracks(path="cislunar_tracks_100.npz", n_orbits=None):
    """Read a bundle written by :func:`save_tracks`."""
    z = np.load(path, allow_pickle=False)
    xyz, ids, fam = z["xyz"], z["ids"], z["fam"]
    d_km, l1, l2 = (float(v) for v in z["meta"])
    rng = z["earth_range"] if "earth_range" in z.files else None
    out = []
    for k in range(len(ids) if n_orbits is None else min(n_orbits, len(ids))):
        f = str(fam[k])
        out.append({"id": int(ids[k]), "f": f,
                    "c": FAMILY_COLOR.get(f, (0x00A5B8, f))[0],
                    "p": xyz[k].ravel().tolist()})
    return out, d_km, l1, l2, (None if rng is None else
                               [round(float(v), 1) for v in rng])


def collect_orbits(selection="cislunar_selection.csv", cache="cache", n_orbits=100,
                   days=2192.0, stride_hours=12, n_transit=None,
                   pulsating=True):
    """Return (orbits, d_km, l1_km, l2_km) in Moon-centred kilometres.

    Moon-bound orbits are kept first; transits are the visual clutter, so the
    default mix is deliberately light on them.
    """
    import pandas as pd
    from astropy.time import Time
    from ssapy.body import MoonPosition

    sel = pd.read_csv(selection)
    sel = sel[sel.family != "near-Moon"]          # lunar orbits excluded
    # round-robin the families so any n keeps every type represented
    groups = [g for _, g in sel.groupby("family", sort=False)]
    rows, i = [], 0
    while len(rows) < n_orbits and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(rows) < n_orbits:
                rows.append(g.iloc[i])
        i += 1
    pick = pd.DataFrame(rows)

    epoch = Time(EPOCH_ISO, scale="tt")
    ns = int(days * 24 / stride_hours)
    idx = np.arange(ns) * stride_hours
    t_gps = (epoch + idx * 3600.0 / 86400.0).gps
    rng = np.linalg.norm(MoonPosition()(t_gps).T, axis=1) / 1e3
    d_km = float(rng.mean())

    out = []
    for _, row in pick.iterrows():
        oid = int(row.orb_id)
        p = Path(cache) / f"orb_{oid}.npz"
        if not p.exists():
            continue
        r = np.load(p)["r"].astype(float)[::stride_hours][:ns]
        xyz = gcrf_to_lunar_fixed(r, t_gps[:r.shape[0]]) / 1e3
        if pulsating:
            # Pulsating (normalised) synodic frame: scaling radially by
            # d_mean/|r_moon(t)| holds the Earth-Moon separation constant by
            # construction, so a fixed Earth is exact rather than a 7.4%
            # approximation.  Purely radial, so synodic longitudes are
            # untouched and the orbit families keep their geometry.
            xyz = xyz * (d_km / rng[:xyz.shape[0]])[:, None]
        col, _lab = FAMILY_COLOR.get(row.family, (0x00A5B8, row.family))
        out.append({"id": oid, "f": row.family, "c": col,
                    "p": [int(round(float(v))) for v in xyz.ravel()]})
    l1f, l2f = lagrange_collinear()
    return out, d_km, (l1f - 1) * d_km, (l2f - 1) * d_km


def collect_stars(mag_limit=7.0, radius_km=4.0e6):
    """Star positions/colours/sizes in the lunar-fixed frame."""
    from astropy.time import Time
    from ssapy.body import MoonPosition

    v, mag, rgb = (np.asarray(a, float)
                   for a in star_directions(mag_limit=mag_limit, frame="gcrf"))
    t = np.atleast_1d(Time(EPOCH_ISO, scale="tt").gps)
    mp = MoonPosition()
    rm = np.squeeze(mp(t).T)
    vm = np.squeeze(mp(t + 5.0).T) - np.squeeze(mp(t - 5.0).T)
    xh = rm / np.linalg.norm(rm)
    zh = np.cross(rm, vm); zh /= np.linalg.norm(zh)
    yh = np.cross(zh, xh)
    p = (v @ np.vstack([xh, yh, zh]).T) * radius_km
    br = np.clip(1.60 - 0.15 * mag, 0.16, 1.0)
    return {"p": p.astype(np.float32).ravel().round(0).tolist(),
            "c": np.clip(rgb * br[:, None], 0, 1).round(3).ravel().tolist(),
            "s": np.clip(4.8 - 0.48 * mag, 1.0, 6.5).round(2).tolist()}


def encode_texture(path, max_w=8192, quality=94):
    """Downscale and base64-encode a texture as a data URI."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, max_w // 2), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    b = base64.b64encode(buf.getvalue()).decode()
    print(f"  {Path(path).name}: {im.width}x{im.height}, {len(b)//1024} KB b64")
    return "data:image/jpeg;base64," + b


def _find_textures(earth=None, moon=None):
    e = earth
    if e is None:
        try:
            from ssapy.utils import find_file
            p = find_file("earth", ext=".png")
            if p and os.path.getsize(p) > 4096:
                e = p
        except Exception:
            pass
    if e is None:
        for c in ("earth.png", "tex/earth_map.png", "tex/earth_day.jpg"):
            if os.path.exists(c):
                e = c; break
    m = moon
    if m is None:
        cache = os.environ.get("SSAPY_TOOLKIT_CACHE") or \
            os.path.join(Path.home(), ".ssapy_toolkit", "moon")
        for c in (os.path.join(cache, "moon_albedo.jpg"),
                  "tex/moon_albedo.jpg", "tex/moon_map.png", "moon.jpg"):
            if os.path.exists(c):
                m = c; break
    return e, m


# --------------------------------------------------------------------------
_TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>html,body{margin:0;background:#000;overflow:hidden}
canvas{display:block}
#hud{position:fixed;left:18px;top:14px;color:#fff;font:600 21px system-ui,sans-serif}
#sub{position:fixed;left:18px;top:44px;color:#a9aabc;font:13px system-ui,sans-serif}
#clk{position:fixed;right:20px;top:14px;color:#fff;font:16px ui-monospace,monospace}
#key{position:fixed;left:18px;bottom:14px;color:#fff;font:13px system-ui,sans-serif}
b.s{display:inline-block;width:26px;height:3px;vertical-align:middle;margin-right:7px}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud">__TITLE__</div><div id="sub">__SUB__</div>
<div id="clk"></div>
<div id="key">__KEY__</div>
<script>__THREE__</script>
<script>
const ORB=__ORBITS__, STAR=__STARS__, ERANGE=__ERANGE__;
const D=__D__, L1X=__L1__, L2X=__L2__, ANIM=__ANIM__;
const cv=document.getElementById('c');
const r=new THREE.WebGLRenderer({canvas:cv,antialias:true,preserveDrawingBuffer:true});
function size(){r.setSize(innerWidth,innerHeight,true);
  cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();}
const sc=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(30,1,200,9e6);
const SUN=new THREE.Vector3(__SUN__).normalize();

// ---- bodies: opaque, depth-writing, so nothing can bleed through them
const VS=`varying vec2 vU;varying vec3 vN;void main(){vU=uv;
vN=normalize(mat3(modelMatrix)*normal);
gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const FS=`uniform sampler2D m;uniform vec3 s;uniform float nt,gn;
varying vec2 vU;varying vec3 vN;void main(){
vec3 b=pow(texture2D(m,vU).rgb,vec3(2.2));      // sRGB -> linear
float ci=dot(normalize(vN),normalize(s));
vec3 c=b*(nt+(1.-nt)*smoothstep(-0.10,0.25,ci))*gn;
gl_FragColor=vec4(pow(clamp(c,0.,1.),vec3(1./2.2)),1.);}`;
function tex(u){const t=new THREE.TextureLoader().load(u);
  t.anisotropy=16;t.minFilter=THREE.LinearMipmapLinearFilter;return t;}
function body(u,rad,x,nt,gn){
  const m=new THREE.Mesh(new THREE.SphereGeometry(rad,256,128),
    new THREE.ShaderMaterial({uniforms:{m:{value:tex(u)},s:{value:SUN},
      nt:{value:nt},gn:{value:gn}},vertexShader:VS,fragmentShader:FS,
      transparent:false,depthWrite:true,depthTest:true}));
  m.position.set(x,0,0);m.rotation.x=Math.PI/2;m.renderOrder=1;sc.add(m);return m;}
const ES=__ESCALE__;
const EARTH=body('__EARTHTEX__',__RE__*ES,-D,0.04,1.05);
body('__MOONTEX__',__RM__*ES,0,0.03,1.30);

// ---- stars: alphaTest, NOT transparent, so they stay in the opaque pass and
// are depth-tested against the bodies instead of being painted over them
{const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(STAR.p),3));
 g.setAttribute('color',new THREE.BufferAttribute(new Float32Array(STAR.c),3));
 g.setAttribute('ps',new THREE.BufferAttribute(new Float32Array(STAR.s),1));
 const pts=new THREE.Points(g,new THREE.ShaderMaterial({
  vertexShader:`attribute float ps;varying vec3 vC;void main(){vC=color;
    gl_PointSize=ps;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader:`varying vec3 vC;void main(){vec2 d=gl_PointCoord-vec2(.5);
    float a=smoothstep(.5,.1,length(d));if(a<0.35)discard;
    gl_FragColor=vec4(vC*a,1.);}`,
  vertexColors:true,transparent:false,depthWrite:true,depthTest:true}));
 pts.renderOrder=0;sc.add(pts);window.__STARS=pts;}

// ---- orbits
const TRAIL=[];let NPT=0;
// Only the moving trail is drawn.  A full-path underlay showed the whole
// six years at once, so each object's future track was visible before it
// arrived, and for the tight lunar orbits it filled into a solid mass.
ORB.forEach(o=>{const col=o.c;NPT=o.p.length/3;
 const g2=new THREE.BufferGeometry();
 g2.setAttribute('position',new THREE.BufferAttribute(new Float32Array(o.p),3));
 const t=new THREE.Line(g2,new THREE.LineBasicMaterial({color:col,
   transparent:true,opacity:0.85,depthWrite:false}));
 t.renderOrder=2;sc.add(t);TRAIL.push(t);});

const TGT=new THREE.Vector3(-D*0.76,0,0);
function place(){const R=D*7.11,e=0.384,a=0.559;
 cam.up.set(0,0,1);
 cam.position.set(TGT.x+R*Math.cos(e)*Math.sin(a),
                  TGT.y-R*Math.cos(e)*Math.cos(a),R*Math.sin(e));
 cam.lookAt(TGT);
 if(window.__STARS) window.__STARS.position.copy(cam.position);}
size();window.addEventListener('resize',()=>{size();place();});

const TAIL=150,SPF=2,NF=Math.max(60,Math.floor((NPT-1)/4));
place();
function draw(f){const i1=Math.min(NPT-1,1+f*SPF);
 TRAIL.forEach(t=>t.geometry.setDrawRange(Math.max(0,i1-TAIL),Math.min(i1,TAIL)));
 
 const d=i1*12/24;document.getElementById('clk').textContent=
   'T + '+d.toFixed(1)+' d    '+(d/30.4375).toFixed(2)+' mo';
 r.render(sc,cam);}

if(ANIM){let f=0;(function loop(){draw(f);f=(f+1)%NF;
  requestAnimationFrame(loop);})();
 // record straight off the canvas: 60 fps, no frame-by-frame capture needed
 const btn=document.createElement('button');
 btn.textContent='record 30 s';
 btn.style.cssText='position:fixed;right:20px;bottom:16px;padding:7px 13px;'+
   'background:#1a1a1a;color:#fff;border:1px solid #63666a;border-radius:5px;'+
   'font:13px system-ui;cursor:pointer';
 document.body.appendChild(btn);
 btn.onclick=()=>{const st=cv.captureStream(60);
  const mr=new MediaRecorder(st,{mimeType:'video/webm;codecs=vp9',
    videoBitsPerSecond:30e6});
  const ch=[];mr.ondataavailable=e=>ch.push(e.data);
  mr.onstop=()=>{const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob(ch,{type:'video/webm'}));
    a.download='cislunar_l2.webm';a.click();btn.textContent='record 30 s';};
  mr.start();btn.textContent='recording...';setTimeout(()=>mr.stop(),30000);};}
else{TRAIL.forEach(t=>t.geometry.setDrawRange(0,NPT));
  document.getElementById('clk').textContent='';
  setTimeout(()=>r.render(sc,cam),400);r.render(sc,cam);}
</script></body></html>"""


def cislunar_webgl(selection="cislunar_selection.csv", cache="cache", n_orbits=100,
          n_transit=None, save_path=None, animate=True, days=2192.0,
          tracks_npz="cislunar_tracks_100.npz",
          mag_limit=7.0, body_scale=16.0, earth_texture=None, moon_texture=None,
          three_js="three.min.js", max_tex=8192,
          title="Cislunar orbits near the Moon and Earth-Moon L2"):
    """Write one self-contained HTML scene."""
    if tracks_npz and Path(tracks_npz).exists():
        orbits, d, l1, l2, erange = load_tracks(tracks_npz, n_orbits)
        print(f"[cislunar_webgl] tracks from {tracks_npz}")
    else:
        orbits, d, l1, l2 = collect_orbits(selection, cache, n_orbits,
                                           days=days, n_transit=n_transit)
        erange = [round(float(v), 1) for v in
                  earth_range_km(len(orbits[0]['p']) // 3)]
    from collections import Counter
    counts = Counter(o["f"] for o in orbits)
    key = "&nbsp;&nbsp;".join(
        f'<b class="s" style="background:#{FAMILY_COLOR[f][0]:06x}"></b>'
        f'{FAMILY_COLOR[f][1]} {counts[f]}'
        for f in FAMILY_COLOR if counts.get(f))
    print(f"[cislunar_webgl] {len(orbits)} orbits: "
          + ", ".join(f"{f} {c}" for f, c in counts.items()))
    stars = collect_stars(mag_limit)
    print(f"[cislunar_webgl] {len(stars['s']):,} stars")

    et, mt = _find_textures(earth_texture, moon_texture)
    if not et or not mt:
        raise FileNotFoundError(f"textures not found (earth={et}, moon={mt})")
    print("[cislunar_webgl] embedding textures")
    e64, m64 = encode_texture(et, max_tex), encode_texture(mt, max_tex)

    tj = Path(three_js)
    if not tj.exists():
        raise FileNotFoundError(f"{three_js} not found (three.js r128)")

    # Sun direction in the lunar-fixed frame, chosen to light the near sides
    sun = "0.49,-0.87,0.05"

    html = (_TPL
            .replace("__THREE__", tj.read_text(encoding="utf-8"))
            .replace("__ORBITS__", json.dumps(orbits))
            .replace("__STARS__", json.dumps(stars))
            .replace("__EARTHTEX__", e64).replace("__MOONTEX__", m64)
            .replace("__D__", repr(round(d, 1)))
            .replace("__L1__", repr(round(l1, 1)))
            .replace("__L2__", repr(round(l2, 1)))
            .replace("__ANIM__", "true" if animate else "false")
            .replace("__SUN__", sun)
            .replace("__ESCALE__", repr(float(body_scale)))
            .replace("__RE__", repr(R_EARTH_KM))
            .replace("__RM__", repr(R_MOON_KM))
            .replace("__ERANGE__", "null" if not erange else json.dumps(erange))
            .replace("__KEY__", key)
            .replace("__TITLE__", title)
            .replace("__SUB__", f"{len(orbits)} orbits from LLNL's One Million "
                                f"Cislunar Orbits catalogue (Yeager et al. 2025) "
                                f"&middot; Earth-Moon pulsating synodic frame &middot; SSAPy"))
    out = Path(ssatk_path(save_path or "demo_gallery/figures/cislunar_l2.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[cislunar_webgl] wrote {out}  ({len(html)/1e6:.1f} MB, self-contained)")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--transits", type=int, default=None)
    ap.add_argument("--selection", default="cislunar_selection.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default=None,
                    help="path under the SSATK figure dir (see figpath.ssatk_path)")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--days", type=float, default=2192.0)
    ap.add_argument("--body-scale", type=float, default=16.0)
    ap.add_argument("--three", default="three.min.js")
    ap.add_argument("--tracks", default="cislunar_tracks_100.npz")
    a = ap.parse_args()
    cislunar_webgl(selection=a.selection, cache=a.cache, n_orbits=a.n,
                   n_transit=a.transits, save_path=a.out, animate=not a.still,
                   days=a.days, body_scale=a.body_scale, three_js=a.three,
                   tracks_npz=a.tracks)