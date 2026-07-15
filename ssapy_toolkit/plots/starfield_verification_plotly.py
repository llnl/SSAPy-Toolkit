"""
ssapy_toolkit/plots/starfield_verification_plotly.py
------------------------------------------------------
Interactive Plotly version of the starfield verification.

Advantages over the matplotlib version:
  - Rotate and zoom to inspect any star direction in detail
  - Hover over reference circles to see the star's name + exact RA/Dec
  - Toggle trace groups on/off in the legend to isolate catalog vs references
  - Zoom in to confirm a colored circle sits exactly on the corresponding speckle

Run:
    conda activate myenv
    cd C:/Users/diamond10/SSAPy-Toolkit
    python -m ssapy_toolkit.plots.starfield_verification_plotly

Opens an interactive HTML file in your browser.
"""

import sys
import pathlib
import numpy as np
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from ssapy_toolkit.plots.figpath import FIG_DIR
    OUT_DIR = pathlib.Path(FIG_DIR)
except Exception:
    OUT_DIR = pathlib.Path.home() / "yu_figures" / "demo_gallery" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Star catalog paths (same as starfield.py) ─────────────────────────────────
import os
HYG_PATHS = [
    os.path.expanduser("~/bright_stars.csv"),
    os.path.expanduser("~/SSAPy/ssapy/data/bright_stars.csv"),
    os.path.join(os.path.dirname(__file__), "bright_stars.csv"),
]

# ── Reference stars: J2000 RA/Dec from SIMBAD ─────────────────────────────────
REFERENCE_STARS = [
    ("Sirius",      6.7525,   -16.716,  -1.46, "#ff2244"),
    ("Canopus",     6.3992,   -52.696,  -0.72, "#ff9900"),
    ("Arcturus",   14.2610,   +19.182,  -0.04, "#00ffcc"),
    ("Vega",       18.6156,   +38.784,  +0.03, "#ffff00"),
    ("Capella",     5.2780,   +45.998,  +0.08, "#00aaff"),
    ("Rigel",       5.2423,    -8.202,  +0.12, "#ff66ff"),
    ("Betelgeuse",  5.9194,    +7.407,  +0.42, "#aaffaa"),
    ("Polaris",     2.5303,   +89.264,  +1.97, "#ffffff"),
]

SPECT_COLORS = {
    'O': 'rgb(155,175,255)', 'B': 'rgb(171,191,255)',
    'A': 'rgb(201,217,255)', 'F': 'rgb(247,247,255)',
    'G': 'rgb(255,245,235)', 'K': 'rgb(255,209,160)',
    'M': 'rgb(255,204,112)',
}


def ra_dec_to_gcrf(ra_h, dec_deg):
    ra_rad  = np.radians(ra_h * 15.0)
    dec_rad = np.radians(dec_deg)
    return np.array([
        np.cos(dec_rad)*np.cos(ra_rad),
        np.cos(dec_rad)*np.sin(ra_rad),
        np.sin(dec_rad),
    ])


def load_catalog(mag_limit=6.5):
    import pandas as pd
    for p in HYG_PATHS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df[(df['mag'] < mag_limit) & (df['mag'] > -10)].dropna(
                subset=['ra','dec','mag'])
            ra_rad  = np.radians(df['ra'].values * 15.0)
            dec_rad = np.radians(df['dec'].values)
            cx = np.cos(dec_rad)*np.cos(ra_rad)
            cy = np.cos(dec_rad)*np.sin(ra_rad)
            cz = np.sin(dec_rad)
            mag  = df['mag'].values
            spect= df['spect'].fillna('G').str[:1].values
            names= (df['proper'].fillna('') if 'proper' in df.columns
                    else df.get('bf', pd.Series([''] * len(df))).fillna('')).values
            return cx, cy, cz, mag, spect, names
    return None


if __name__ == "__main__":
    R = 1.0  # unit sphere

    fig = go.Figure()

    # ── Catalog stars ──────────────────────────────────────────────────────────
    cat = load_catalog(mag_limit=6.5)
    if cat is not None:
        cx, cy, cz, mag, spect, names = cat
        # depth variation matching starfield.py exactly
        mag_min, mag_max = mag.min(), mag.max()
        depth = 0.5 + 0.5*(mag - mag_min)/(mag_max - mag_min + 1e-6)
        sizes = np.clip(0.5*(6.5 - mag)**1.1, 0.3, 5.0)
        colors = [SPECT_COLORS.get(s, SPECT_COLORS['G']) for s in spect]

        hover = [f"{n if n else '—'}<br>mag={m:.2f}<br>({x:.3f}, {y:.3f}, {z:.3f})"
                 for n, m, x, y, z in zip(names, mag, cx, cy, cz)]

        # True sphere layer (depth=1.0) — what correct positions look like
        fig.add_trace(go.Scatter3d(
            x=cx*R, y=cy*R, z=cz*R,
            mode='markers',
            marker=dict(size=sizes*0.6, color=colors, opacity=0.5),
            text=hover, hoverinfo='text',
            name='Catalog (true sphere, depth=1.0)',
            legendgroup='catalog_true',
        ))

        # Depth-varied layer — what starfield.py actually renders
        fig.add_trace(go.Scatter3d(
            x=cx*depth, y=cy*depth, z=cz*depth,
            mode='markers',
            marker=dict(size=sizes*0.9, color=colors, opacity=0.85),
            text=hover, hoverinfo='text',
            name='Catalog (with depth variation, as rendered)',
            legendgroup='catalog_depth',
        ))
        print(f"Loaded {len(mag)} catalog stars")
    else:
        print("WARNING: bright_stars.csv not found — catalog layer omitted.")
        print(f"Looked in: {HYG_PATHS}")

    # ── Reference stars ────────────────────────────────────────────────────────
    for name, ra_h, dec_deg, mag_v, color in REFERENCE_STARS:
        v = ra_dec_to_gcrf(ra_h, dec_deg)

        # Large marker at unit-sphere position
        fig.add_trace(go.Scatter3d(
            x=[v[0]], y=[v[1]], z=[v[2]],
            mode='markers+text',
            marker=dict(size=10, color=color,
                        line=dict(color='white', width=1.5), opacity=1.0),
            text=[name],
            textposition='top center',
            textfont=dict(color=color, size=11),
            hovertext=f"{name}<br>RA {ra_h:.4f}h  Dec {dec_deg:+.3f}°<br>mag {mag_v:.2f}",
            hoverinfo='text',
            name=f"{name} (ref, depth=1.0)",
            legendgroup='refs',
            legendgrouptitle_text='Reference stars' if name == 'Sirius' else None,
        ))

        # Show the depth-variation range as a line (depth 0.48 to 1.02)
        fig.add_trace(go.Scatter3d(
            x=[v[0]*0.48, v[0]*1.02],
            y=[v[1]*0.48, v[1]*1.02],
            z=[v[2]*0.48, v[2]*1.02],
            mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            hoverinfo='skip', showlegend=False,
        ))

    # ── Principal axes ─────────────────────────────────────────────────────────
    for axis, label, col in [
        ([1,0,0], 'X  RA=0h Dec=0°  (vernal equinox)', '#ff4444'),
        ([0,1,0], 'Y  RA=6h Dec=0°',                   '#44ff44'),
        ([0,0,1], 'Z  Dec=+90°  (North Celestial Pole)','#4444ff'),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]*1.1], y=[0, axis[1]*1.1], z=[0, axis[2]*1.1],
            mode='lines+text',
            line=dict(color=col, width=3),
            text=['', label], textposition='top center',
            textfont=dict(color=col, size=10),
            hoverinfo='skip', name=label,
        ))

    # ── Milky Way equator (galactic plane reference) ───────────────────────────
    gnp = ra_dec_to_gcrf(192.85/15.0, 27.13)
    b1  = np.cross(gnp, [0,1,0]); b1 /= np.linalg.norm(b1)
    b2  = np.cross(gnp, b1);      b2 /= np.linalg.norm(b2)
    th  = np.linspace(0, 2*np.pi, 200)
    mw  = np.outer(np.cos(th), b1) + np.outer(np.sin(th), b2)
    fig.add_trace(go.Scatter3d(
        x=mw[:,0]*1.01, y=mw[:,1]*1.01, z=mw[:,2]*1.01,
        mode='lines',
        line=dict(color='#8899dd', width=1),
        opacity=0.4,
        name='Galactic equator',
        hoverinfo='skip',
    ))

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=("Starfield verification — interactive<br>"
                  "<sup>Colored circles = independently computed GCRF positions.<br>"
                  "Catalog speckles should sit on/under each circle.<br>"
                  "Toggle legend items to isolate layers. "
                  "Dotted line = depth range add_starfield() uses per direction.</sup>"),
            font=dict(color='white', size=13), x=0.5,
        ),
        paper_bgcolor='#050810',
        scene=dict(
            bgcolor='#050810',
            xaxis=dict(range=[-1.2,1.2], showbackground=False,
                       gridcolor='rgba(255,255,255,0.05)',
                       color='rgba(255,255,255,0.3)', title='X (GCRF)'),
            yaxis=dict(range=[-1.2,1.2], showbackground=False,
                       gridcolor='rgba(255,255,255,0.05)',
                       color='rgba(255,255,255,0.3)', title='Y (GCRF)'),
            zaxis=dict(range=[-1.2,1.2], showbackground=False,
                       gridcolor='rgba(255,255,255,0.05)',
                       color='rgba(255,255,255,0.3)', title='Z (GCRF)'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=-1.2, z=0.7),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(
            font=dict(color='white', size=10),
            bgcolor='rgba(12,16,28,0.85)',
            bordercolor='rgba(255,255,255,0.12)',
            borderwidth=1, x=0.01, y=0.99,
        ),
        margin=dict(l=0, r=0, t=100, b=0),
    )

    out = OUT_DIR / "starfield_verification_plotly.html"
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f"\nSaved -> {out}")
    fig.show()
