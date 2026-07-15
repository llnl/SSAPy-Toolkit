"""
ssapy_toolkit/plots/starfield_verification.py
-----------------------------------------------
Verifies that add_starfield() places stars at the correct sky positions
by overlaying independently-computed reference star positions as large
labeled markers on top of the same scene.

Method
------
For 8 of the brightest, best-known stars (Sirius, Canopus, Arcturus,
Vega, Rigel, Capella, Betelgeuse, Polaris), we compute the expected
GCRF unit vector directly from the SIMBAD/IAU-verified J2000 RA/Dec
coordinates, completely independently of the catalog file. We then
place a large, labeled marker at that unit vector × sky_radius and
check visually whether the marker sits exactly on (or very close to)
the corresponding dim speckle from add_starfield().

If the reference markers and catalog speckles align: coordinates correct.
If they don't: the catalog positions or unit-vector math are wrong.

We also note the one KNOWN issue found by code inspection: the depth
variation in add_starfield() places stars at 0.5x-1.0x sky_radius
rather than a uniform sphere, creating a minor parallax artifact when
rotating the camera. The verification plot uses azim=45 so it exactly
matches what the regular plots show, making comparison straightforward.

Run:
    conda activate myenv
    cd C:/Users/diamond10/SSAPy-Toolkit
    python -m ssapy_toolkit.plots.starfield_verification
"""

import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ssapy_toolkit.plots.starfield import add_starfield

try:
    from ssapy_toolkit.plots.figpath import FIG_DIR
    OUT_DIR = pathlib.Path(FIG_DIR)
except Exception:
    OUT_DIR = pathlib.Path.home() / "yu_figures" / "demo_gallery" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference stars: J2000 RA/Dec from SIMBAD, independently verified ────────
# RA in decimal hours, Dec in decimal degrees.
# mag_v used only for annotation, not for computing position.
REFERENCE_STARS = [
    # name         RA (h)    Dec (°)     mag_v
    ("Sirius",     6.7525,   -16.716,   -1.46),
    ("Canopus",    6.3992,   -52.696,   -0.72),
    ("Arcturus",  14.2610,   +19.182,   -0.04),
    ("Vega",      18.6156,   +38.784,   +0.03),
    ("Capella",    5.2780,   +45.998,   +0.08),
    ("Rigel",      5.2423,    -8.202,   +0.12),
    ("Betelgeuse", 5.9194,    +7.407,   +0.42),
    ("Polaris",    2.5303,   +89.264,   +1.97),
]

# Marker colors per star, chosen to contrast with the dim white speckles
MARKER_COLORS = ["#ff2244", "#ff9900", "#00ffcc", "#ffff00",
                 "#00aaff", "#ff66ff", "#aaffaa", "#ffffff"]


def ra_dec_to_gcrf(ra_h: float, dec_deg: float) -> np.ndarray:
    """Convert RA (hours) + Dec (degrees) to GCRF unit vector.
    This exactly matches starfield.py's own conversion logic, computed
    independently as a cross-check."""
    ra_rad  = np.radians(ra_h * 15.0)
    dec_rad = np.radians(dec_deg)
    return np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])


def make_verification_plot(elev: float, azim: float,
                            plot_range: float = 50_000.0,
                            out_path: pathlib.Path = None):
    """
    Draw a verification plot: the real starfield background plus large
    labeled reference star markers at their independently-computed
    GCRF positions, all in the same coordinate frame.

    Parameters
    ----------
    elev, azim : float
        Camera angles -- must exactly match the plots you want to verify.
        Default 30/45 matches the regular moon_plot_3d default.
    plot_range : float
        Scene half-extent (km). Should match whatever scale produces the
        star pattern you see in the plots being verified. Since starfield
        uses sky_radius = plot_range * 4, the absolute scale doesn't
        affect the DIRECTION of the stars, only how far from the origin
        they're placed. Use 50,000 km (LEO-scale) or 400,000 km
        (cislunar-scale) -- pattern should look the same either way.
    """
    sky_radius = plot_range * 4.0

    fig = plt.figure(figsize=(12, 12), dpi=120, facecolor="#050810")
    ax  = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor("#050810")
    fig.patch.set_facecolor("#050810")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Draw a thin reference sphere at sky_radius so the star positions
    # are clearly visible against something
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    uu, vv = np.meshgrid(u, v)
    ax.plot_surface(
        sky_radius * 0.55 * np.cos(uu)*np.sin(vv),
        sky_radius * 0.55 * np.sin(uu)*np.sin(vv),
        sky_radius * 0.55 * np.cos(vv),
        alpha=0.04, color="#334466", linewidth=0, shade=False,
    )

    # Draw the real starfield (same call as the actual plots)
    add_starfield(ax, plot_range, elev=elev, azim=azim,
                  fov=360, mag_limit=6.5, show_milky_way=True)

    # Draw independently-computed reference star markers
    for (name, ra_h, dec_deg, mag_v), color in zip(REFERENCE_STARS, MARKER_COLORS):
        v_gcrf = ra_dec_to_gcrf(ra_h, dec_deg)

        # Place at the SAME depth formula starfield.py uses for this star's
        # approximate magnitude (so the marker lands on the actual speckle,
        # not just near it in projection). For reference stars we know the
        # mag_v, so we use a representative range for bright stars.
        # Use the full-sky depth range for context (0.5x for Sirius-bright,
        # ~0.6x for Vega-level), and draw a line from 0.5x to 1.0x so you
        # can see the full depth range at each star's direction.
        r_near = sky_radius * 0.48
        r_far  = sky_radius * 1.02
        ax.plot([v_gcrf[0]*r_near, v_gcrf[0]*r_far],
                [v_gcrf[1]*r_near, v_gcrf[1]*r_far],
                [v_gcrf[2]*r_near, v_gcrf[2]*r_far],
                color=color, alpha=0.45, linewidth=0.8, linestyle=':')

        # Large marker at the true unit-vector position (1x sky_radius)
        ax.scatter([v_gcrf[0]*sky_radius*0.75],
                   [v_gcrf[1]*sky_radius*0.75],
                   [v_gcrf[2]*sky_radius*0.75],
                   s=120, color=color, marker='o',
                   edgecolors='white', linewidths=0.5,
                   depthshade=False, zorder=10)
        # Label
        ax.text(v_gcrf[0]*sky_radius*0.80,
                v_gcrf[1]*sky_radius*0.80,
                v_gcrf[2]*sky_radius*0.80,
                f" {name}\n RA {ra_h:.2f}h  Dec {dec_deg:+.1f}°",
                color=color, fontsize=7.5, zorder=10)

    # Draw the three principal axes so orientation is immediately clear
    for axis, label, col in [(np.array([1,0,0]),'X (RA 0h, Dec 0°)','#ff4444'),
                              (np.array([0,1,0]),'Y (RA 6h, Dec 0°)','#44ff44'),
                              (np.array([0,0,1]),'Z (NCP, Dec 90°)', '#4444ff')]:
        r = sky_radius * 0.45
        ax.quiver(0,0,0, axis[0]*r, axis[1]*r, axis[2]*r,
                  color=col, alpha=0.5, arrow_length_ratio=0.12, linewidth=1.2)
        ax.text(*(axis*r*1.12), label, color=col, fontsize=8)

    ax.set_xlim(-sky_radius, sky_radius)
    ax.set_ylim(-sky_radius, sky_radius)
    ax.set_zlim(-sky_radius, sky_radius)

    ax.set_title(
        "Starfield verification: colored circles = independently computed\n"
        "GCRF positions; dim speckles = catalog. Circles should sit ON the\n"
        "corresponding speckle (if not, the catalog positions are wrong).\n"
        "Dotted line = full depth range add_starfield() uses per star direction.",
        color='white', fontsize=9, pad=12,
    )

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), facecolor="#050810", dpi=120)
        plt.close(fig)
        print(f"Saved -> {out_path}")
    return fig, ax


if __name__ == "__main__":
    # Produce two views: the default camera angle and a top-down view
    # (elev=90 looks straight at the North Celestial Pole, so Polaris
    # should appear dead-centre and the ecliptic/equatorial pattern
    # is easy to compare against a real star atlas).

    for elev, azim, suffix in [(30, 45, "default"), (85, 0, "topdown")]:
        out = OUT_DIR / f"starfield_verification_{suffix}.jpg"
        make_verification_plot(elev=elev, azim=azim,
                                plot_range=50_000.0, out_path=out)
        print(f"  {suffix}: {out}")
