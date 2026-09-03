"""
moon_orbit_plotly.py — Moon and lunar orbit in one Plotly/WebGL scene
======================================================================
Drop into:
  ~/SSAPy-Toolkit/ssapy_toolkit/plots/moon_orbit_plotly.py

Why this exists
---------------
moon_plot_3d.py draws the Moon through matplotlib's plot_surface with
facecolors. That caps two things at once:

  * colour resolution equals MESH resolution, not texture resolution. At
    n=96 the whole Moon gets ~18k colour samples, so the 13.7 MB
    moon.png in SSAPy-Data is downsampled to 512x256 and then sampled far
    below even that.
  * plot_surface with facecolors antialiases the edge of every quad, which
    is the faint grid visible across the disc in demo_moon_surface.jpg.

moon_render.moon_mesh_plotly already solves both, and nothing was calling
it from the lunar plots: real texture first, per-vertex diffuse lighting
from displaced (bumpy) normals so craters read as relief rather than
paint, opposition backscatter, and optional Earth-shadow eclipse shading.
This module is the thin layer that puts that Moon and an orbit track in
the same figure, which is the figure the demos never produced -- the
surface view and the orbit view are currently separate outputs.

Usage
-----
    from ssapy_toolkit.plots.moon_orbit_plotly import moon_orbit_plotly

    fig = moon_orbit_plotly(r=r_m, t=t_gps,
                            save_path="~/ssatk_figures/moon_llo.html")

r is in metres in the GCRF frame, matching moon_plot_3d's convention, and
is converted to the lunar-fixed frame internally. Pass
r_frame="moon_centered" for data that is already Moon-centred (e.g. real
Horizons output queried with CENTER='coord@301').
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go

from .moon_render import moon_mesh_plotly, R_MOON_KM
from .starfield import starfield_traces
from ..coordinates import gcrf_to_lunar_fixed

try:
    from .sun_mpl import sun_direction_in_frame
except ImportError:                                    # pragma: no cover
    sun_direction_in_frame = None


def _as_km(arr):
    """
    moon_plot_3d takes r in metres. Accept either, and decide by magnitude:
    a lunar orbit is a few thousand km, so anything past 1e6 is metres.
    """
    arr = np.asarray(arr, dtype=float)
    return arr / 1e3 if np.nanmax(np.abs(arr)) > 1e6 else arr


def _sun_hat(t, sun_azimuth_deg=None, sun_elevation_deg=None):
    """
    Unit vector toward the Sun in the lunar-fixed frame.

    Evaluated at the MIDPOINT of the span, for the reason moon_plot_3d
    gives: in the rotating lunar-fixed frame the apparent Sun direction
    sweeps about 13 deg/day, so freezing on the first sample is badly
    wrong for anything longer than a day.

    An explicit azimuth/elevation overrides the ephemeris, which is useful
    for choosing a terminator that shows the relief well.
    """
    if sun_azimuth_deg is not None and sun_elevation_deg is not None:
        az, el = np.radians(sun_azimuth_deg), np.radians(sun_elevation_deg)
        return np.array([np.cos(el) * np.cos(az),
                         np.cos(el) * np.sin(az),
                         np.sin(el)])
    if sun_direction_in_frame is None or t is None or len(np.atleast_1d(t)) == 0:
        return None
    try:
        t_arr = np.atleast_1d(t)
        return sun_direction_in_frame(t_arr[len(t_arr) // 2], gcrf_to_lunar_fixed)
    except Exception as ex:                            # pragma: no cover
        print(f"[moon_orbit_plotly] Sun direction unavailable ({ex}); "
              f"rendering the Moon unshaded.")
        return None


def moon_orbit_plotly(r=None, t=None, r_frame="gcrf",
                      title="Moon — lunar-fixed frame",
                      subtitle=None,
                      sun_azimuth_deg=None, sun_elevation_deg=None,
                      n_lat=180, n_lon=360,
                      show_stars=True, mag_limit=6.0,
                      orbit_colorscale="Turbo", orbit_width=4,
                      save_path=None, show=False):
    """
    Render the Moon with an orbit track around it.

    r, t              : trajectory and times, as moon_plot_3d takes them
                        (r in metres, GCRF, unless r_frame is overridden)
    n_lat, n_lon      : Moon mesh resolution; 180x360 is moon_render's
                        default and gives ~65k vertices
    sun_*_deg         : override the solar ephemeris to place the
                        terminator deliberately
    save_path         : .html keeps it interactive, .png/.jpg needs kaleido

    Returns the plotly Figure.
    """
    fig = go.Figure()

    # ---- orbit into the lunar-fixed frame --------------------------------
    xyz = None
    if r is not None:
        r_km = _as_km(r)
        if r_frame == "moon_centered":
            xyz = r_km
        else:
            xyz = _as_km(gcrf_to_lunar_fixed(np.asarray(r, dtype=float), t))
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)

    # ---- scene scale ------------------------------------------------------
    if xyz is not None and len(xyz):
        span = float(np.nanmax(np.linalg.norm(xyz, axis=1)))
        scene_radius = max(span * 1.25, R_MOON_KM * 1.6)
    else:
        scene_radius = R_MOON_KM * 4.0

    # ---- stars first, so everything else draws over them ------------------
    if show_stars:
        try:
            for trace in starfield_traces(sky_radius=scene_radius * 50,
                                          when=None, frame="gcrf",
                                          mag_limit=mag_limit):
                fig.add_trace(trace)
        except Exception as ex:                        # pragma: no cover
            print(f"[moon_orbit_plotly] Starfield unavailable ({ex}).")

    # ---- the Moon ---------------------------------------------------------
    sun_hat = _sun_hat(t, sun_azimuth_deg, sun_elevation_deg)
    fig.add_trace(moon_mesh_plotly(
        center=np.zeros(3), radius=R_MOON_KM, sun_hat=sun_hat,
        mode="solar",           # normally sunlit; no Earth-shadow check here
        n_lat=n_lat, n_lon=n_lon,
    ))

    # ---- the orbit --------------------------------------------------------
    if xyz is not None and len(xyz):
        alt = np.linalg.norm(xyz, axis=1) - R_MOON_KM
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="lines",
            line=dict(width=orbit_width, color=np.arange(len(xyz)),
                      colorscale=orbit_colorscale),
            name="Orbit",
            hovertemplate="altitude %{customdata:.0f} km<extra></extra>",
            customdata=alt,
        ))
        fig.add_trace(go.Scatter3d(
            x=[xyz[-1, 0]], y=[xyz[-1, 1]], z=[xyz[-1, 2]],
            mode="markers",
            marker=dict(size=4, color="#FFD400", line=dict(width=0)),
            name="Spacecraft", hoverinfo="skip", showlegend=False,
        ))
        if subtitle is None:
            subtitle = (f"orbit altitude {alt.min():.0f}–{alt.max():.0f} km "
                        f"above a {R_MOON_KM:.0f} km radius")

    # ---- layout -----------------------------------------------------------
    heading = title if not subtitle else f"{title}<br><sub>{subtitle}</sub>"
    axis = dict(visible=False, showgrid=False, zeroline=False,
                showbackground=False,
                range=[-scene_radius, scene_radius])
    fig.update_layout(
        title=dict(text=heading, x=0.5, xanchor="center",
                   font=dict(color="#EAEAEA", size=18)),
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        showlegend=False, margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.25, y=1.25, z=0.55)),
            bgcolor="#000000",
        ),
    )

    # ---- output -----------------------------------------------------------
    if save_path:
        save_path = os.path.expanduser(str(save_path))
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".pdf")):
            fig.write_image(save_path, width=1600, height=1200, scale=2)
        else:
            fig.write_html(save_path, include_plotlyjs="cdn")
        print(f"[moon_orbit_plotly] Saved -> {save_path}")
    if show:
        fig.show()
    return fig
