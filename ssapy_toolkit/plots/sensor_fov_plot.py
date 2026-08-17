"""
ssapy_toolkit/plots/sensor_fov_plot.py
=======================================
Sensor Field-of-View cone plot — standalone script.

Run via the GUI "Export Plots" tab, or directly:

    python -m ssapy_toolkit.plots.sensor_fov_plot

Reads GUI_CONFIG from environment (injected by toolkit_gui.py run_script()),
or falls back to sensible defaults for standalone testing.

Output
------
  <output_dir>/sensor_fov_plot.jpg   — static snapshot at `time_index`
  <output_dir>/sensor_fov_anim.html  — interactive animated HTML (optional)

Pointing modes
--------------
  nadir         — cone points toward Earth centre  (Earth-observing sensor)
  anti-nadir    — cone points away from Earth       (star tracker / deep-space)
  velocity      — cone points along velocity vector (ram direction)
  anti-velocity — cone points anti-ram              (de-orbit engine exhaust etc.)
  custom        — cone points along a fixed GCRF unit vector you supply
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

try:
    from .eclipse_brightness_plot import sun_direction_eci
except ImportError:
    from eclipse_brightness_plot import sun_direction_eci

from .plotutils import (
    _pop_save_path_aliases,
    normalize_orbit_trajectory,
    save_plotly_figure,
)

# ---------------------------------------------------------------------------
# SSAPy imports (confirmed working API from checkpoint)
# ---------------------------------------------------------------------------
try:
    import ssapy
    import astropy.time
    _SSAPY_OK = True
except ImportError as _e:
    _SSAPY_OK = False
    _SSAPY_ERR = str(_e)

# ---------------------------------------------------------------------------
# Core layer imports
# ---------------------------------------------------------------------------
try:
    from .layers import SensorFOVLayer          # added this session
    _LAYER_OK = True
except ImportError:
    # Inline fallback — copy of the geometry helpers so the script is self-contained
    _LAYER_OK = False

# ---------------------------------------------------------------------------
# Inline geometry if core import failed
# ---------------------------------------------------------------------------
if not _LAYER_OK:
    def _orthonormal_basis(axis):
        axis = axis / np.linalg.norm(axis)
        helper = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
        u = np.cross(axis, helper); u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        return u, v

    def _cone_mesh3d(apex, direction, half_angle_deg, length_km,
                     n_sides=64, color="#00CED1", opacity=0.35, name="Sensor FOV"):
        apex = np.asarray(apex, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        radius = length_km * math.tan(math.radians(half_angle_deg))
        base_centre = apex + direction * length_km
        u, v = _orthonormal_basis(direction)
        angles = np.linspace(0., 2.*math.pi, n_sides, endpoint=False)
        rim = (base_centre[None, :]
               + radius * (np.cos(angles)[:, None] * u[None, :]
                           + np.sin(angles)[:, None] * v[None, :]))
        vx = np.empty(n_sides + 2); vy = np.empty(n_sides + 2); vz = np.empty(n_sides + 2)
        vx[0], vy[0], vz[0] = apex
        vx[1:n_sides+1] = rim[:, 0]; vy[1:n_sides+1] = rim[:, 1]; vz[1:n_sides+1] = rim[:, 2]
        ci = n_sides + 1; vx[ci], vy[ci], vz[ci] = base_centre
        ti, tj, tk = [], [], []
        for s in range(n_sides):
            c = s+1; nx = (s+1)%n_sides+1
            ti+=[0, ci]; tj+=[c, c]; tk+=[nx, nx]
        return go.Mesh3d(x=vx, y=vy, z=vz, i=ti, j=tj, k=tk,
                         color=color, opacity=opacity, name=name, showlegend=True,
                         flatshading=False,
                         lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5))

    class SensorFOVLayer:
        POINTING_MODES = ("nadir","anti-nadir","velocity","anti-velocity","custom")
        def __init__(self, r_gcrf_km, v_gcrf_kms=None, time_index=0,
                     half_angle_deg=15., cone_length_km=20_000., pointing_mode="nadir",
                     custom_direction=(1.,0.,0.), color="#00CED1", opacity=0.35,
                     n_sides=64, name="Sensor FOV", show_boresight=True, boresight_color="#FFFFFF"):
            self.r = np.asarray(r_gcrf_km, float)
            self.v = np.asarray(v_gcrf_kms, float) if v_gcrf_kms is not None else None
            self.time_index = int(time_index); self.ha = float(half_angle_deg)
            self.length = float(cone_length_km); self.mode = pointing_mode
            self.custom = np.asarray(custom_direction, float)
            self.color = color; self.opacity = opacity; self.n = int(n_sides)
            self.name = name; self.show_bs = show_boresight; self.bs_color = boresight_color
        @property
        def apex_km(self): return self.r[self.time_index]
        @property
        def boresight_direction(self):
            r = self.r[self.time_index]
            m = self.mode
            if   m == "nadir":         d = -r
            elif m == "anti-nadir":    d =  r
            elif m == "velocity":      d =  self.v[self.time_index]
            elif m == "anti-velocity": d = -self.v[self.time_index]
            elif m == "custom":        d = self.custom.copy()
            else: raise ValueError(m)
            return d / np.linalg.norm(d)
        def traces(self):
            apex = self.apex_km; direction = self.boresight_direction
            out = [_cone_mesh3d(apex, direction, self.ha, self.length, self.n,
                                self.color, self.opacity, self.name)]
            if self.show_bs:
                tip = apex + direction * self.length
                out.append(go.Scatter3d(x=[apex[0],tip[0]], y=[apex[1],tip[1]],
                                        z=[apex[2],tip[2]], mode="lines",
                                        line=dict(color=self.bs_color, width=2, dash="dot"),
                                        name="Boresight", showlegend=False))
            return out


# ---------------------------------------------------------------------------
# Propagation helper
# ---------------------------------------------------------------------------
MU_M3S2 = 3.986004418e14   # m³/s²

def propagate_orbit(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate orbit using the confirmed SSAPy API.
    Returns r_km (N,3) and v_kms (N,3).
    """
    if not _SSAPY_OK:
        raise RuntimeError(f"ssapy not available: {_SSAPY_ERR}")

    a_km    = float(cfg["a_km"])
    e       = float(cfg["e"])
    inc_deg = float(cfg["inc_deg"])
    argp_deg= float(cfg.get("argp_deg", 0.0))
    raan_deg= float(cfg.get("raan_deg", 0.0))
    nu_deg  = float(cfg.get("nu_deg",  0.0))
    epoch   = cfg.get("epoch", "2025-04-25 00:00:00")
    n_orbits= float(cfg.get("n_orbits", 3.0))
    dt_s    = float(cfg.get("dt_s", 10.0))

    a_m  = a_km * 1e3
    T_s  = 2.0 * math.pi * math.sqrt(a_m**3 / MU_M3S2)
    n_steps = max(100, int(round(n_orbits * T_s / dt_s)))

    t0    = astropy.time.Time(epoch, format="iso", scale="utc")
    t_gps = t0.gps + np.arange(n_steps) * dt_s

    orbit  = ssapy.Orbit.fromKeplerianElements(
        a_m, e,
        math.radians(inc_deg), math.radians(argp_deg),
        math.radians(raan_deg), math.radians(nu_deg),
        t0.gps,
    )
    orbits = orbit.at(t_gps)
    r_m = np.array([o.r for o in orbits])   # (N,3) metres GCRF
    v_ms = np.array([o.v for o in orbits])  # (N,3) m/s GCRF
    return r_m / 1e3, v_ms / 1e3            # → km, km/s


# ---------------------------------------------------------------------------
# Earth sphere helper
# ---------------------------------------------------------------------------
def _earth_sphere(r_km: float = 6371.0, n: int = 60) -> go.Surface:
    phi   = np.linspace(0, math.pi, n)
    theta = np.linspace(0, 2*math.pi, n)
    PHI, THETA = np.meshgrid(phi, theta)
    X = r_km * np.sin(PHI) * np.cos(THETA)
    Y = r_km * np.sin(PHI) * np.sin(THETA)
    Z = r_km * np.cos(PHI)
    return go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0,"#1a3a5c"],[0.4,"#1f6f3e"],[1,"#f0f0f0"]],
        showscale=False, name="Earth", opacity=1.0,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
    )


def _sensor_fov_traces(layer):
    """Return traces from either the shared layer or the local fallback."""
    if hasattr(layer, "build_traces"):
        return layer.build_traces()
    return layer.traces()


def _default_sun_direction_gcrf():
    return np.asarray(sun_direction_eci(np.array([0.0]))[0], dtype=float)


def _satellite_marker(pos):
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode="markers",
        marker=dict(size=7, color="#FFD700", symbol="circle", line=dict(color="#FFFFFF", width=1)),
        name="Satellite",
        showlegend=True,
    )


def _target_marker(pos, earth_radius_km=6371.0):
    norm = np.linalg.norm(pos)
    if norm <= earth_radius_km:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", showlegend=False)
    target = np.asarray(pos, dtype=float) / norm * earth_radius_km * 1.012
    return go.Scatter3d(
        x=[target[0]], y=[target[1]], z=[target[2]],
        mode="markers",
        marker=dict(size=5, color="#FF4D4D", symbol="diamond", line=dict(color="#FFFFFF", width=1)),
        name="Boresight ground intercept",
        showlegend=True,
    )


def _surface_track_trace(r_km, earth_radius_km=6371.0):
    norms = np.linalg.norm(r_km, axis=1)
    valid = norms > earth_radius_km
    if not np.any(valid):
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", showlegend=False)
    pts = r_km[valid] / norms[valid, None] * earth_radius_km * 1.006
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(color="#FFB000", width=4),
        name="Subsatellite ground track",
        showlegend=True,
    )


def _sensor_scene_traces(layer, pos):
    traces = list(_sensor_fov_traces(layer))
    if not traces or getattr(traces[0], "name", None) != "Satellite":
        traces.insert(0, _satellite_marker(pos))
    traces.append(_target_marker(pos))
    return traces


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
def build_figure(cfg: dict, r_km: np.ndarray, v_kms: np.ndarray) -> go.Figure:
    """
    Build the full 3D scene with orbit, satellite marker, and FOV cone.
    """
    time_index    = int(cfg.get("fov_time_index", 0))
    half_angle    = float(cfg.get("fov_half_angle_deg", 15.0))
    cone_length   = float(cfg.get("fov_cone_length_km", 20_000.0))
    pointing_mode = cfg.get("fov_pointing_mode", "nadir")
    custom_dir    = cfg.get("fov_custom_direction", [1.0, 0.0, 0.0])
    fov_color     = cfg.get("fov_color", "#00CED1")
    fov_opacity   = float(cfg.get("fov_opacity", 0.35))
    show_bs       = bool(cfg.get("fov_show_boresight", True))
    animate       = bool(cfg.get("fov_animate", False))
    anim_step     = int(cfg.get("fov_anim_step", 10))
    bg_color      = cfg.get("bg_color", "#0a0a14")
    sun_hat       = np.asarray(cfg.get("sun_direction_gcrf", _default_sun_direction_gcrf()), dtype=float)

    N = len(r_km)
    time_index = max(0, min(time_index, N - 1))

    fig = go.Figure()

    # Earth
    fig.add_trace(_earth_sphere())

    # Orbit path
    fig.add_trace(go.Scatter3d(
        x=r_km[:, 0], y=r_km[:, 1], z=r_km[:, 2],
        mode="lines",
        line=dict(color="#00BFFF", width=2),
        name="Orbit",
    ))
    fig.add_trace(_surface_track_trace(r_km))

    # Satellite marker, sensor cone, boresight, and Earth footprint.
    pos = r_km[time_index]
    fov = SensorFOVLayer(
        r_gcrf_km     = r_km,
        v_gcrf_kms    = v_kms,
        time_index    = time_index,
        half_angle_deg= half_angle,
        cone_length_km= cone_length,
        pointing_mode = pointing_mode,
        custom_direction = custom_dir,
        color         = fov_color,
        opacity       = fov_opacity,
        show_boresight= show_bs,
        sun_direction_gcrf = sun_hat,
        show_sun_shading = True,
        show_footprint = True,
    )
    dynamic_start = len(fig.data)
    sensor_traces = _sensor_scene_traces(fov, pos)
    dynamic_trace_indices = list(range(dynamic_start, dynamic_start + len(sensor_traces)))
    for t in sensor_traces:
        fig.add_trace(t)

    # Animation frames
    if animate:
        indices = range(0, N, anim_step)
        frames = []
        for idx in indices:
            fov.time_index = idx
            pos_f = r_km[idx]
            frame_data = _sensor_scene_traces(fov, pos_f)
            frames.append(go.Frame(data=frame_data, name=str(idx),
                                   traces=dynamic_trace_indices))
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False, y=0.02, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[str(i)], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=True))],
                            label=str(i)) for i in indices],
                transition=dict(duration=0), x=0.05, len=0.9, y=0.0,
            )],
        )

    # Layout
    axis_range = float(cfg.get("axis_range_km", max(np.linalg.norm(r_km, axis=1).max() * 1.15, 8000.)))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-axis_range, axis_range], showgrid=False,
                       showbackground=False, zeroline=False, title="X (km)"),
            yaxis=dict(range=[-axis_range, axis_range], showgrid=False,
                       showbackground=False, zeroline=False, title="Y (km)"),
            zaxis=dict(range=[-axis_range, axis_range], showgrid=False,
                       showbackground=False, zeroline=False, title="Z (km)"),
            bgcolor=bg_color,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=0.6, z=0.8)),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color="#CCCCCC"),
        title=dict(
            text=(f"Earth-observation Sensor FOV — {pointing_mode.title()} pointing | "
                  f"Footprint, ground track, and cone | Step {time_index}/{N-1}"),
            x=0.5,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#333", borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_sensor_fov(
    orbit=None,
    r=None,
    v=None,
    t=None,
    cfg: dict | None = None,
    save_path=None,
    show: bool = False,
    r_units: str = "auto",
    v_units: str = "auto",
    **kwargs,
) -> go.Figure:
    """Build a sensor field-of-view plot from SSAPy or array trajectory data.

    Parameters
    ----------
    orbit : ssapy.Orbit, optional
        SSAPy orbit object.  If ``t`` is omitted, one orbit is sampled using
        the configured ``n_orbits``/``dt_s`` behavior.
    r, v, t : array-like, optional
        Position, velocity, and time outputs from SSAPy.  Raw SSAPy ``r``/``v``
        arrays are metres and metres/second; these are auto-detected.  Already
        converted km and km/s arrays also work.
    cfg : dict, optional
        Plot configuration.  Keyword arguments override entries in this dict.
    save_path : path-like, optional
        Save as HTML or static image.  Standard aliases such as ``save=`` and
        ``savefig=`` are also accepted through ``kwargs``.
    """
    save_path, kwargs = _pop_save_path_aliases(kwargs, save_path=save_path)
    plot_cfg = DEFAULT_CFG.copy()
    if cfg:
        plot_cfg.update(cfg)
    plot_cfg.update(kwargs)

    if orbit is not None or r is not None:
        samples = int(plot_cfg.get("n_steps", 360))
        if "n_steps" not in plot_cfg and plot_cfg.get("dt_s") and orbit is not None:
            period = getattr(orbit, "period", None)
            try:
                samples = max(2, int(round(float(plot_cfg.get("n_orbits", 1.0)) * float(period) / float(plot_cfg["dt_s"]))))
            except Exception:
                samples = 360
        r_km, v_kms, t_time = normalize_orbit_trajectory(
            orbit=orbit,
            r=r,
            v=v,
            t=t,
            require_velocity=True,
            r_units=r_units,
            v_units=v_units,
            n_steps=samples,
            n_orbits=float(plot_cfg.get("n_orbits", 1.0)),
        )
        if "sun_direction_gcrf" not in plot_cfg and len(t_time):
            plot_cfg["sun_direction_gcrf"] = sun_direction_eci(
                np.array([0.0]),
                epoch_jd=float(t_time[0].jd),
            )[0]
    else:
        r_km, v_kms = propagate_orbit(plot_cfg)

    fig = build_figure(plot_cfg, r_km, v_kms)
    if save_path:
        save_plotly_figure(fig, save_path=save_path, default_name="sensor_fov_plot")
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CFG = dict(
    # Orbit (ISS-like LEO)
    a_km=6786.1, e=0.0003, inc_deg=51.6, argp_deg=0.0, raan_deg=0.0, nu_deg=0.0,
    epoch="2025-04-25 00:00:00", n_orbits=2.0, dt_s=10.0,
    # FOV
    fov_time_index=0,
    fov_half_angle_deg=15.0,
    fov_cone_length_km=8_000.0,
    fov_pointing_mode="nadir",
    fov_custom_direction=[1.0, 0.0, 0.0],
    fov_color="#00CED1",
    fov_opacity=0.35,
    fov_show_boresight=True,
    fov_animate=False,
    fov_anim_step=15,
    # Display
    bg_color="#0a0a14",
    axis_range_km=9000.0,
    # Output
    output_dir=str(Path.home() / "ssatk_figures" / "demo_gallery" / "figures"),
)


if __name__ == "__main__":
    import json

    # --- Load config from GUI or fall back to defaults ---
    cfg = DEFAULT_CFG.copy()
    env_cfg = os.environ.get("GUI_CONFIG", "")
    if env_cfg:
        try:
            cfg.update(json.loads(env_cfg))
            print("[sensor_fov_plot] Loaded GUI_CONFIG from environment.")
        except json.JSONDecodeError as e:
            print(f"[sensor_fov_plot] Warning: could not parse GUI_CONFIG ({e}); using defaults.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Propagate ---
    print(f"[sensor_fov_plot] Propagating orbit  (a={cfg['a_km']} km, "
          f"inc={cfg['inc_deg']}°, n_orbits={cfg['n_orbits']}) ...")
    if not _SSAPY_OK:
        print(f"[sensor_fov_plot] ERROR: ssapy not available — {_SSAPY_ERR}")
        raise RuntimeError(f"ssapy not available: {_SSAPY_ERR}")

    r_km, v_kms = propagate_orbit(cfg)
    N = len(r_km)
    print(f"[sensor_fov_plot]   → {N} steps propagated.")

    # Clamp time_index to valid range
    cfg["fov_time_index"] = max(0, min(int(cfg["fov_time_index"]), N - 1))
    print(f"[sensor_fov_plot] Sensor at step {cfg['fov_time_index']}/{N-1}  "
          f"pointing={cfg['fov_pointing_mode']}  "
          f"half-angle={cfg['fov_half_angle_deg']}°  "
          f"length={cfg['fov_cone_length_km']} km")

    # --- Build figure ---
    print("[sensor_fov_plot] Building Plotly scene ...")
    fig = build_figure(cfg, r_km, v_kms)

    # --- Save static JPEG ---
    jpg_path = output_dir / "sensor_fov_plot.jpg"
    try:
        fig.write_image(str(jpg_path), width=1920, height=1080, scale=2)
        print(f"[sensor_fov_plot] Saved → {jpg_path}")
    except Exception as e:
        print(f"[sensor_fov_plot] write_image failed ({e}). "
              f"Try: pip install kaleido --break-system-packages")

    # --- Save interactive HTML ---
    html_path = output_dir / "sensor_fov_plot.html"
    fig.write_html(str(html_path))
    print(f"[sensor_fov_plot] Saved → {html_path}")
    print("[sensor_fov_plot] Done.")
