"""
core/sun.py
-----------
Sun model and shadow-terminator utilities for SSAPy-Toolkit 3D Plotly plots.

Public API
----------
  sun_position_eci(epoch_jd)            → (unit_vec: ndarray, dist_au: float)
  SunLayer(sun_pos_eci, ...)            → .build_traces() → list[go.Surface | go.Scatter3d]
  EarthShadingLayer(sun_pos_eci, ...)   → .build_traces() → list[go.Surface]
  MoonShadingLayer(sun_pos_eci, ...)    → .build_traces() → list[go.Surface]
  shadow_cone_params(sun_pos, R_body)   → dict with umbra / penumbra geometry

Drop-in with the existing layers API:
    scene.add_layer("sun")          # auto-detects sun position from OrbitalState epoch
    scene.add_layer("earth_shadow")
    scene.add_layer("moon_shadow")  # only useful for cislunar / moon plots

Or call directly:
    from core.sun import SunLayer, EarthShadingLayer
    traces = SunLayer(sun_hat * VISUAL_DIST_KM).build_traces()
    fig.add_traces(traces)
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Physical / display constants
# ---------------------------------------------------------------------------

R_EARTH_KM    = 6_371.0          # mean Earth radius
R_MOON_KM     = 1_737.4          # mean Moon radius
R_SUN_KM      = 695_700.0        # true solar radius (not used for display)
AU_KM         = 149_597_870.7    # 1 AU in km

# Visual sun placement: distance in the scene (not to scale; chosen so the
# sun sphere is clearly visible without overwhelming LEO / GEO scenes).
# Increase for cislunar scenes where axes span hundreds of thousands of km.
VISUAL_DIST_KM_LEO      = 80_000.0    # good for LEO / GEO
VISUAL_DIST_KM_CISLUNAR = 600_000.0   # good for cislunar / Moon plots
VISUAL_SUN_RADIUS_KM    = 5_500.0     # display radius (obviously not 1:1 scale)

# Corona glow rings: (scale_factor_vs_core, opacity, colour)
_CORONA_RINGS = [
    (1.35, 0.22, "rgba(255, 200,  60, {a})"),
    (1.80, 0.12, "rgba(255, 150,  20, {a})"),
    (2.60, 0.05, "rgba(255, 100,   0, {a})"),
]

# Shadow terminator: surface colorscale going from transparent (day side)
# to a deep blue-black (night side)
_EARTH_SHADOW_COLORSCALE = [
    [0.00, "rgba(  0,   0,  30, 0.00)"],   # full sun  – transparent
    [0.40, "rgba(  0,   0,  20, 0.18)"],   # penumbra
    [0.72, "rgba(  0,   0,  15, 0.52)"],   # deep penumbra
    [1.00, "rgba(  0,   0,  10, 0.75)"],   # umbra core
]

_MOON_SHADOW_COLORSCALE = [
    [0.00, "rgba(  0,   0,  30, 0.00)"],
    [0.40, "rgba( 10,   5,  25, 0.15)"],
    [0.72, "rgba(  5,   2,  18, 0.48)"],
    [1.00, "rgba(  0,   0,   8, 0.70)"],
]


# ---------------------------------------------------------------------------
# Solar ephemeris  (low-precision, < 0.01° error for dates near J2000)
# ---------------------------------------------------------------------------

def sun_position_eci(epoch_jd: float) -> tuple[np.ndarray, float]:
    """Return geocentric ECI unit vector toward the Sun plus distance in AU.

    Uses the low-precision solar coordinates from Astronomical Algorithms
    (Meeus), accurate to ~0.01° for dates within a few centuries of J2000.

    Parameters
    ----------
    epoch_jd : float
        Julian Date (e.g. 2451545.0 = J2000.0 = 2000-Jan-1.5 TT)

    Returns
    -------
    sun_hat : ndarray, shape (3,)
        Unit vector from Earth to Sun in ECI (GCRS-like).
    dist_au : float
        Earth–Sun distance in AU.
    """
    T = (epoch_jd - 2_451_545.0) / 36_525.0          # Julian centuries from J2000

    # Geometric mean longitude of the Sun (degrees)
    L0 = (280.46646 + 36_000.76983 * T) % 360.0

    # Mean anomaly of the Sun (degrees)
    M_deg = (357.52911 + 35_999.05029 * T - 0.0001537 * T**2) % 360.0
    M = np.radians(M_deg)

    # Equation of center
    C = ((1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M)
         + (0.019993 - 0.000101 * T) * np.sin(2 * M)
         + 0.000289 * np.sin(3 * M))

    # Sun's true longitude and true anomaly
    sun_lon = L0 + C
    nu = M_deg + C

    # Sun–Earth distance (AU)
    e = 0.016708634 - 0.000042037 * T
    dist_au = 1.000001018 * (1 - e**2) / (1 + e * np.cos(np.radians(nu)))

    # Apparent longitude (aberration correction ~−0.00569°)
    omega = 125.04 - 1934.136 * T
    apparent_lon = sun_lon - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    # Mean obliquity of the ecliptic (degrees)
    eps0 = (23.0 + 26.0/60 + 21.448/3600
            - (46.8150/3600) * T
            - (0.00059/3600) * T**2
            + (0.001813/3600) * T**3)
    # Apparent obliquity
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))

    lon_r = np.radians(apparent_lon)
    eps_r = np.radians(eps)

    sun_x = np.cos(lon_r)
    sun_y = np.cos(eps_r) * np.sin(lon_r)
    sun_z = np.sin(eps_r) * np.sin(lon_r)

    sun_hat = np.array([sun_x, sun_y, sun_z])
    sun_hat /= np.linalg.norm(sun_hat)          # normalise (already ~1 but be safe)

    return sun_hat, dist_au


def jd_from_datetime(dt) -> float:
    """Convert a Python datetime (UTC) to Julian Date."""
    import datetime
    if not isinstance(dt, datetime.datetime):
        raise TypeError("dt must be a datetime.datetime")
    # Algorithm from Meeus Ch.7
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    day_frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0 +
                dt.microsecond / 3_600_000_000.0) / 24.0
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + dt.day + day_frac + B - 1524.5


# ---------------------------------------------------------------------------
# Helper: unit sphere mesh
# ---------------------------------------------------------------------------

def _unit_sphere_mesh(nu: int = 50, nv: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) arrays for a unit sphere surface."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    return x, y, z


# ---------------------------------------------------------------------------
# SunLayer
# ---------------------------------------------------------------------------

class SunLayer:
    """Plotly traces forming a glowing 3D sun model.

    The sun is rendered as:
      • A bright inner sphere (yellow → white core)
      • 3 nested corona rings with decreasing opacity and warm orange tones
      • A directional label annotation (optional)

    Parameters
    ----------
    sun_pos_eci : array-like, shape (3,)
        Position of the sun *centre* in the scene coordinate frame (km).
        Typically ``sun_hat * VISUAL_DIST_KM``.  The direction is what
        matters visually; the magnitude sets how far the sphere is placed.
    radius_km : float, optional
        Display radius of the core sun sphere (km).  Default 5 500 km.
    show_corona : bool
        Add concentric glow rings.  Default True.
    show_label : bool
        Add a text annotation "☀ Sun".  Default True.
    name_prefix : str
        Prefix for trace names / legendgroups.
    """

    def __init__(
        self,
        sun_pos_eci,
        radius_km: float = VISUAL_SUN_RADIUS_KM,
        show_corona: bool = True,
        show_label: bool = True,
        name_prefix: str = "Sun",
    ):
        self.pos = np.asarray(sun_pos_eci, dtype=float)
        self.radius = radius_km
        self.show_corona = show_corona
        self.show_label = show_label
        self.name_prefix = name_prefix

    # ------------------------------------------------------------------
    def _core_sphere(self) -> go.Surface:
        """Bright inner sphere with a yellow-white radial gradient."""
        x, y, z = _unit_sphere_mesh(nu=60, nv=30)
        # surfacecolor = cosine of angle from "facing camera" direction
        # We fake a specular highlight by using the z-component of the normal.
        # This gives a bright white centre and warm yellow edges.
        brightness = np.clip(0.6 + 0.4 * z, 0, 1)  # [0.6, 1.0]

        colorscale = [
            [0.0, "rgb(255, 180,  20)"],   # warm gold (limb)
            [0.5, "rgb(255, 220,  80)"],   # yellow
            [0.8, "rgb(255, 245, 180)"],   # pale yellow
            [1.0, "rgb(255, 255, 240)"],   # near-white core
        ]

        return go.Surface(
            x=self.pos[0] + self.radius * x,
            y=self.pos[1] + self.radius * y,
            z=self.pos[2] + self.radius * z,
            surfacecolor=brightness,
            colorscale=colorscale,
            cmin=0, cmax=1,
            showscale=False,
            opacity=1.0,
            name=self.name_prefix,
            legendgroup=self.name_prefix,
            showlegend=True,
            hoverinfo="name",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
            lightposition=dict(x=0, y=0, z=0),
        )

    def _corona_sphere(self, scale: float, opacity: float, colour_template: str) -> go.Surface:
        """One translucent corona halo shell."""
        x, y, z = _unit_sphere_mesh(nu=40, nv=20)
        r = self.radius * scale
        colour = colour_template.format(a=opacity)

        return go.Surface(
            x=self.pos[0] + r * x,
            y=self.pos[1] + r * y,
            z=self.pos[2] + r * z,
            surfacecolor=np.ones_like(x),
            colorscale=[[0, colour], [1, colour]],
            cmin=0, cmax=1,
            showscale=False,
            opacity=opacity,
            name=f"{self.name_prefix} corona",
            legendgroup=self.name_prefix,
            showlegend=False,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
        )

    def _label_scatter(self) -> go.Scatter3d:
        """Small ☀ annotation at the sun position."""
        return go.Scatter3d(
            x=[self.pos[0]],
            y=[self.pos[1]],
            z=[self.pos[2] + self.radius * 2.8],
            mode="text",
            text=["☀ Sun"],
            textfont=dict(color="rgba(255,230,100,0.85)", size=12),
            name=f"{self.name_prefix} label",
            legendgroup=self.name_prefix,
            showlegend=False,
            hoverinfo="skip",
        )

    def build_traces(self) -> list:
        """Return a list of Plotly traces ready to add to a Figure."""
        traces = [self._core_sphere()]
        if self.show_corona:
            for scale, opacity, colour in _CORONA_RINGS:
                traces.append(self._corona_sphere(scale, opacity, colour))
        if self.show_label:
            traces.append(self._label_scatter())
        return traces


# ---------------------------------------------------------------------------
# Shadow terminator helpers
# ---------------------------------------------------------------------------

def _shadow_surface_color(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sun_hat: np.ndarray,
    R: float,
) -> np.ndarray:
    """Compute per-vertex shadow intensity in [0, 1].

    0  → full sunlight (transparent overlay)
    1  → deep shadow / umbra (opaque dark overlay)

    The transition spans ~10° around the terminator for a soft penumbra.
    """
    # Outward unit normal = (x, y, z) / R
    dot = (x * sun_hat[0] + y * sun_hat[1] + z * sun_hat[2]) / R

    # Soft terminator: map [-1, +terminator_width] → [1, 0]
    # Penumbra width in dot-product units (cos): ~0.17 ≈ 10°
    width = 0.17
    shadow = np.clip((-dot) / (1.0 + width) + width / (1.0 + width), 0.0, 1.0)

    # Gamma-correct for perceptual uniformity
    return shadow ** 0.7


# ---------------------------------------------------------------------------
# EarthShadingLayer
# ---------------------------------------------------------------------------

class EarthShadingLayer:
    """Translucent night-side shadow overlay for Earth.

    Renders a sphere (radius = R_EARTH_KM + 1 km clearance) with a
    per-vertex colorscale derived from the angle to the Sun.  The day side
    is fully transparent; the night side builds up to ~75 % dark opacity.

    Parameters
    ----------
    sun_pos_eci : array-like, shape (3,)
        Sun position in the scene frame (km).  Only the direction matters.
    earth_center : array-like, shape (3,), optional
        Earth centre in scene coordinates.  Default [0, 0, 0].
    radius_km : float, optional
        Overlay sphere radius.  Default R_EARTH_KM + 1.
    nu, nv : int
        Sphere mesh resolution.  Default 80 × 40 (smooth terminator).
    """

    def __init__(
        self,
        sun_pos_eci,
        earth_center=None,
        radius_km: float = R_EARTH_KM + 1.0,
        nu: int = 80,
        nv: int = 40,
    ):
        self.sun_hat = np.asarray(sun_pos_eci, dtype=float)
        if np.linalg.norm(self.sun_hat) > 0:
            self.sun_hat = self.sun_hat / np.linalg.norm(self.sun_hat)
        self.center = np.zeros(3) if earth_center is None else np.asarray(earth_center, dtype=float)
        self.radius = radius_km
        self.nu = nu
        self.nv = nv

    def build_traces(self) -> list:
        """Return a single go.Surface overlay trace."""
        x, y, z = _unit_sphere_mesh(self.nu, self.nv)
        shadow = _shadow_surface_color(x, y, z, self.sun_hat, 1.0)

        return [go.Surface(
            x=self.center[0] + self.radius * x,
            y=self.center[1] + self.radius * y,
            z=self.center[2] + self.radius * z,
            surfacecolor=shadow,
            colorscale=_EARTH_SHADOW_COLORSCALE,
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=1.0,          # per-pixel opacity encoded in colorscale alpha
            name="Earth shadow",
            legendgroup="Earth shadow",
            showlegend=True,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
        )]


# ---------------------------------------------------------------------------
# MoonShadingLayer
# ---------------------------------------------------------------------------

class MoonShadingLayer:
    """Translucent night-side shadow overlay for the Moon.

    Parameters
    ----------
    sun_pos_eci : array-like, shape (3,)
        Sun position in the scene frame (km).
    moon_center : array-like, shape (3,)
        Moon centre in scene frame (km).  Required — Moon is not at origin.
    radius_km : float, optional
        Overlay sphere radius.  Default R_MOON_KM + 0.5.
    nu, nv : int
        Mesh resolution.  Default 60 × 30.
    """

    def __init__(
        self,
        sun_pos_eci,
        moon_center,
        radius_km: float = R_MOON_KM + 0.5,
        nu: int = 60,
        nv: int = 30,
    ):
        self.moon_center = np.asarray(moon_center, dtype=float)
        # Sun direction as seen from the Moon
        sun_from_moon = np.asarray(sun_pos_eci, dtype=float) - self.moon_center
        norm = np.linalg.norm(sun_from_moon)
        self.sun_hat = sun_from_moon / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        self.radius = radius_km
        self.nu = nu
        self.nv = nv

    def build_traces(self) -> list:
        """Return a single go.Surface overlay trace."""
        x, y, z = _unit_sphere_mesh(self.nu, self.nv)
        shadow = _shadow_surface_color(x, y, z, self.sun_hat, 1.0)

        return [go.Surface(
            x=self.moon_center[0] + self.radius * x,
            y=self.moon_center[1] + self.radius * y,
            z=self.moon_center[2] + self.radius * z,
            surfacecolor=shadow,
            colorscale=_MOON_SHADOW_COLORSCALE,
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=1.0,
            name="Moon shadow",
            legendgroup="Moon shadow",
            showlegend=True,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
        )]


# ---------------------------------------------------------------------------
# Umbra / penumbra cone geometry (for pass prediction / eclipse detection)
# ---------------------------------------------------------------------------

def shadow_cone_params(
    sun_pos_km: np.ndarray,
    R_body_km: float,
    R_sun_km: float = R_SUN_KM,
) -> dict:
    """Compute umbra and penumbra cone geometry for a spherical body.

    Parameters
    ----------
    sun_pos_km : ndarray, shape (3,)
        Sun position in the body-centred frame (km).
    R_body_km : float
        Radius of the occulting body (km).
    R_sun_km : float, optional
        Solar radius (km).  Default 695 700 km.

    Returns
    -------
    dict with keys:
        sun_hat        : unit vector toward Sun
        D              : Earth–Sun distance (km)
        umbra_half_angle   : half-angle of umbra cone (rad)
        penumbra_half_angle: half-angle of penumbra cone (rad)
        umbra_length   : length of umbra cone from body centre (km)
        in_umbra(r)    : callable → bool, True if point r is in umbra
        in_penumbra(r) : callable → bool
    """
    sun = np.asarray(sun_pos_km, dtype=float)
    D = float(np.linalg.norm(sun))
    sun_hat = sun / D

    # Umbra: sin(half_angle) = (R_sun - R_body) / D
    umbra_ha = np.arcsin(max((R_sun_km - R_body_km) / D, 0.0))
    # Penumbra: sin(half_angle) = (R_sun + R_body) / D
    penumbra_ha = np.arcsin((R_sun_km + R_body_km) / D)

    # Umbra cone apex distance from body centre
    umbra_len = R_body_km / np.sin(umbra_ha) if umbra_ha > 0 else np.inf

    def _in_cone(r: np.ndarray, half_angle: float, length: float) -> bool:
        """True if point r lies inside the cone (shadow region)."""
        r = np.asarray(r, dtype=float)
        # Only shadow is on anti-sun side
        along = -np.dot(r, sun_hat)          # positive = anti-sun side
        if along <= 0:
            return False
        perp = np.sqrt(max(np.dot(r, r) - along**2, 0.0))
        cone_radius_at_along = along * np.tan(half_angle)
        return perp <= cone_radius_at_along and along <= length

    return {
        "sun_hat":                sun_hat,
        "D_km":                   D,
        "umbra_half_angle_rad":   umbra_ha,
        "penumbra_half_angle_rad": penumbra_ha,
        "umbra_length_km":        umbra_len,
        "in_umbra":               lambda r: _in_cone(r, umbra_ha, umbra_len),
        "in_penumbra":            lambda r: _in_cone(r, penumbra_ha, D * 10),
    }


# ---------------------------------------------------------------------------
# Convenience: auto-detect visual distance from scene scale
# ---------------------------------------------------------------------------

def auto_sun_position(sun_hat: np.ndarray, scene_radius_km: float) -> np.ndarray:
    """Place the visual sun at 12× the scene radius in the sun direction.

    Parameters
    ----------
    sun_hat : ndarray
        Unit vector toward the Sun.
    scene_radius_km : float
        Approximate radius of the plotted region (e.g. 8 000 for LEO,
        400 000 for cislunar).
    """
    dist = max(scene_radius_km * 12.0, VISUAL_DIST_KM_LEO)
    return sun_hat * dist


def auto_sun_radius(scene_radius_km: float) -> float:
    """Return a display sun radius proportional to the scene scale."""
    return max(scene_radius_km * 0.07, VISUAL_SUN_RADIUS_KM * 0.5)