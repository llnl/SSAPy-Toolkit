"""
globe_orbit_daynight_plotly.py — Plotly version, standalone
================================================================
Same content as globe_orbit_daynight_plot.py (Earth with day/night
terminator, an orbit around it, and the Sun rendered as a real body in
frame), but in Plotly instead of matplotlib. SSAPy and SSAPy-Toolkit are used
when available for texture discovery and Earth orientation; deterministic
fallbacks keep the renderer usable without either optional package.

Earth and Sun are both genuine 3D mesh/sphere geometry (go.Mesh3d /
go.Surface), matching the real 3D sphere approach used in the toolkit's
own layers.py (EarthLayer / SunLayer), not flat markers.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import warnings

import numpy as np
import plotly.graph_objects as go

from .plotutils import (
    _pop_save_path_aliases,
    _raise_unrecognized_kwargs,
    normalize_orbit_trajectory,
)
from .scene_primitives import (
    earth_rotation_deg_from_time,
    sun_position_and_radius,
)

try:
    from ssapy_toolkit.io.eclipse_asset_resolver import asset_candidates
except ImportError:
    from ssapy_toolkit.io.eclipse_asset_resolver import asset_candidates

# Shared eclipse physics are optional for standalone globe plots.  Import the
# module first, then retrieve newer helpers with ``getattr`` so an older
# eclipse_brightness_plot.py does not disable the still-compatible illumination
# function merely because it lacks the coordinate-transform wrapper.
try:
    import ssapy_toolkit.compute.eclipse_brightness as _eclipse_physics
except ImportError:
    try:
        import eclipse_brightness_plot as _eclipse_physics
    except ImportError:
        _eclipse_physics = None

if _eclipse_physics is not None:
    illumination_fraction = getattr(_eclipse_physics, "illumination_fraction", None)
    irradiance_fraction = getattr(_eclipse_physics, "irradiance_fraction", illumination_fraction)
    R_SUN_KM = float(getattr(_eclipse_physics, "R_SUN_KM", 695_700.0))
    AU_KM = float(getattr(_eclipse_physics, "AU_KM", 149_597_870.7))
    RP_EARTH_KM = float(getattr(_eclipse_physics, "RP_EARTH_KM", 6_356.752314245))
    _shared_sun_position_eci = getattr(_eclipse_physics, "sun_position_eci", None)
    _shared_itrf_to_gcrf_km = getattr(_eclipse_physics, "itrf_to_gcrf_km", None)
else:
    illumination_fraction = None
    irradiance_fraction = None
    _shared_sun_position_eci = None
    _shared_itrf_to_gcrf_km = None
    R_SUN_KM, AU_KM = 695_700.0, 149_597_870.7
    RP_EARTH_KM = 6_356.752314245


MU_EARTH_KM3S2 = 398_600.4418
RE_KM = 6_378.137


def propagate_eci(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg,
                   n_orbits=1.0, n_steps=1500):
    def _solve_kepler(M, e, tol=1e-10, max_iter=60):
        E = M.copy()
        for _ in range(max_iter):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if np.max(np.abs(dE)) < tol:
                break
        return E

    inc, raan, argp = np.radians([inc_deg, raan_deg, argp_deg])
    nu0 = np.radians(nu0_deg)
    E0 = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu0/2), np.sqrt(1+e)*np.cos(nu0/2))
    M0 = E0 - e*np.sin(E0)
    T_s = 2*np.pi*np.sqrt(a_km**3/MU_EARTH_KM3S2)
    t_s = np.linspace(0, n_orbits*T_s, n_steps)
    n_rad_s = np.sqrt(MU_EARTH_KM3S2/a_km**3)
    E = _solve_kepler(M0 + n_rad_s*t_s, e)
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r_mag = a_km*(1-e*np.cos(E))
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    R11 = cO*cw - sO*sw*ci; R12 = -cO*sw - sO*cw*ci
    R21 = sO*cw + cO*sw*ci; R22 = -sO*sw + cO*cw*ci
    R31 = sw*si;             R32 = cw*si
    xp, yp = r_mag*np.cos(nu), r_mag*np.sin(nu)
    x = R11*xp + R12*yp; y = R21*xp + R22*yp; z = R31*xp + R32*yp
    return t_s, np.stack([x, y, z], axis=1), T_s


def sun_direction_eci(t_s, epoch_jd=2_460_500.0):
    """GCRF Sun direction, delegating to the shared finite-distance model."""
    if _shared_sun_position_eci is not None:
        p = _shared_sun_position_eci(np.asarray(t_s, dtype=float), epoch_jd=epoch_jd)
        return p / np.linalg.norm(p, axis=-1, keepdims=True)
    jd = epoch_jd + np.asarray(t_s, dtype=float)/86400.0
    d = jd - 2_451_545.0
    L = np.radians((280.460 + 0.9856474*d) % 360)
    g = np.radians((357.528 + 0.9856003*d) % 360)
    lam = L + np.radians(1.915)*np.sin(g) + np.radians(0.020)*np.sin(2*g)
    eps = np.radians(23.439291 - 0.0000004*d)
    return np.stack([np.cos(lam), np.cos(eps)*np.sin(lam), np.sin(eps)*np.sin(lam)], axis=-1)


_earth_texture_cache: dict[tuple[str, int, int, int, int], np.ndarray] = {}
_earth_texture_source_cache: dict[tuple[str, int, int, int, int], str] = {}
_earth_base_printed: set[tuple[str, int, int]] = set()


def _normalize(v, *, name="vector"):
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must be a finite, non-zero vector")
    return arr / norm


def _smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        raise ValueError("edge1 must be larger than edge0")
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _srgb_to_linear(rgb):
    """Convert encoded sRGB values in [0, 1] to linear-light RGB."""
    x = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    """Convert linear-light RGB values to encoded sRGB in [0, 1]."""
    x = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _ocean_mask_from_texture(rgb_srgb):
    """Infer open ocean from the real Earth texture without fake geography.

    The official SSAPy texture uses very dark, blue-dominant ocean pixels.
    A smooth confidence mask is sufficient for a view-dependent water glint
    and avoids the old renderer's unrelated random land/ocean mask.
    """
    rgb = np.clip(np.asarray(rgb_srgb, dtype=float), 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    blue_excess = b - np.maximum(r, g)
    confidence = 7.5*blue_excess + 4.0*np.clip(0.22-luminance, 0.0, 0.22)
    confidence *= (
        (blue_excess > 0.004)
        & (b >= r)
        & (b >= 0.92*g)
        & (luminance < 0.30)
    )
    return np.clip(confidence, 0.0, 1.0)


def _texture_candidates(texture_path=None):
    """Yield Earth images through the shared SSAPy-Data resolver."""
    for path, _source, _root in asset_candidates(
        "earth_albedo", explicit=texture_path, policy="data-first"
    ):
        yield path


def _load_real_earth_texture(n_lat, n_lon, texture_path=None, *, return_source=False):
    """Load an equirectangular north-up Earth texture.

    The SSAPy ``earth.png`` convention is already longitude -180..+180 from
    left to right and latitude +90..-90 from top to bottom.  No 180-degree
    roll is applied.  The previous renderer rolled this texture by half its
    width, which placed every continent on the opposite meridian.
    """
    from PIL import Image

    n_lat = int(n_lat)
    n_lon = int(n_lon)
    if n_lat < 4 or n_lon < 8:
        raise ValueError("Earth texture resolution must be at least 4 x 8")

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    if texture_path is not None:
        explicit = Path(texture_path).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"Earth texture does not exist: {explicit}")
        candidates = (explicit,)
    else:
        candidates = tuple(_texture_candidates(None))

    last_error = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            stat = candidate.stat()
        except OSError as ex:
            last_error = ex
            continue
        # Include file identity in the cache key.  During iterative rendering a
        # texture can be replaced at the same path; a path-only cache silently
        # kept the old pixels for the rest of the Python process.
        key = (str(candidate), int(stat.st_mtime_ns), int(stat.st_size), n_lat, n_lon)
        if key not in _earth_texture_cache:
            try:
                with Image.open(candidate) as image:
                    image = image.convert("RGB")
                    if image.size != (n_lon, n_lat):
                        # Longitude is periodic.  Resizing one isolated copy
                        # lets the Lanczos kernel see an artificial hard edge at
                        # +/-180 degrees and can leave a visible antimeridian
                        # seam.  Resize three tiled copies and crop the middle.
                        width, height = image.size
                        tiled = Image.new("RGB", (width * 3, height))
                        tiled.paste(image, (0, 0))
                        tiled.paste(image, (width, 0))
                        tiled.paste(image, (2 * width, 0))
                        tiled = tiled.resize((n_lon * 3, n_lat), resampling)
                        image = tiled.crop((n_lon, 0, 2 * n_lon, n_lat))
                    array = np.asarray(image, dtype=np.uint8)
                    array.setflags(write=False)
                    _earth_texture_cache[key] = array
                    _earth_texture_source_cache[key] = str(candidate)
            except Exception as ex:
                last_error = ex
                continue
        result = _earth_texture_cache[key]
        return (result, _earth_texture_source_cache[key]) if return_source else result

    if last_error is not None:
        warnings.warn(f"Earth texture could not be loaded: {last_error}", RuntimeWarning)
    return (None, "procedural fallback") if return_source else None


def _smooth(field, sigma=1.0):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma, mode=("nearest", "wrap"))
    except Exception:
        return field


@lru_cache(maxsize=16)
def _procedural_earth_texture_cached(n_lat, n_lon, seed=7):
    """Deterministic emergency texture used only when no real asset exists."""
    lat = np.linspace(90.0, -90.0, n_lat)
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, lat)
    try:
        from global_land_mask import globe
        land = globe.is_land(np.clip(Lat, -89.999, 89.999), Lon)
        land = _smooth(land.astype(float), sigma=0.65)
    except Exception:
        # A low-frequency deterministic field is preferable to frame-to-frame
        # random continents.  It is clearly a fallback and is never mixed with
        # a real SSAPy texture for relief, glint, or city-light placement.
        rng = np.random.default_rng(seed)
        field = np.zeros_like(Lat)
        for _ in range(18):
            clat = rng.uniform(-60.0, 70.0)
            clon = rng.uniform(-180.0, 180.0)
            s_lat = rng.uniform(10.0, 28.0)
            s_lon = rng.uniform(14.0, 38.0)
            dlat = (Lat - clat) / s_lat
            dlon = ((Lon - clon + 180.0) % 360.0 - 180.0) / s_lon
            field += rng.uniform(0.7, 1.3) * np.exp(-0.5 * (dlat*dlat + dlon*dlon))
        land = _smooth((field > np.quantile(field, 0.63)).astype(float), sigma=0.9)

    ocean = np.array([0.018, 0.070, 0.145])
    land_green = np.array([0.105, 0.225, 0.090])
    desert = np.array([0.48, 0.38, 0.22])
    land3 = np.clip(land, 0.0, 1.0)[..., None]
    desert_band = np.clip(1.0 - np.abs(np.abs(Lat) - 24.0) / 19.0, 0.0, 1.0)[..., None]
    land_rgb = land_green * (1.0 - 0.65*desert_band) + desert * (0.65*desert_band)
    rgb = ocean * (1.0-land3) + land_rgb * land3
    ice = _smooth(_smoothstep(66.0, 82.0, np.abs(Lat)), sigma=0.35)[..., None]
    rgb = rgb*(1.0-ice) + np.array([0.88, 0.91, 0.95])*ice
    result = np.rint(np.clip(rgb, 0.0, 1.0)*255.0).astype(np.uint8)
    result.setflags(write=False)
    return result


def _procedural_earth_texture(n_lat, n_lon, seed=7):
    """Cached wrapper around the deterministic fallback texture."""
    return _procedural_earth_texture_cached(int(n_lat), int(n_lon), int(seed))


def _gmst_deg(jd):
    jd = float(jd)
    T = (jd - 2_451_545.0) / 36_525.0
    return (280.46061837 + 360.98564736629*(jd-2_451_545.0)
            + 0.000387933*T*T - T*T*T/38_710_000.0) % 360.0


def _nearest_rotation_rows(matrix):
    """Return the nearest proper orthogonal row-vector rotation matrix."""
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def _itrf_to_gcrf_rotation(time_jd=None, rotation_deg=0.0):
    """Row-vector rotation from ITRF/body-fixed coordinates to GCRF.

    When a Julian Date is supplied, the shared SSAPy-Toolkit/Astropy wrapper
    is used on the three Cartesian basis vectors, so the rendered texture,
    WGS-84 ellipsoid, and GCRF eclipse overlays use exactly the same Earth
    orientation model.  A continuous GMST rotation is the deterministic
    fallback; texture columns are never integer-rolled.
    """
    if time_jd is not None and _shared_itrf_to_gcrf_km is not None:
        try:
            basis = np.asarray(
                _shared_itrf_to_gcrf_km(np.eye(3), float(time_jd)), dtype=float
            ).reshape(3, 3)
            return _nearest_rotation_rows(basis)
        except Exception as ex:
            warnings.warn(
                f"High-fidelity ITRF->GCRF rotation failed; using GMST fallback: {ex}",
                RuntimeWarning,
            )
    angle = _gmst_deg(time_jd) if time_jd is not None else float(rotation_deg)
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # Rows are the GCRF images of ITRF basis vectors; row points transform
    # with ``points_gcrf = points_itrf @ rotation_rows``.
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _wgs84_vertices(n_lat, n_lon, *, altitude_km=0.0):
    """Create a seam-closed, non-degenerate WGS-84 geodetic mesh.

    Meshes are immutable and cached because static and animated scenes reuse
    the same resolutions many times.  Callers must treat returned arrays as
    read-only.
    """
    return _wgs84_vertices_cached(int(n_lat), int(n_lon), float(altitude_km))


@lru_cache(maxsize=32)
def _wgs84_vertices_cached(n_lat, n_lon, altitude_km):
    n_lat = int(n_lat)
    n_lon = int(n_lon)
    if n_lat < 4 or n_lon < 8:
        raise ValueError("Earth mesh resolution must be at least 4 x 8")
    # Include an exact equatorial ring.  With an even row count, linspace
    # misses latitude zero and makes the rendered equatorial radius slightly
    # smaller than WGS-84.
    if n_lat % 2 == 0:
        n_lat += 1

    lat_rows = np.linspace(90.0, -90.0, n_lat)
    ring_lat = lat_rows[1:-1]
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, ring_lat)
    phi = np.radians(Lat)
    lam = np.radians(Lon)

    # Geodetic height is measured along the ellipsoid normal.  Increasing
    # both semiaxes and recomputing flattening (the previous implementation)
    # does not produce a constant-altitude shell away from the equator and
    # poles.  Keep the WGS-84 ellipsoid fixed and apply the standard h terms.
    a = RE_KM
    b = RP_EARTH_KM
    h = float(altitude_km)
    if not np.isfinite(h) or h <= -b:
        raise ValueError("altitude_km must be finite and greater than -Earth polar radius")
    e2 = 1.0 - (b*b)/(a*a)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    N = a / np.sqrt(1.0 - e2*sin_phi*sin_phi)
    x = (N+h)*cos_phi*np.cos(lam)
    y = (N+h)*cos_phi*np.sin(lam)
    z = (N*(1.0-e2)+h)*sin_phi
    ring_vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # A geodetic surface normal has this simple direction even though the
    # position vector of an oblate ellipsoid is not radial at mid-latitudes.
    ring_normals = np.stack(
        [cos_phi*np.cos(lam), cos_phi*np.sin(lam), sin_phi], axis=-1
    ).reshape(-1, 3)

    north = np.array([[0.0, 0.0, b+h]])
    south = np.array([[0.0, 0.0, -(b+h)]])
    north_n = np.array([[0.0, 0.0, 1.0]])
    south_n = np.array([[0.0, 0.0, -1.0]])
    vertices = np.vstack([ring_vertices, north, south])
    normals = np.vstack([ring_normals, north_n, south_n])

    lat_v = np.concatenate([Lat.ravel(), [90.0, -90.0]])
    lon_v = np.concatenate([Lon.ravel(), [0.0, 0.0]])

    ring_count = n_lat - 2
    faces = []
    cols = np.arange(n_lon, dtype=int)
    cols_next = (cols + 1) % n_lon
    for row in range(ring_count - 1):
        top = row*n_lon
        bottom = (row+1)*n_lon
        faces.append(np.column_stack([
            top + cols, bottom + cols, bottom + cols_next
        ]))
        faces.append(np.column_stack([
            top + cols, bottom + cols_next, top + cols_next
        ]))
    north_idx = ring_count*n_lon
    south_idx = north_idx + 1
    faces.append(np.column_stack([
        np.full(n_lon, north_idx, dtype=int), cols_next, cols
    ]))
    last = (ring_count-1)*n_lon
    faces.append(np.column_stack([
        np.full(n_lon, south_idx, dtype=int), last+cols, last+cols_next
    ]))
    faces = np.vstack(faces).astype(np.int32)

    # Correct winding independently for the side and cap triangles.  This
    # removes the inward-facing cap and seam artifacts produced by the old
    # duplicated-pole latitude/longitude grid.
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    orient = np.einsum("ij,ij->i", np.cross(p1-p0, p2-p0), (p0+p1+p2)/3.0)
    flip = orient < 0.0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()
    for array in (vertices, normals, faces, lat_rows, lat_v, lon_v):
        array.setflags(write=False)
    return vertices, normals, faces, lat_rows, lat_v, lon_v


@dataclass
class EarthSurfaceData:
    display_vertices: np.ndarray
    physical_vertices_gcrf_km: np.ndarray
    normals_gcrf: np.ndarray
    faces: np.ndarray
    base_rgb_srgb: np.ndarray
    shaded_rgb_srgb: np.ndarray
    eclipse_visibility: np.ndarray
    ocean_mask: np.ndarray
    specular_linear: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    texture_source: str
    rotation_rows: np.ndarray
    n_lat_effective: int
    n_lon_effective: int


def _earth_surface_data(
    sun_hat,
    n_lat=180,
    n_lon=360,
    radius_scale=1.0,
    center=(0.0, 0.0, 0.0),
    shadow_body_center_km=None,
    shadow_body_radius_km=None,
    rotation_deg=0.0,
    time_jd=None,
    sun_position_km=None,
    physical_center_km=(0.0, 0.0, 0.0),
    night_floor=0.003,
    show_city_lights=False,
    texture_path=None,
    exposure=1.0,
    view_hat=None,
    specular_strength=0.42,
):
    """Compute geometry and physically ordered color layers for Earth.

    Texture values are converted from sRGB to linear light before day/night
    and eclipse attenuation is applied, then converted back to sRGB.  This
    replaces the old encoded-RGB multiplication plus global highlight-lifting
    curve, which washed out the dayside and made the nightside glow.
    """
    if show_city_lights:
        warnings.warn(
            "Synthetic city lights were removed because their random land mask "
            "did not align with the SSAPy texture. No night-light layer is applied.",
            RuntimeWarning,
        )

    sun_hat = _normalize(sun_hat, name="sun_hat")
    center = np.asarray(center, dtype=float).reshape(3)
    physical_center = np.asarray(physical_center_km, dtype=float).reshape(3)
    if sun_position_km is not None:
        # A finite Sun vector is authoritative: it defines both the local
        # lighting direction and the apparent solar angular radius.  Keeping a
        # separately supplied direction when the two disagree can put the
        # terminator and the Moon shadow on different parts of Earth.
        sun_absolute = np.asarray(sun_position_km, dtype=float).reshape(3)
        sun_hat = _normalize(sun_absolute - physical_center, name="Sun relative position")
    else:
        sun_absolute = physical_center + sun_hat * AU_KM
    view_hat = _normalize(view_hat if view_hat is not None else sun_hat, name="view_hat")
    radius_scale = float(radius_scale)
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("radius_scale must be finite and positive")

    n_lat = int(n_lat)
    n_lon = int(n_lon)
    body_vertices, body_normals, faces, lat_rows, lat_v, lon_v = _wgs84_vertices(
        n_lat, n_lon
    )
    rotation_rows = _itrf_to_gcrf_rotation(time_jd=time_jd, rotation_deg=rotation_deg)
    physical_local_gcrf = body_vertices @ rotation_rows
    normals_gcrf = body_normals @ rotation_rows
    normals_gcrf /= np.linalg.norm(normals_gcrf, axis=1, keepdims=True)
    physical_vertices = physical_center + physical_local_gcrf
    display_vertices = center + physical_local_gcrf*radius_scale

    n_lat_effective = len(lat_rows)
    texture, source = _load_real_earth_texture(
        n_lat_effective, n_lon, texture_path=texture_path, return_source=True
    )
    if texture is None:
        texture = _procedural_earth_texture(n_lat_effective, n_lon)
        source = "procedural fallback"
    print_key = (source, int(n_lat_effective), int(n_lon))
    if print_key not in _earth_base_printed:
        print(f"[globe_orbit_daynight_plotly] Earth base: {source}")
        _earth_base_printed.add(print_key)

    # The mesh excludes duplicate pole vertices. Texture rows are north to
    # south and longitudes are already -180..+180, matching the body grid.
    ring_rgb = texture[1:-1].reshape(-1, 3).astype(float)/255.0
    north_rgb = texture[0].mean(axis=0, keepdims=True).astype(float)/255.0
    south_rgb = texture[-1].mean(axis=0, keepdims=True).astype(float)/255.0
    base_rgb = np.vstack([ring_rgb, north_rgb, south_rgb])

    mu = np.clip(normals_gcrf @ sun_hat, -1.0, 1.0)
    # Lambertian diffuse response.  The former 0.82 power artificially
    # brightened grazing-incidence terrain and broadened the dayside.
    direct_geometric = np.clip(mu, 0.0, 1.0)
    direct = direct_geometric.copy()
    sun_altitude = np.arcsin(mu)
    sky = _smoothstep(np.radians(-6.0), np.radians(2.0), sun_altitude)

    eclipse_visibility = np.ones(len(display_vertices), dtype=float)
    if (shadow_body_center_km is not None and shadow_body_radius_km is not None
            and irradiance_fraction is not None):
        shadow_center = np.asarray(shadow_body_center_km, dtype=float).reshape(3)
        eval_from_occluder = physical_vertices - shadow_center
        # ``sun_absolute`` and ``shadow_center`` are in the same physical
        # frame.  The old fallback used ``sun_hat*AU`` directly and therefore
        # forgot to subtract the Moon's geocentric position, shifting the
        # apparent discs by up to roughly the lunar parallax scale.
        sun_from_occluder = sun_absolute - shadow_center
        sun_vectors = np.broadcast_to(sun_from_occluder, eval_from_occluder.shape)
        eclipse_visibility = np.asarray(
            irradiance_fraction(
                eval_from_occluder,
                R_body_km=float(shadow_body_radius_km),
                R_sun_km=R_SUN_KM,
                sun_position_km=sun_vectors,
                photometry="quadratic-visible",
                quadrature_order=48,
            ),
            dtype=float,
        ).reshape(-1)
        eclipse_visibility = np.clip(eclipse_visibility, 0.0, 1.0)

    # Direct sunlight is attenuated by the exact finite-disc overlap. Diffuse
    # atmospheric sky light remains weakly present in totality instead of an
    # arbitrary six-percent floor being multiplied into the entire surface.
    direct *= eclipse_visibility
    sky *= 0.15 + 0.85*np.sqrt(eclipse_visibility)
    night_floor = float(np.clip(night_floor, 0.0, 0.08))
    illumination = night_floor + 0.035*sky + 0.965*direct

    rgb_linear = _srgb_to_linear(base_rgb) * illumination[:, None]

    # Real view-dependent ocean sun glint.  It is tied to the blue, dark
    # ocean pixels in the SSAPy map and to the Sun/camera half-vector, so it
    # moves correctly with the camera and cannot appear over continents.
    ocean_mask = _ocean_mask_from_texture(base_rgb)
    half_vec = sun_hat + view_hat
    if np.linalg.norm(half_vec) > 1e-12:
        half_hat = half_vec / np.linalg.norm(half_vec)
        ndoth = np.clip(normals_gcrf @ half_hat, 0.0, 1.0)
        ndotv = np.clip(normals_gcrf @ view_hat, 0.0, 1.0)
        fresnel = 0.18 + 0.82*(1.0-ndotv)**5
        specular_linear = (
            float(np.clip(specular_strength, 0.0, 2.0))
            * ocean_mask * ndoth**150 * fresnel
            * direct_geometric * eclipse_visibility
        )
    else:
        specular_linear = np.zeros(len(base_rgb), dtype=float)
    rgb_linear += specular_linear[:, None]*np.array([0.72, 0.88, 1.00])
    # Keep surface color neutral at the terminator.  Atmospheric sunset color
    # belongs in the separate limb-scattering layer; adding it directly to
    # the ground produced a conspicuous orange stripe across the globe.
    rgb_linear *= float(np.clip(exposure, 0.1, 4.0))
    shaded_rgb = _linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))

    return EarthSurfaceData(
        display_vertices=display_vertices,
        physical_vertices_gcrf_km=physical_vertices,
        normals_gcrf=normals_gcrf,
        faces=faces,
        base_rgb_srgb=base_rgb,
        shaded_rgb_srgb=shaded_rgb,
        eclipse_visibility=eclipse_visibility,
        ocean_mask=ocean_mask,
        specular_linear=specular_linear,
        latitude_deg=lat_v,
        longitude_deg=lon_v,
        texture_source=source,
        rotation_rows=rotation_rows,
        n_lat_effective=int(n_lat_effective),
        n_lon_effective=int(n_lon),
    )


def _rgb_strings(rgb, alpha=None):
    rgb8 = np.rint(np.clip(rgb, 0.0, 1.0)*255.0).astype(np.uint8)
    if alpha is None:
        return [f"rgb({r},{g},{b})" for r, g, b in rgb8]
    alpha = np.clip(np.asarray(alpha, dtype=float).reshape(-1), 0.0, 1.0)
    return [f"rgba({r},{g},{b},{a:.4f})" for (r, g, b), a in zip(rgb8, alpha)]


def _earth_mesh(
    sun_hat,
    n_lat=180,
    n_lon=360,
    radius_scale=1.0,
    center=(0.0, 0.0, 0.0),
    shadow_body_center_km=None,
    shadow_body_radius_km=None,
    rotation_deg=0.0,
    time_jd=None,
    sun_position_km=None,
    physical_center_km=(0.0, 0.0, 0.0),
    night_floor=0.003,
    show_city_lights=False,
    texture_path=None,
    exposure=1.0,
    view_hat=None,
    specular_strength=0.42,
):
    """Render a WGS-84 Earth with correctly oriented texture and lighting.

    ``time_jd`` is preferred over ``rotation_deg`` because it lets the shared
    SSAPy-Toolkit coordinate transform orient the body in GCRF.  The legacy
    rotation argument remains for standalone schematic calls.
    """
    data = _earth_surface_data(
        sun_hat,
        n_lat=n_lat,
        n_lon=n_lon,
        radius_scale=radius_scale,
        center=center,
        shadow_body_center_km=shadow_body_center_km,
        shadow_body_radius_km=shadow_body_radius_km,
        rotation_deg=rotation_deg,
        time_jd=time_jd,
        sun_position_km=sun_position_km,
        physical_center_km=physical_center_km,
        night_floor=night_floor,
        show_city_lights=show_city_lights,
        texture_path=texture_path,
        exposure=exposure,
        view_hat=view_hat,
        specular_strength=specular_strength,
    )
    v = data.display_vertices
    f = data.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        vertexcolor=_rgb_strings(data.shaded_rgb_srgb),
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        name="Earth", hoverinfo="skip", showlegend=False,
    )


def _earth_atmosphere_trace(
    center=(0.0, 0.0, 0.0),
    radius_scale=1.0,
    *,
    sun_hat=(1.0, 0.0, 0.0),
    view_hat=None,
    time_jd=None,
    rotation_deg=0.0,
    n_lat=54,
    n_lon=108,
    altitude_km=105.0,
    max_alpha=0.34,
):
    """Camera-aware atmospheric limb glow rather than a uniform blue shell.

    The old atmosphere trace placed a constant-opacity sphere over the whole
    planet, so front and back surfaces blended into a blue haze across the
    disk. Here alpha approaches zero at the apparent disk centre and rises at
    the limb, with stronger scattering on the sunlit side.
    """
    sun_hat = _normalize(sun_hat, name="sun_hat")
    view_hat = _normalize(view_hat if view_hat is not None else sun_hat, name="view_hat")
    body_vertices, body_normals, faces, _, _, _ = _wgs84_vertices(
        n_lat, n_lon, altitude_km=altitude_km
    )
    rotation_rows = _itrf_to_gcrf_rotation(time_jd=time_jd, rotation_deg=rotation_deg)
    local_gcrf = body_vertices @ rotation_rows
    normals = body_normals @ rotation_rows
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    center = np.asarray(center, dtype=float).reshape(3)
    v = center + local_gcrf*float(radius_scale)

    mu_view = np.abs(normals @ view_hat)
    limb = np.clip(1.0-mu_view, 0.0, 1.0) ** 2.4
    mu_sun = np.clip(normals @ sun_hat, -1.0, 1.0)
    day = 0.18 + 0.82*np.sqrt(np.clip(mu_sun, 0.0, 1.0))
    twilight = np.exp(-(mu_sun/0.075)**2)
    alpha = float(np.clip(max_alpha, 0.0, 0.8))*limb*np.clip(day+0.28*twilight, 0.0, 1.0)
    blue = np.array([0.24, 0.56, 1.00])
    warm = np.array([1.00, 0.34, 0.10])
    warm_mix = np.clip(twilight*0.30, 0.0, 0.30)[:, None]
    rgb = blue*(1.0-warm_mix) + warm*warm_mix

    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        vertexcolor=_rgb_strings(rgb, alpha=alpha),
        opacity=0.999,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        hoverinfo="skip", showlegend=False, name="Atmosphere",
    )


def _ray_ellipsoid_first_hit(origin, direction, axes_km=(RE_KM, RE_KM, RP_EARTH_KM)):
    """Nearest non-negative ray hit on an axis-aligned ellipsoid."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = _normalize(direction, name="ray direction")
    axes = np.asarray(axes_km, dtype=float).reshape(3)
    q = origin / axes
    v = direction / axes
    aa = float(np.dot(v, v))
    bb = float(2.0 * np.dot(q, v))
    cc = float(np.dot(q, q) - 1.0)
    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0:
        return None
    root = np.sqrt(max(disc, 0.0))
    roots = np.array([(-bb-root)/(2.0*aa), (-bb+root)/(2.0*aa)])
    roots = roots[roots >= 0.0]
    return None if roots.size == 0 else origin + float(np.min(roots))*direction


def _surface_point_in_direction(direction, axes_km=(RE_KM, RE_KM, RP_EARTH_KM)):
    direction = _normalize(direction, name="surface direction")
    axes = np.asarray(axes_km, dtype=float).reshape(3)
    scale = 1.0 / np.sqrt(np.sum((direction/axes)**2))
    return direction * scale


def _shadow_patch_center_body(occluder_body, sun_body, patch_center_body=None):
    """Choose a body-fixed surface point near maximum solar obscuration."""
    if patch_center_body is not None:
        return _surface_point_in_direction(patch_center_body)
    axis = _normalize(occluder_body - sun_body, name="shadow axis")
    hit = _ray_ellipsoid_first_hit(occluder_body, axis)
    if hit is not None:
        return hit
    # For a partial eclipse the central axis can miss Earth.  The surface
    # point nearest the infinite axis is a stable centre for the penumbral
    # patch and avoids snapping to a coarse latitude/longitude vertex.
    closest = occluder_body - np.dot(occluder_body, axis)*axis
    if np.linalg.norm(closest) < 1e-9:
        closest = -occluder_body
    return _surface_point_in_direction(closest)


def _estimate_shadow_patch_half_angle_deg(occluder_body, sun_body, R_occ_km):
    """Angular patch radius large enough to include the solar penumbra."""
    sun_from_occ = sun_body - occluder_body
    D = float(np.linalg.norm(sun_from_occ))
    if D <= R_SUN_KM + R_occ_km:
        return 55.0
    axis = -sun_from_occ / D
    earth_from_occ = -occluder_body
    x = max(float(np.dot(earth_from_occ, axis)), 0.0)
    theta = float(np.arcsin(np.clip((R_SUN_KM + R_occ_km)/D, 0.0, 1.0)))
    vertex = R_occ_km / max(np.sin(theta), 1e-15)
    r_pen = (vertex + x) * np.tan(theta)
    angular = np.degrees(np.arcsin(np.clip(r_pen/RE_KM, 0.0, 0.985)))
    return float(np.clip(angular + 8.0, 30.0, 65.0))


def _earth_eclipse_shadow_trace(
    sun_hat,
    shadow_body_center_km,
    shadow_body_radius_km,
    *,
    sun_position_km=None,
    physical_center_km=(0.0, 0.0, 0.0),
    center=(0.0, 0.0, 0.0),
    radius_scale=1.0,
    time_jd=None,
    rotation_deg=0.0,
    patch_center_gcrf_km=None,
    resolution=241,
    half_angle_deg=None,
    nudge=1.0007,
    night_floor=0.003,
):
    """High-resolution local finite-disc Moon-shadow layer on WGS-84.

    A global 1--1.5 degree texture mesh can undersample a narrow totality
    corridor.  This independent local mesh resolves the umbra and penumbra
    without forcing every Earth animation frame to carry a sub-degree global
    texture.  The base Earth should be rendered without eclipse attenuation;
    this layer applies the missing direct-light ratio as a black RGBA overlay.
    """
    if illumination_fraction is None:
        raise RuntimeError("Finite-disc eclipse physics is unavailable")
    physical_center = np.asarray(physical_center_km, dtype=float).reshape(3)
    display_center = np.asarray(center, dtype=float).reshape(3)
    shadow_center = np.asarray(shadow_body_center_km, dtype=float).reshape(3)
    sun_hat = _normalize(sun_hat, name="sun_hat")
    if sun_position_km is None:
        sun_absolute = physical_center + sun_hat*AU_KM
    else:
        sun_absolute = np.asarray(sun_position_km, dtype=float).reshape(3)
        sun_hat = _normalize(sun_absolute-physical_center, name="Sun relative position")
    radius_scale = float(radius_scale)
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("radius_scale must be finite and positive")
    R_occ = float(shadow_body_radius_km)
    if not np.isfinite(R_occ) or R_occ <= 0.0:
        raise ValueError("shadow_body_radius_km must be finite and positive")

    resolution = max(25, int(resolution))
    if resolution % 2 == 0:
        resolution += 1
    rotation_rows = _itrf_to_gcrf_rotation(time_jd=time_jd, rotation_deg=rotation_deg)
    gcrf_to_body = rotation_rows.T
    occluder_body = (shadow_center-physical_center) @ gcrf_to_body
    sun_body = (sun_absolute-physical_center) @ gcrf_to_body
    patch_body = None
    if patch_center_gcrf_km is not None:
        patch_body = (np.asarray(patch_center_gcrf_km, dtype=float).reshape(3)
                      - physical_center) @ gcrf_to_body
    center_body = _shadow_patch_center_body(occluder_body, sun_body, patch_body)
    center_dir = _normalize(center_body, name="shadow patch centre")

    if half_angle_deg is None:
        half_angle_deg = _estimate_shadow_patch_half_angle_deg(
            occluder_body, sun_body, R_occ)
    half_angle = np.radians(float(np.clip(half_angle_deg, 5.0, 75.0)))
    tangent_extent = np.tan(half_angle)
    q = np.linspace(-tangent_extent, tangent_extent, resolution)
    QX, QY = np.meshgrid(q, q)
    angular = np.arctan(np.hypot(QX, QY))

    ref = np.array([0.0, 0.0, 1.0]) if abs(center_dir[2]) < 0.90 else np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(ref, center_dir), name="shadow patch tangent")
    v = np.cross(center_dir, u)
    directions = (center_dir[None, None, :]
                  + QX[..., None]*u[None, None, :]
                  + QY[..., None]*v[None, None, :])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    axes = np.array([RE_KM, RE_KM, RP_EARTH_KM])
    radial_scale = 1.0/np.sqrt(np.sum((directions/axes)**2, axis=-1))
    body_points = directions*radial_scale[..., None]
    normal_body = body_points/(axes*axes)
    normal_body /= np.linalg.norm(normal_body, axis=-1, keepdims=True)
    local_gcrf = body_points.reshape(-1, 3) @ rotation_rows
    normals_gcrf = normal_body.reshape(-1, 3) @ rotation_rows
    physical_points = physical_center + local_gcrf

    eval_from_occluder = physical_points-shadow_center
    sun_from_occluder = sun_absolute-shadow_center
    visibility = np.asarray(irradiance_fraction(
        eval_from_occluder,
        R_body_km=R_occ,
        R_sun_km=R_SUN_KM,
        sun_position_km=np.broadcast_to(sun_from_occluder, eval_from_occluder.shape),
        photometry="quadratic-visible", quadrature_order=48,
    ), dtype=float).reshape(-1)
    visibility = np.clip(visibility, 0.0, 1.0)

    mu = np.clip(normals_gcrf @ sun_hat, -1.0, 1.0)
    direct = np.clip(mu, 0.0, 1.0)
    sky = _smoothstep(np.radians(-6.0), np.radians(2.0), np.arcsin(mu))
    night_floor = float(np.clip(night_floor, 0.0, 0.08))
    base_illum = night_floor + 0.035*sky + 0.965*direct
    shadow_illum = (night_floor
                    + 0.035*sky*(0.15+0.85*np.sqrt(visibility))
                    + 0.965*direct*visibility)
    ratio = np.clip(shadow_illum/np.maximum(base_illum, 1e-12), 0.0, 1.0)
    # Plotly alpha blends encoded RGB.  Gamma-encoding the linear-light ratio
    # gives the black overlay the same perceived attenuation as the surface
    # shader to first order, instead of making a 50% penumbra look half-black.
    alpha = 1.0-_linear_to_srgb(ratio)
    edge = 1.0-_smoothstep(0.90*half_angle, half_angle, angular.reshape(-1))
    alpha *= edge
    alpha[visibility > 1.0-1e-7] = 0.0
    alpha[(direct <= 0.0) & (sky <= 1e-7)] = 0.0
    alpha = np.clip(alpha, 0.0, 0.995)

    display_vertices = display_center + local_gcrf*radius_scale*float(nudge)
    ids = np.arange(resolution*resolution, dtype=np.int32).reshape(resolution, resolution)
    a = ids[:-1, :-1].ravel(); b = ids[1:, :-1].ravel()
    c = ids[1:, 1:].ravel(); d = ids[:-1, 1:].ravel()
    faces_i = np.concatenate([a, a])
    faces_j = np.concatenate([c, d])
    faces_k = np.concatenate([b, c])
    black = np.zeros((len(display_vertices), 3), dtype=float)
    return go.Mesh3d(
        x=display_vertices[:, 0], y=display_vertices[:, 1], z=display_vertices[:, 2],
        i=faces_i, j=faces_j, k=faces_k,
        vertexcolor=_rgb_strings(black, alpha=alpha), opacity=0.999,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        hoverinfo="skip", showlegend=False, name="Finite-disc eclipse shadow",
        meta=dict(
            minimum_visibility=float(np.min(visibility)),
            half_angle_deg=float(np.degrees(half_angle)),
            resolution=int(resolution),
        ),
    )


def _sun_sphere_traces(pos, radius_km, seed=11, *, n=64, view_hat=None,
                       glow=True):
    """
    Real granulation-style texture instead of an arbitrary pole-to-equator
    brightness band (the old `surfacecolor=np.sin(SV)` made the sphere look
    like it had permanently dark poles and a bright equator ring, tied to
    an arbitrary axis rather than looking like an actual glowing surface).

    Two things layered together, matching how the real Sun actually looks:
      1. Procedural granulation — mottled brightness variation from many
         overlapping blotches at different scales, like solar granulation
         cells, not a smooth gradient.
      2. Limb darkening — brightness falls off toward the visible edge as
         seen from directly "above" each point (this is a real optical
         effect: you're seeing cooler, less dense gas at a grazing angle
         near the apparent edge of any self-luminous sphere). Approximated
         here using the angle between each point's outward normal and the
         sphere's own polar axis is NOT how real limb darkening works —
         real limb darkening depends on the *viewer's* direction, which a
         static per-vertex colour array can't follow as you rotate the
         plot. This uses a fixed reference direction as a stand-in, which
         is imperfect but at least reads as "glowing sphere" from a normal
         viewing angle instead of a banded one.
    """
    n = max(20, int(n))
    su = np.linspace(0, 2*np.pi, n)
    sv = np.linspace(0, np.pi, n // 2)
    SU, SV = np.meshgrid(su, sv)
    nx, ny, nz = np.cos(SU)*np.sin(SV), np.sin(SU)*np.sin(SV), np.cos(SV)
    sx = pos[0] + radius_km*nx
    sy = pos[1] + radius_km*ny
    sz = pos[2] + radius_km*nz

    rng = np.random.default_rng(seed)
    granulation = np.zeros_like(SU)
    for _ in range(40):
        # Random blotch centres directly in (nx,ny,nz) space (angular
        # distance via dot product) so blotches wrap seamlessly around
        # the sphere with no seam at the poles or +-180 longitude.
        c = rng.normal(size=3); c /= np.linalg.norm(c)
        dot = nx*c[0] + ny*c[1] + nz*c[2]
        spread = rng.uniform(0.85, 0.97)   # dot-product threshold, not degrees
        # Smoothstep instead of a linear clip ramp — a linear ramp between
        # threshold and 1.0 still has a visible kink where it hits zero;
        # smoothstep's zero derivative at both ends blends each blotch
        # into its neighbors with no seam, same fix as the Moon/Earth
        # procedural fields.
        t = np.clip((dot - spread) / (1 - spread), 0, 1)
        smoothstep = t * t * (3 - 2 * t)
        granulation += smoothstep * rng.uniform(-0.25, 0.25)

    # Reference-direction brightness falloff (imperfect stand-in for real
    # viewer-relative limb darkening — see docstring)
    ref = (np.asarray(view_hat, dtype=float) if view_hat is not None
           else np.array([0.4, 0.4, 0.82]))
    ref /= np.linalg.norm(ref)
    limb = np.clip(nx*ref[0] + ny*ref[1] + nz*ref[2], 0, 1) ** 0.35
    brightness = np.clip(0.55 + 0.45*limb + granulation, 0.15, 1.0)

    traces = [go.Surface(
        x=sx, y=sy, z=sz,
        colorscale=[[0, "#7A2E00"], [0.35, "#D9550A"],
                   [0.65, "#FFA500"], [1, "#FFFBEA"]],
        surfacecolor=brightness,
        cmin=0.15, cmax=1.0,
        showscale=False,
        lighting=dict(ambient=1.0, diffuse=0.0),
        name="Sun", hovertemplate="Sun<extra></extra>",
    )]
    if not glow:
        return traces
    for glow_scale, glow_op in [(1.32, 0.13), (1.72, 0.055)]:
        gx = pos[0] + radius_km*glow_scale*nx
        gy = pos[1] + radius_km*glow_scale*ny
        gz = pos[2] + radius_km*glow_scale*nz
        traces.append(go.Surface(
            x=gx, y=gy, z=gz,
            colorscale=[[0, "#FFD700"], [1, "#FFD700"]],
            showscale=False, opacity=glow_op,
            lighting=dict(ambient=1.0, diffuse=0.0),
            hoverinfo="skip", showlegend=False, name="Sun glow",
        ))
    return traces


def _write_plotly_output(fig, save_path, *, width=1400, height=1000, scale=1.0):
    """Write a Plotly figure using portable ``PathLike`` handling.

    HTML output is dependency-free. Raster/vector output is delegated to
    Plotly/Kaleido, but failures are converted to an actionable error rather
    than leaking a backend-specific exception after an otherwise successful
    render.
    """
    path = Path(save_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".html", ".htm"}:
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    else:
        try:
            fig.write_image(str(path), width=int(width), height=int(height), scale=float(scale))
        except Exception as ex:
            raise RuntimeError(
                "Static Plotly export requires a working Kaleido installation; "
                "use an .html destination for dependency-free interactive output."
            ) from ex
    return path


def plot_globe_orbit_daynight_plotly(a_km=None, e=None, inc_deg=None, raan_deg=0.0,
                                     argp_deg=0.0, nu0_deg=0.0,
                                     sat_name="Satellite",
                                     n_orbits=1.0, n_steps=1500, save_path=None,
                                     epoch_jd=2_460_500.0,
                                     show_sun_body=True, *, orbit=None, r=None,
                                     v=None, t=None, r_units="auto", v_units="auto",
                                     earth_time=None, reference_index=-1,
                                     highlight_reference=True,
                                     **kwargs):
    """Plot an Earth day/night globe with an orbit trajectory.

    The trajectory may be supplied as Keplerian elements, an SSAPy ``Orbit``
    object, or time-series arrays from ``ssapy.rv``/``Orbit.at``.  Array inputs
    may be in metres or kilometres; units are auto-detected unless explicitly
    set with ``r_units``/``v_units``.

    Earth shading comes from ``_earth_mesh``, which works in linear light and
    no longer synthesises city lights or procedural continents.
    """
    save_path, kwargs = _pop_save_path_aliases(kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(kwargs, "plot_globe_orbit_daynight_plotly")

    default_epoch_jd = 2_460_500.0
    if orbit is not None or r is not None:
        r_eci, _, t_time = normalize_orbit_trajectory(
            orbit=orbit, r=r, v=v, t=t,
            require_velocity=False,
            r_units=r_units, v_units=v_units,
            n_steps=n_steps, n_orbits=n_orbits,
        )
        ref_idx = int(reference_index) % len(r_eci)
        reference_time = earth_time if earth_time is not None else (
            t_time[ref_idx] if len(t_time) else None)
        epoch_jd = float(reference_time.jd) if hasattr(reference_time, "jd") else epoch_jd
        sun_hat = sun_direction_eci(np.array([0.0]), epoch_jd=epoch_jd)[0]
        rotation_deg = (earth_rotation_deg_from_time(reference_time)
                        if reference_time is not None else 0.0)
    else:
        if a_km is None or e is None or inc_deg is None:
            raise ValueError(
                "Provide Keplerian a_km/e/inc_deg, orbit=, or r= trajectory input.")
        t_s, r_eci, _ = propagate_eci(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg,
                                      n_orbits=n_orbits, n_steps=n_steps)
        ref_idx = int(reference_index) % len(r_eci)
        if earth_time is not None:
            reference_time = earth_time
            epoch_jd = float(reference_time.jd) if hasattr(reference_time, "jd") else epoch_jd
            sun_hat = sun_direction_eci(np.array([0.0]), epoch_jd=epoch_jd)[0]
            rotation_deg = earth_rotation_deg_from_time(reference_time)
        else:
            sun_hat = sun_direction_eci(t_s, epoch_jd=epoch_jd)[0]
            rotation_deg = earth_rotation_deg_from_time(
                epoch_jd=default_epoch_jd, relative_seconds=float(t_s[ref_idx]))

    orbit_r = np.max(np.linalg.norm(r_eci, axis=1))
    frame_r = max(orbit_r, RE_KM*1.3)

    fig = go.Figure()
    camera_eye = np.array([1.35, 1.35, 1.10])
    fig.add_trace(_earth_mesh(sun_hat, rotation_deg=rotation_deg, time_jd=epoch_jd))
    fig.add_trace(_earth_atmosphere_trace(
        sun_hat=sun_hat, view_hat=camera_eye, time_jd=epoch_jd,
    ))

    # Orbit path, colour-mapped along its length
    n_pts = len(r_eci)
    colors = np.linspace(0, 1, n_pts)
    fig.add_trace(go.Scatter3d(
        x=r_eci[:, 0], y=r_eci[:, 1], z=r_eci[:, 2],
        mode="lines",
        line=dict(color=colors, colorscale="Plasma", width=6),
        name=sat_name, hoverinfo="skip",
    ))

    if highlight_reference and n_pts > 1:
        fig.add_trace(go.Scatter3d(
            x=[r_eci[ref_idx, 0]], y=[r_eci[ref_idx, 1]], z=[r_eci[ref_idx, 2]],
            mode="markers",
            marker=dict(size=7, color="#FFD700", symbol="diamond",
                        line=dict(color="white", width=1)),
            name="Earth orientation point",
            hovertemplate=(f"Earth orientation sample {ref_idx + 1}/{n_pts}"
                           "<br>X=%{x:.0f} km<br>Y=%{y:.0f} km"
                           "<br>Z=%{z:.0f} km<extra></extra>"),
            showlegend=False,
        ))

    sun_pos = None
    sun_radius = 0.0
    if show_sun_body:
        sun_pos, sun_radius = sun_position_and_radius(
            scene_radius_km=frame_r, sun_hat=sun_hat,
            distance_mode="angular", distance_factor=1.55,
            radius_mode="angular",
        )
        for tr in _sun_sphere_traces(sun_pos, sun_radius):
            fig.add_trace(tr)

    lim = frame_r * 1.9
    if sun_pos is not None:
        lim = max(lim, float(np.linalg.norm(sun_pos)) + sun_radius * 2.5)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="X [km]", backgroundcolor="black",
                      gridcolor="#333", color="white"),
            yaxis=dict(range=[-lim, lim], title="Y [km]", backgroundcolor="black",
                      gridcolor="#333", color="white"),
            zaxis=dict(range=[-lim, lim], title="Z [km]", backgroundcolor="black",
                      gridcolor="#333", color="white"),
            bgcolor="black",
            aspectmode="cube",
            camera=dict(eye=dict(x=float(camera_eye[0]), y=float(camera_eye[1]), z=float(camera_eye[2]))),
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        title=dict(text=f"{sat_name} — orbit around Earth, with the Sun shown in frame",
                  x=0.5, font=dict(color="white", size=16)),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    if save_path:
        path = _write_plotly_output(fig, save_path, width=1400, height=1000, scale=1.0)
        print(f"Saved -> {path}")
    return fig


if __name__ == "__main__":
    # Write the dependency-free interactive artifact first. Static export is
    # attempted from the same in-memory figure and skipped cleanly when
    # Kaleido is unavailable, so a standalone run never fails after doing all
    # of the rendering work.
    # Never write into the package directory: that puts generated artifacts
    # inside the repository and inside any installed copy.
    from .figpath import figpath
    out_dir = Path(figpath("demo_gallery/figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_globe_orbit_daynight_plotly(
        a_km=26_560.0, e=0.001, inc_deg=55.0,
        sat_name="GPS-like MEO", n_orbits=1.0, n_steps=1500,
        save_path=out_dir / "globe_orbit_daynight_plotly.html",
    )
    try:
        png_path = _write_plotly_output(
            figure, out_dir / "globe_orbit_daynight_plotly.png",
            width=1400, height=1000, scale=1.0,
        )
        print(f"Saved -> {png_path}")
    except RuntimeError as ex:
        warnings.warn(f"PNG export skipped: {ex}", RuntimeWarning)