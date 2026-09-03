"""High-fidelity Moon renderer shared by the eclipse scenes.

The renderer uses the current SSAPy-Data lunar photomosaic when available,
keeps the Moon synchronously oriented toward Earth, shades in linear light,
and applies the same finite-distance apparent-disc overlap model used by the
eclipse calculations.  The mesh has one vertex at each pole and a periodic
longitude seam, so it contains no degenerate pole triangles or open seam.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import warnings

import numpy as np
import plotly.graph_objects as go

try:
    from ssapy_toolkit.io.eclipse_asset_resolver import asset_candidates
except ImportError:
    from ssapy_toolkit.io.eclipse_asset_resolver import asset_candidates

try:
    from ssapy_toolkit.compute.eclipse_brightness import illumination_fraction, irradiance_fraction, R_SUN_KM, AU_KM
except ImportError:
    try:
        from ssapy_toolkit.compute.eclipse_brightness import illumination_fraction, irradiance_fraction, R_SUN_KM, AU_KM
    except ImportError:
        illumination_fraction = None
        irradiance_fraction = None
        R_SUN_KM, AU_KM = 695_700.0, 149_597_870.7

R_MOON_KM = 1_737.4
RE_KM = 6_378.137

_moon_texture_cache: dict[tuple[str, int, int, int, int], np.ndarray] = {}
_moon_texture_source_cache: dict[tuple[str, int, int, int, int], str] = {}
_moon_base_printed: set[tuple[str, int, int]] = set()
_moon_albedo_cache: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
_moon_relief_cache: dict[tuple[tuple[int, ...], int], np.ndarray] = {}


def _normalize(v, *, name="vector"):
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must be a finite, non-zero vector")
    return arr / norm


def _srgb_to_linear(rgb):
    x = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    x = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92*x, 1.055*x**(1.0/2.4)-0.055)


def _moon_texture_candidates(texture_path=None):
    """Yield lunar images through the shared SSAPy-Data resolver."""
    for path, _source, _root in asset_candidates(
        "moon_albedo", explicit=texture_path, policy="data-first"
    ):
        yield path


def _periodic_resize_rgb(image, n_lat, n_lon):
    """Resize an equirectangular texture with periodic longitude support."""
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    if image.size == (n_lon, n_lat):
        return image
    width, height = image.size
    tiled = Image.new("RGB", (3*width, height))
    for k in range(3):
        tiled.paste(image, (k*width, 0))
    tiled = tiled.resize((3*n_lon, n_lat), resampling)
    return tiled.crop((n_lon, 0, 2*n_lon, n_lat))


def _load_real_moon_texture(n_lat, n_lon, texture_path=None, *, return_source=False):
    """Load a north-up, -180..+180 SSAPy lunar equirectangular texture.

    No half-width roll is applied.  SSAPy's Moon texture already follows the
    same longitude convention as the renderer; the former roll moved the
    familiar near-side maria to the wrong hemisphere.
    """
    from PIL import Image

    n_lat, n_lon = int(n_lat), int(n_lon)
    if n_lat < 4 or n_lon < 8:
        raise ValueError("Moon texture resolution must be at least 4 x 8")

    if texture_path is not None:
        explicit = Path(texture_path).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"Moon texture does not exist: {explicit}")
        candidates = (explicit,)
    else:
        candidates = tuple(_moon_texture_candidates())

    last_error = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            stat = candidate.stat()
            key = (str(candidate), int(stat.st_mtime_ns), int(stat.st_size), n_lat, n_lon)
            if key not in _moon_texture_cache:
                with Image.open(candidate) as image:
                    image = image.convert("RGB")
                    image = _periodic_resize_rgb(image, n_lat, n_lon)
                    array = np.asarray(image, dtype=np.uint8)
                    array.setflags(write=False)
                    _moon_texture_cache[key] = array
                    _moon_texture_source_cache[key] = str(candidate)
            result = _moon_texture_cache[key]
            source = _moon_texture_source_cache[key]
            return (result, source) if return_source else result
        except Exception as ex:
            last_error = ex

    if last_error is not None:
        warnings.warn(f"Moon texture could not be loaded: {last_error}", RuntimeWarning)
    return (None, "procedural fallback") if return_source else None


def _smooth(field, sigma=1.4):
    try:
        from scipy.ndimage import gaussian_filter
        # Latitude does not wrap, longitude does.
        return gaussian_filter(field, sigma=sigma, mode=("nearest", "wrap"))
    except Exception:
        return field


def _procedural_moon_albedo(Lat, Lon, seed=3):
    """Deterministic maria/highland fallback used only without a real map."""
    key = (Lat.shape, int(seed))
    if key in _moon_albedo_cache:
        return _moon_albedo_cache[key]
    rng = np.random.default_rng(seed)
    albedo = np.full_like(Lat, 0.72, dtype=float)
    for _ in range(10):
        clat = rng.uniform(-45, 45)
        clon = rng.uniform(-70, 70)
        slat, slon = rng.uniform(9, 24), rng.uniform(13, 34)
        dlat = (Lat-clat)/slat
        dlon = ((Lon-clon+180) % 360-180)/slon
        albedo -= rng.uniform(0.10, 0.22)*np.exp(-0.5*(dlat*dlat+dlon*dlon))
    for count, rmin, rmax, depth in ((70, 3.5, 10.0, 0.10), (450, 0.5, 2.4, 0.065)):
        for _ in range(count):
            clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
            rad = rng.uniform(rmin, rmax)
            d = np.sqrt((Lat-clat)**2 + (((Lon-clon+180) % 360)-180)**2)
            albedo -= depth*np.clip(1-d/rad, 0, 1)**3
    albedo += rng.normal(0, 0.008, Lat.shape)
    result = np.clip(_smooth(albedo, 0.45), 0.27, 0.95)
    result.setflags(write=False)
    _moon_albedo_cache[key] = result
    return result


def _procedural_moon_relief(Lat, Lon, seed=3):
    """Deterministic crater relief for the procedural fallback only."""
    key = (Lat.shape, int(seed))
    if key in _moon_relief_cache:
        return _moon_relief_cache[key]
    rng = np.random.default_rng(seed)
    relief = np.zeros_like(Lat, dtype=float)
    for count, rmin, rmax, bowl_scale, rim_scale in (
        (65, 3.5, 10.0, 0.55, 0.38),
        (450, 0.55, 2.5, 0.35, 0.25),
    ):
        for _ in range(count):
            clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
            rad = rng.uniform(rmin, rmax)
            d = np.sqrt((Lat-clat)**2 + (((Lon-clon+180) % 360)-180)**2)
            bowl = np.clip(1-d/rad, 0, 1)
            rim = np.clip(1-np.abs(d-0.9*rad)/(0.18*rad), 0, 1)
            relief -= bowl_scale*bowl**3
            relief += rim_scale*rim**2
    result = _smooth(relief, 0.42)
    result.setflags(write=False)
    _moon_relief_cache[key] = result
    return result


@lru_cache(maxsize=32)
def _moon_unit_mesh_cached(n_lat, n_lon):
    n_lat, n_lon = int(n_lat), int(n_lon)
    if n_lat < 5 or n_lon < 8:
        raise ValueError("Moon mesh resolution must be at least 5 x 8")
    if n_lat % 2 == 0:
        n_lat += 1

    lat_rows = np.linspace(90.0, -90.0, n_lat)
    ring_lat = lat_rows[1:-1]
    lon = np.linspace(-180.0, 180.0, n_lon, endpoint=False)
    Lon, Lat = np.meshgrid(lon, ring_lat)
    phi, lam = np.radians(Lat), np.radians(Lon)
    ring_dirs = np.stack([
        np.cos(phi)*np.cos(lam),
        np.cos(phi)*np.sin(lam),
        np.sin(phi),
    ], axis=-1).reshape(-1, 3)
    dirs = np.vstack([ring_dirs, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    lat_v = np.concatenate([Lat.ravel(), [90.0, -90.0]])
    lon_v = np.concatenate([Lon.ravel(), [0.0, 0.0]])

    ring_count = n_lat-2
    cols = np.arange(n_lon, dtype=np.int32)
    next_cols = (cols+1) % n_lon
    blocks = []
    for row in range(ring_count-1):
        top, bottom = row*n_lon, (row+1)*n_lon
        blocks.append(np.column_stack([top+cols, bottom+cols, bottom+next_cols]))
        blocks.append(np.column_stack([top+cols, bottom+next_cols, top+next_cols]))
    north_idx, south_idx = ring_count*n_lon, ring_count*n_lon+1
    blocks.append(np.column_stack([np.full(n_lon, north_idx), next_cols, cols]))
    last = (ring_count-1)*n_lon
    blocks.append(np.column_stack([np.full(n_lon, south_idx), last+cols, last+next_cols]))
    faces = np.vstack(blocks).astype(np.int32)

    p0, p1, p2 = dirs[faces[:, 0]], dirs[faces[:, 1]], dirs[faces[:, 2]]
    orient = np.einsum("ij,ij->i", np.cross(p1-p0, p2-p0), (p0+p1+p2)/3.0)
    flip = orient < 0
    faces[flip, 1], faces[flip, 2] = faces[flip, 2].copy(), faces[flip, 1].copy()

    for array in (dirs, faces, lat_rows, lat_v, lon_v):
        array.setflags(write=False)
    return dirs, faces, lat_rows, lat_v, lon_v


def _moon_unit_mesh(n_lat, n_lon):
    return _moon_unit_mesh_cached(int(n_lat), int(n_lon))


def _mesh_vertex_normals(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals = np.zeros_like(vertices)
    for col in range(3):
        np.add.at(normals, faces[:, col], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norm, 1e-15)
    flip = np.einsum("ij,ij->i", normals, vertices) < 0
    normals[flip] *= -1
    return normals


def _vertex_normals(X, Y, Z, center=(0.0, 0.0, 0.0)):
    """Backward-compatible grid-normal helper used by older callers/tests."""
    Xu = np.gradient(X, axis=1); Yu = np.gradient(Y, axis=1); Zu = np.gradient(Z, axis=1)
    Xv = np.gradient(X, axis=0); Yv = np.gradient(Y, axis=0); Zv = np.gradient(Z, axis=0)
    nx = Yu*Zv-Zu*Yv
    ny = Zu*Xv-Xu*Zv
    nz = Xu*Yv-Yu*Xv
    norm = np.sqrt(nx*nx+ny*ny+nz*nz)+1e-15
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    center = np.asarray(center, dtype=float)
    Xr, Yr, Zr = X-center[0], Y-center[1], Z-center[2]
    flip = nx*Xr+ny*Yr+nz*Zr < 0
    return np.where(flip, -nx, nx), np.where(flip, -ny, ny), np.where(flip, -nz, nz)


def _synchronous_body_axes(real_center_km):
    """Approximate tidally locked lunar axes; body +X faces Earth."""
    if real_center_km is None:
        return np.eye(3)
    r = np.asarray(real_center_km, dtype=float)
    x_axis = -_normalize(r, name="Moon geocentric position")
    north = np.array([0.0, 0.0, 1.0])
    z_axis = north-x_axis*np.dot(north, x_axis)
    if np.linalg.norm(z_axis) < 1e-10:
        north = np.array([0.0, 1.0, 0.0])
        z_axis = north-x_axis*np.dot(north, x_axis)
    z_axis = _normalize(z_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


@dataclass
class MoonSurfaceData:
    display_vertices: np.ndarray
    physical_vertices_gcrf_km: np.ndarray
    normals_gcrf: np.ndarray
    faces: np.ndarray
    base_rgb_srgb: np.ndarray
    shaded_rgb_srgb: np.ndarray
    eclipse_visibility: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    texture_source: str
    body_axes: np.ndarray
    n_lat_effective: int
    n_lon_effective: int
    used_relief: bool


def _flatten_texture_rows(texture):
    ring = texture[1:-1].reshape(-1, 3).astype(float)/255.0
    north = texture[0].mean(axis=0, keepdims=True).astype(float)/255.0
    south = texture[-1].mean(axis=0, keepdims=True).astype(float)/255.0
    return np.vstack([ring, north, south])


def _flatten_scalar_rows(field):
    ring = field[1:-1].reshape(-1)
    return np.concatenate([ring, [float(np.mean(field[0])), float(np.mean(field[-1]))]])


def _moon_surface_data(
    center,
    radius,
    sun_hat=None,
    seed=3,
    real_center_km=None,
    mode="lunar",
    eclipse_tint=True,
    n_lat=181,
    n_lon=360,
    real_sun_position_km=None,
    relief_exaggeration=0.0,
    ambient_floor=0.008,
    texture_path=None,
    view_hat=None,
    exposure=1.08,
    eclipse_occluder_radius_km=RE_KM,
):
    """Compute Moon geometry and linear-light surface colors."""
    center = np.asarray(center, dtype=float).reshape(3)
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if mode not in ("lunar", "solar", "normal"):
        raise ValueError("mode must be 'lunar', 'solar', or 'normal'")

    dirs_body, faces, lat_rows, lat_v, lon_v = _moon_unit_mesh(n_lat, n_lon)
    n_lat_effective = len(lat_rows)
    n_lon_effective = int(n_lon)
    body_axes = _synchronous_body_axes(real_center_km)

    texture, source = _load_real_moon_texture(
        n_lat_effective, n_lon_effective, texture_path=texture_path, return_source=True,
    )
    use_real_texture = texture is not None
    Lon, Lat = np.meshgrid(
        np.linspace(-180.0, 180.0, n_lon_effective, endpoint=False),
        lat_rows,
    )
    if use_real_texture:
        base_rgb = _flatten_texture_rows(texture)
        # Do not place random crater geometry beneath a real photomosaic: the
        # relief shadows would not line up with the mapped craters.
        relief_v = np.zeros(len(dirs_body), dtype=float)
        used_relief = False
    else:
        albedo = _procedural_moon_albedo(Lat, Lon, seed=seed)
        procedural_rgb = np.repeat(albedo[..., None], 3, axis=-1)*np.array([1.00, 0.98, 0.94])
        base_rgb = _flatten_texture_rows(np.rint(np.clip(procedural_rgb, 0, 1)*255).astype(np.uint8))
        relief = _procedural_moon_relief(Lat, Lon, seed=seed)
        relief_v = _flatten_scalar_rows(relief)
        used_relief = bool(abs(float(relief_exaggeration)) > 0)

    print_key = (source, n_lat_effective, n_lon_effective)
    if print_key not in _moon_base_printed:
        print(f"[moon_render] Moon base: {source}")
        _moon_base_printed.add(print_key)

    radial_factor = 1.0+relief_v*float(relief_exaggeration)
    local_body = dirs_body*radial_factor[:, None]
    local_normals_body = _mesh_vertex_normals(local_body, faces) if used_relief else dirs_body
    world_dirs = dirs_body@body_axes.T
    world_local = local_body@body_axes.T
    world_normals = local_normals_body@body_axes.T
    world_normals /= np.linalg.norm(world_normals, axis=1, keepdims=True)
    display_vertices = center+radius*world_local

    if real_center_km is None:
        physical_center = np.zeros(3)
    else:
        physical_center = np.asarray(real_center_km, dtype=float).reshape(3)
    physical_vertices = physical_center+R_MOON_KM*world_dirs

    if real_sun_position_km is not None:
        sun_position = np.asarray(real_sun_position_km, dtype=float).reshape(3)
        light_hat = _normalize(sun_position-physical_center, name="Sun relative to Moon")
    elif sun_hat is not None:
        light_hat = _normalize(sun_hat, name="sun_hat")
        sun_position = physical_center+light_hat*AU_KM
    else:
        light_hat = None
        sun_position = None

    if view_hat is None:
        if real_center_km is not None:
            view_hat = -_normalize(real_center_km, name="Moon geocentric position")
        elif light_hat is not None:
            view_hat = light_hat
        else:
            view_hat = np.array([1.0, 0.0, 0.0])
    view_hat = _normalize(view_hat, name="view_hat")

    ambient_floor = float(np.clip(ambient_floor, 0.0, 0.08))
    direct_reflectance = np.zeros(len(world_normals), dtype=float)
    if light_hat is not None:
        mu0 = np.clip(world_normals@light_hat, 0.0, 1.0)
        mu = np.clip(world_normals@view_hat, 0.0, 1.0)
        # Lunar regolith is better represented by a Lommel-Seeliger response
        # than by a pure Lambert sphere.  A small Lambert term keeps the limb
        # stable as the interactive camera moves.
        ls = np.where(mu0 > 0, 2.0*mu0/np.maximum(mu0+mu, 1e-6), 0.0)
        direct_reflectance = np.clip(0.78*ls+0.22*mu0, 0.0, 1.18)
        phase = np.arccos(np.clip(np.dot(light_hat, view_hat), -1.0, 1.0))
        opposition = 1.0+0.14*np.exp(-(phase/np.radians(5.0))**2)
        direct_reflectance *= opposition

    eclipse_visibility = np.ones(len(world_normals), dtype=float)
    geom = None
    if (mode == "lunar" and real_center_km is not None and sun_position is not None
            and irradiance_fraction is not None):
        eclipse_visibility, geom = irradiance_fraction(
            physical_vertices,
            R_body_km=float(eclipse_occluder_radius_km),
            R_sun_km=R_SUN_KM,
            sun_position_km=np.broadcast_to(sun_position, physical_vertices.shape),
            return_geometry=True,
            photometry="quadratic-visible",
            quadrature_order=48,
        )
        eclipse_visibility = np.clip(np.asarray(eclipse_visibility, dtype=float), 0.0, 1.0)
        direct_reflectance *= eclipse_visibility

    base_linear = _srgb_to_linear(base_rgb)
    # A modest gain compensates for the albedo map being displayed under a
    # physical BRDF instead of as an already-lit photograph.
    base_linear *= float(np.clip(exposure, 0.25, 3.0))
    rgb_linear = base_linear*(ambient_floor+0.992*direct_reflectance[:, None])

    # Earthshine is strongest near solar eclipse/new Moon geometry, when the
    # Earth-facing lunar hemisphere sees an almost full Earth.  It reveals the
    # real texture without pretending the dark side is directly sunlit.
    if real_center_km is not None and light_hat is not None:
        earthward = -_normalize(real_center_km, name="Moon geocentric position")
        moon_from_earth = -earthward
        earth_fullness = 0.5*(1.0+np.clip(np.dot(light_hat, moon_from_earth), -1.0, 1.0))
        mu_e = np.clip(world_normals@earthward, 0.0, 1.0)**0.65
        earthshine = 0.036*earth_fullness*mu_e
        rgb_linear += base_linear*earthshine[:, None]*np.array([0.72, 0.84, 1.00])

    if geom is not None and eclipse_tint:
        a_occ = np.asarray(geom.occluder_angular_radius_rad)
        a_sun = np.asarray(geom.sun_angular_radius_rad)
        sep = np.asarray(geom.separation_rad)
        umbra_depth = np.clip((a_occ-a_sun-sep)/np.maximum(2.0*a_sun, 1e-12), 0.0, 1.0)
        blocked = (1.0-eclipse_visibility)**1.55
        # Refracted sunlight is brighter near the umbral edge and red-dominant
        # throughout.  This is an intentionally simple atmosphere model, but
        # it preserves the exact geometric boundary and the mapped albedo.
        refracted = blocked*(0.045+0.14*np.exp(-2.5*umbra_depth))
        rgb_linear += base_linear*refracted[:, None]*np.array([1.00, 0.16, 0.035])

    shaded = _linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))
    return MoonSurfaceData(
        display_vertices=display_vertices,
        physical_vertices_gcrf_km=physical_vertices,
        normals_gcrf=world_normals,
        faces=faces,
        base_rgb_srgb=base_rgb,
        shaded_rgb_srgb=shaded,
        eclipse_visibility=eclipse_visibility,
        latitude_deg=lat_v,
        longitude_deg=lon_v,
        texture_source=source,
        body_axes=body_axes,
        n_lat_effective=n_lat_effective,
        n_lon_effective=n_lon_effective,
        used_relief=used_relief,
    )


def _rgb_strings(rgb):
    rgb8 = np.rint(np.clip(rgb, 0.0, 1.0)*255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb8]


def moon_mesh_plotly(
    center,
    radius,
    sun_hat=None,
    seed=3,
    real_center_km=None,
    mode="lunar",
    eclipse_tint=True,
    n_lat=181,
    n_lon=360,
    real_sun_position_km=None,
    relief_exaggeration=0.0,
    ambient_floor=0.008,
    texture_path=None,
    view_hat=None,
    exposure=1.08,
    eclipse_occluder_radius_km=RE_KM,
):
    """Return a high-quality, texture-mapped Plotly Moon mesh."""
    data = _moon_surface_data(
        center=center,
        radius=radius,
        sun_hat=sun_hat,
        seed=seed,
        real_center_km=real_center_km,
        mode=mode,
        eclipse_tint=eclipse_tint,
        n_lat=n_lat,
        n_lon=n_lon,
        real_sun_position_km=real_sun_position_km,
        relief_exaggeration=relief_exaggeration,
        ambient_floor=ambient_floor,
        texture_path=texture_path,
        view_hat=view_hat,
        exposure=exposure,
        eclipse_occluder_radius_km=eclipse_occluder_radius_km,
    )
    v, f = data.display_vertices, data.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        vertexcolor=_rgb_strings(data.shaded_rgb_srgb),
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        name="Moon", hoverinfo="skip", showlegend=False,
    )
