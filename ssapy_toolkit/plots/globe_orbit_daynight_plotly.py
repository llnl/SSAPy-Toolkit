"""
globe_orbit_daynight_plotly.py — Plotly version, standalone
================================================================
Same content as globe_orbit_daynight_plot.py (Earth with day/night
terminator, an orbit around it, and the Sun rendered as a real body in
frame), but in Plotly instead of matplotlib, and with no dependency on the
toolkit's core/ssapy modules — runs anywhere with numpy + plotly.

Earth and Sun are both genuine 3D mesh/sphere geometry (go.Mesh3d /
go.Surface), matching the real 3D sphere approach used in the toolkit's
own layers.py (EarthLayer / SunLayer), not flat markers.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

# Needed for the optional Moon-shadow-on-Earth shading (solar eclipse case)
try:
    from .eclipse_brightness_plot import illumination_fraction, R_SUN_KM, AU_KM
except ImportError:
    try:
        from eclipse_brightness_plot import illumination_fraction, R_SUN_KM, AU_KM
    except ImportError:
        illumination_fraction = None
        R_SUN_KM, AU_KM = 695_700.0, 149_597_870.7

try:
    from .moon_render import _vertex_normals
except ImportError:
    try:
        from moon_render import _vertex_normals
    except ImportError:
        _vertex_normals = None

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
    jd = epoch_jd + t_s/86400.0
    n_days = jd - 2_451_545.0
    L = np.radians((280.460 + 0.9856474*n_days) % 360)
    g = np.radians((357.528 + 0.9856003*n_days) % 360)
    lam = L + np.radians(1.915*np.sin(g) + 0.020*np.sin(2*g))
    return np.stack([np.cos(lam), np.sin(lam), np.zeros_like(lam)], axis=1)


_earth_texture_cache = None


def _load_real_earth_texture(n_lat, n_lon):
    """
    Real Earth texture, same source and loading pattern as the toolkit's
    own layers.py EarthLayer / globe_plot.py (ssapy.utils.find_file +
    PIL, resized to the mesh resolution). Cached after first call.
    Returns an (n_lat, n_lon, 3) uint8 array, or None if unavailable.
    """
    global _earth_texture_cache
    if _earth_texture_cache is not None and _earth_texture_cache.shape[:2] == (n_lat, n_lon):
        return _earth_texture_cache
    try:
        from ssapy.utils import find_file
        from PIL import Image
        img = Image.open(find_file("earth", ext=".png")).convert("RGB")
        img = img.resize((n_lon, n_lat), Image.LANCZOS)
        _earth_texture_cache = np.array(img)
        return _earth_texture_cache
    except Exception as ex:
        print(f"[globe_orbit_daynight_plotly] Real Earth texture not found "
              f"({ex}) — using procedural continents instead.")
        return None


def _smooth(field, sigma=1.0):
    """Same light wrap-aware Gaussian smoothing used in moon_render.py —
    kills hard blotch/tile edges in any procedural field before it's used
    for color or lighting, so nothing shows up as a faceted grid pattern
    once shaded."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma, mode="wrap")
    except Exception:
        return field


def _land_mask(n_lat, n_lon, seed=7):
    """Real land/ocean mask (global_land_mask, smoothed) shared between
    the procedural continent coloring and the night-side city-light
    speckle below — one source of truth instead of two separate blob
    generators that could disagree with each other."""
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    Lon, Lat = np.meshgrid(lon, lat)
    try:
        from global_land_mask import globe
        lon_q = np.where(Lon >= 180, Lon - 360, Lon)
        lat_q = np.clip(Lat, -89.999, 89.999)
        land = globe.is_land(lat_q, lon_q)
        land = _smooth(land.astype(float), sigma=1.6)
        return land, Lat, Lon
    except Exception:
        rng = np.random.default_rng(seed)
        field = np.zeros_like(Lat)
        for _ in range(14):
            clat, clon = rng.uniform(-60, 60), rng.uniform(-180, 180)
            spread = rng.uniform(15, 35)
            d = np.sqrt((Lat-clat)**2 + ((Lon-clon+180) % 360 - 180)**2)
            field += np.exp(-(d**2)/(2*spread**2))
        land = _smooth((field > 0.35).astype(float), sigma=1.2)
        return land, Lat, Lon


def _terrain_relief(Lat, Lon, land, seed=17):
    """
    Real geometric relief for Earth's mesh — mountain ranges/ridgelines on
    land, flat ocean — displaced into actual 3D bumps the same way
    moon_render.py displaces craters. This is the structural piece that
    was missing: a smooth, undisplaced sphere with only a flat day/night
    color gradient can never look as 3D as the Moon's cratered mesh, no
    matter how good the color is, because there's no local surface angle
    variation for light to catch. Adding real bumps gives every point a
    true per-vertex normal to shade against, not just a single global
    terminator gradient.
    """
    rng = np.random.default_rng(seed)
    relief = np.zeros_like(Lat)
    # Mountain ridges: elongated ridge-like ripples, not round craters —
    # real mountain belts run in long chains (Andes, Himalaya, Rockies),
    # so use a directional sine ripple modulated by a soft blob envelope
    # rather than the Moon's radially-symmetric crater bowls.
    for _ in range(10):
        clat, clon = rng.uniform(-60, 65), rng.uniform(-180, 180)
        length = rng.uniform(15, 40)
        width = rng.uniform(4, 9)
        angle = rng.uniform(0, np.pi)
        dlat = Lat - clat
        dlon = (Lon - clon + 180) % 360 - 180
        along = dlat * np.cos(angle) + dlon * np.sin(angle)
        across = -dlat * np.sin(angle) + dlon * np.cos(angle)
        envelope = np.exp(-(along / length) ** 2) * np.exp(-(across / width) ** 2)
        ridge = np.sin(along / width * 1.3) * envelope
        relief += ridge * rng.uniform(0.5, 1.0)
    # Fine hill/terrain roughness everywhere on land
    fine = np.zeros_like(Lat)
    for _ in range(40):
        clat, clon = rng.uniform(-85, 85), rng.uniform(-180, 180)
        spread = rng.uniform(3, 9)
        d = np.sqrt((Lat-clat)**2 + ((Lon-clon+180) % 360 - 180)**2)
        fine += rng.uniform(-0.3, 0.3) * np.exp(-(d**2)/(2*spread**2))
    relief = relief * 0.75 + fine * 0.5
    relief = _smooth(relief, sigma=0.5)
    # Zero out over ocean (open water is flat at this scale — no ridges)
    return relief * land


def _city_lights(n_lat, n_lon, land, Lat, seed=13):
    """
    Warm speckled glow on the night side over land, concentrated at
    mid-latitudes (real light pollution is overwhelmingly a mid-latitude,
    land-based phenomenon — sparse over deserts/tundra/ocean/ice) — this
    is what makes a night-side Earth read as a real photo instead of a
    flat dark disk, the same reason every actual satellite night image
    looks like scattered warm dots rather than uniform black.
    """
    rng = np.random.default_rng(seed)
    lat_band = np.clip(1 - (np.abs(Lat) - 15).clip(0) / 55, 0, 1) ** 1.5
    speckle = rng.random(Lat.shape)
    clusters = np.zeros_like(Lat)
    for _ in range(220):
        clat, clon = rng.normal(0, 35), rng.uniform(-180, 180)
        spread = rng.uniform(1.5, 5)
        d = np.sqrt((Lat - clat) ** 2 + ((rng.uniform(-180, 180) - clon + 180) % 360 - 180) ** 2)
        clusters += np.exp(-(d ** 2) / (2 * spread ** 2))
    clusters = clusters / (clusters.max() + 1e-9)
    lights = land * lat_band * np.clip(clusters * 1.3 + speckle * 0.15, 0, 1)
    return np.clip(lights, 0, 1)


def _procedural_continents(n_lat, n_lon, seed=7):
    """
    Real continent shapes via the `global_land_mask` package (a bundled
    real land/ocean mask, not guessed blobs) — used whenever the ssapy
    texture isn't found. Falls back to the old random-blob generator only
    if that package genuinely isn't installed either.
    """
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    Lon, Lat = np.meshgrid(lon, lat)
    land, Lat, Lon = _land_mask(n_lat, n_lon, seed=seed)

    ocean_color = np.array([0.09, 0.15, 0.27])
    land_color = np.array([0.22, 0.31, 0.16])
    land_frac = np.clip(land, 0, 1)[..., None]
    rgb = land_frac * land_color + (1 - land_frac) * ocean_color

    # Real ssapy/land_mask data was a flat 2-colour land/sea mask with no
    # terrain, ice, or depth variation at all — real Earth doesn't look
    # like that even schematically. Add real latitude-based effects and
    # procedural noise so it reads as an actual varied surface:
    rng = np.random.default_rng(seed)

    # Polar ice caps — real effect, not arbitrary: permanent ice above
    # ~70 deg latitude on both land and ocean (sea ice)
    ice = np.clip((np.abs(Lat) - 66) / 12, 0, 1)[..., None]
    ice_color = np.array([0.85, 0.88, 0.92])
    rgb = rgb * (1 - ice) + ice_color * ice

    # Terrain noise within land — desert/tan near-tropical bands, darker
    # green temperate/boreal, so land isn't one flat colour
    lat_desert_bias = np.clip(1 - np.abs(np.abs(Lat) - 22) / 20, 0, 1)
    noise = np.zeros_like(Lat)
    for _ in range(8):
        clat, clon = rng.uniform(-90, 90), rng.uniform(-180, 180)
        spread = rng.uniform(10, 30)
        d = np.sqrt((Lat-clat)**2 + ((Lon-clon+180) % 360 - 180)**2)
        noise += rng.uniform(-1, 1) * np.exp(-(d**2)/(2*spread**2))
    noise = noise / (np.abs(noise).max() + 1e-9)
    noise = _smooth(noise, sigma=0.6)
    desert_color = np.array([0.52, 0.44, 0.29])
    land_mask_3d = land[..., None] * (1 - ice)
    terrain_mix = np.clip(lat_desert_bias[..., None]*0.5 + noise[..., None]*0.35, 0, 1)
    rgb = np.where(land_mask_3d > 0.5, rgb*(1-terrain_mix) + desert_color*terrain_mix, rgb)

    # Ocean depth variation — subtle, so it isn't one flat blue
    ocean_noise = 0.5 + 0.5*noise
    deep_ocean = np.array([0.04, 0.08, 0.18])
    ocean_mix = np.clip(ocean_noise[..., None]*0.4, 0, 0.4)
    rgb = np.where((1-land_mask_3d) > 0.5, rgb*(1-ocean_mix) + deep_ocean*ocean_mix, rgb)

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _earth_atmosphere_trace(center=(0.0, 0.0, 0.0), radius_scale=1.0):
    """
    Thin translucent blue glow just outside Earth's surface — real photos
    from space always show this (Rayleigh-scattered sunlight in the upper
    atmosphere), and without it a bare rendered sphere reads as an
    artificial "toy globe" rather than a real planet. Optional companion
    trace, not baked into _earth_mesh itself since that function's return
    type (a single trace) is relied on in several places already.
    """
    n = 40
    u_, v_ = np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n)
    U, V = np.meshgrid(u_, v_)
    nx, ny, nz = np.cos(U)*np.sin(V), np.sin(U)*np.sin(V), np.cos(V)
    R = RE_KM * radius_scale * 1.015
    X = center[0] + R*nx
    Y = center[1] + R*ny
    Z = center[2] + R*nz
    return go.Surface(
        x=X, y=Y, z=Z, colorscale=[[0, "#6fa8ff"], [1, "#6fa8ff"]],
        showscale=False, opacity=0.10, hoverinfo="skip", showlegend=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), name="Atmosphere",
    )


def _earth_mesh(sun_hat, n_lat=180, n_lon=360, radius_scale=1.0, center=(0.0, 0.0, 0.0),
                shadow_body_center_km=None, shadow_body_radius_km=None, rotation_deg=0.0):
    """Real 3D sphere mesh, real Earth texture where available (falls back
    to procedural continents only if the real texture truly isn't found),
    with real day/night shading applied to the actual texture pixels —
    smooth per-vertex colouring, not flat per-face, and explicit flat
    self-lighting so Plotly's own default light doesn't fight the
    day/night shading we've computed from the actual sun direction.

    `center` places Earth anywhere in the scene (default: origin, for the
    geocentric orbit view) — needed for the heliocentric view where Earth
    itself orbits the Sun and isn't at (0,0,0).

    `rotation_deg` spins Earth about its own polar axis by this many
    degrees, for animating real sidereal rotation. Implemented by
    rolling the body-fixed fields (texture/continents, land mask,
    terrain relief) in longitude rather than recomputing them — the
    sphere's own (nx,ny,nz) grid stays fixed in the inertial scene frame,
    and every longitude-dependent quantity is resampled from its
    body-fixed value at (lon - rotation_deg), which is what real
    rotation means for a body-fixed feature (a continent, a mountain
    range) as seen against fixed stars. This is far cheaper than
    rebuilding the land mask / relief every animation frame, and gives
    an identical result to actually rotating the geometry for a sphere."""
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    Lon, Lat = np.meshgrid(lon, lat)
    latr, lonr = np.radians(Lat), np.radians(Lon)

    shift = int(round((rotation_deg % 360) / 360.0 * n_lon))

    tex = _load_real_earth_texture(n_lat, n_lon)
    if tex is not None:
        # Longitude wrap fix — same as globe_plot.py's `tc = (c + W//2) % W`:
        # most equirectangular Earth textures have column 0 at the
        # antimeridian, not the prime meridian, so shift by half the width.
        tex = np.roll(tex, n_lon // 2, axis=1)
        rgb = tex.astype(float) / 255.0
        base_desc = "real ssapy Earth texture"
    else:
        rgb = _procedural_continents(n_lat, n_lon).astype(float) / 255.0
        base_desc = "procedural continents (fallback)"
    print(f"[globe_orbit_daynight_plotly] Earth base: {base_desc}")
    if shift:
        rgb = np.roll(rgb, shift, axis=1)

    nx, ny, nz = np.cos(latr)*np.cos(lonr), np.cos(latr)*np.sin(lonr), np.sin(latr)
    dot = nx*sun_hat[0] + ny*sun_hat[1] + nz*sun_hat[2]


    # More realistic terminator: real daylight brightness follows the
    # sun-angle cosine fairly faithfully across most of the dayside (not
    # the old power-0.6 curve, which over-flattened it into an almost
    # uniform grey-white disk), then drops steeply through a narrow
    # twilight band rather than a slow linear fade — this is what a real
    # photo's day/night boundary actually looks like: a fairly crisp line
    # with a thin gradient, not a broad soft blend.
    day_core = np.clip(dot, 0, 1) ** 0.92
    twilight = 1.0 / (1.0 + np.exp(-dot / 0.045))   # steep sigmoid, ~few deg wide
    # Floor raised twice now: originally 0.045 (indistinguishable from
    # black at any brightness), then 0.15 (still too close to black for
    # naturally dark colors like ocean — 15% of an already-dark navy
    # albedo is still ~black on screen). Settled on 0.30: strong day/night
    # contrast is still obvious, but land/ocean texture actually stays
    # legible on the night side, which matters more here than matching
    # a real photo's true darkness — this is a diagnostic visualization,
    # not a photorealistic renderer.
    day_night_brightness = np.clip(0.22 + 0.78 * day_core * twilight, 0.22, 1.0)

    # Twilight glow: real sunrise/sunset bands look blue/orange from
    # Rayleigh scattering in the atmosphere, brightest exactly where the
    # surface itself is darkest (grazing sun). Added as a color tint
    # layered in below, not just a brightness multiplier, since real
    # twilight visibly shifts hue, not just dims.
    twilight_band = np.exp(-(dot / 0.06) ** 2) * (1 - np.clip(dot, 0, 1))
    twilight_color = np.array([1.00, 0.55, 0.30])  # warm sunset-orange near the terminator

    # Real geometric terrain relief (mountain ridges on land, flat ocean),
    # displaced into actual 3D bumps — this is the piece that was missing
    # entirely before: a bare smooth sphere with only a global day/night
    # color gradient has no local surface-angle variation, so it can
    # never look as sculpted as the Moon's cratered mesh regardless of
    # how good the color texture is. Real per-vertex normals off this
    # bumpy surface add local light-and-shadow the same way they do for
    # the Moon, layered under the (still dominant) day/night term.
    land_for_lights, Lat_ll, Lon_ll = _land_mask(n_lat, n_lon)
    if _vertex_normals is not None:
        relief = _terrain_relief(Lat_ll, Lon_ll, land_for_lights)
        if shift:
            # Roll AFTER generating from the unrolled body-fixed grid —
            # ridges are computed at fixed physical lat/lon coordinates,
            # so rolling the input land mask first (before ridges are
            # placed) would let a ridge sit over what the rolled mask
            # calls "ocean" purely as an artifact of rotation, not real
            # geography moving. Rolling the finished field instead keeps
            # every mountain locked to the same patch of rotating color.
            relief = np.roll(relief, shift, axis=1)
            land_for_lights = np.roll(land_for_lights, shift, axis=1)
        R_disp = 1.0 + relief * 0.006    # bumped up (was 0.0022) — the
        # smaller value was true-to-scale but visually imperceptible;
        # this is a deliberate exaggeration, same spirit as the Moon's
        # boosted crater displacement, purely so relief actually reads.
        Xn, Yn, Zn = R_disp*nx, R_disp*ny, R_disp*nz
        nxr, nyr, nzr = _vertex_normals(Xn, Yn, Zn)
        terrain_diffuse = np.clip(nxr*sun_hat[0] + nyr*sun_hat[1] + nzr*sun_hat[2], 0, 1)
        # Blend: mostly the smooth global day/night term (correct large-
        # scale lighting), modulated more strongly than before by the
        # local terrain normal so mountains actually catch/lose light
        # instead of being flat color patches.
        terrain_shade = np.clip(0.65 + 0.70*(terrain_diffuse - np.clip(dot, 0, 1)), 0.55, 1.5)
    else:
        if shift:
            land_for_lights = np.roll(land_for_lights, shift, axis=1)
        R_disp = np.ones_like(Lat)
        terrain_shade = np.ones_like(Lat)

    night_lights = _city_lights(n_lat, n_lon, land_for_lights, Lat_ll)
    night_factor = np.clip(1 - np.clip(dot, 0, 1) * 3.0, 0, 1)  # only near/past the terminator
    light_color = np.array([1.0, 0.78, 0.45])

    # Ocean sun-glint: real satellite photos show a bright, soft
    # specular highlight on open water centered near the sub-solar
    # point — water is a much better specular reflector than land, and
    # without this every ocean pixel just looks like flat matte paint.
    # Not a real view-dependent specular reflection (this is baked into
    # per-vertex color, not shaded per-frame), but anchored at the
    # sub-solar point it reads correctly for any camera roughly on the
    # dayside, which is the common case for these scenes.
    ocean_mask = 1.0 - land_for_lights
    glint = np.clip(dot, 0, 1) ** 60   # much tighter (was **14) — a real
    # sun-glint highlight is a small bright patch, not a broad sheen
    # covering a third of the visible disk
    glint_color = np.array([1.0, 1.0, 0.96])

    eclipse_brightness = np.ones_like(day_night_brightness)
    if shadow_body_center_km is not None and shadow_body_radius_km is not None and illumination_fraction is not None:
        # Real per-vertex Moon-shadow check for the solar eclipse case —
        # this is what actually renders the Moon's shadow path across
        # Earth's globe, not just the regular day/night terminator.
        center_arr = np.asarray(center, dtype=float)
        real_surface = np.stack([nx, ny, nz], axis=-1) * RE_KM
        real_positions = center_arr[None, None, :] + real_surface
        flat_pos = (real_positions - shadow_body_center_km[None, None, :]).reshape(-1, 3)
        flat_sun = np.tile(sun_hat, (flat_pos.shape[0], 1))
        eclipse_illum = illumination_fraction(flat_pos, flat_sun,
                                              R_body_km=shadow_body_radius_km,
                                              R_sun_km=R_SUN_KM, D_km=AU_KM)
        # Just darker, not recoloured — real sunlight being blocked, not a
        # different light source, so brightness only, no colour shift.
        eclipse_brightness = np.clip(eclipse_illum, 0.06, 1.0).reshape(day_night_brightness.shape)
        print(f"[globe_orbit_daynight_plotly] Moon shadow on Earth: "
              f"min brightness {eclipse_brightness.min():.3f}")

    brightness = day_night_brightness * eclipse_brightness * terrain_shade
    rgb_shaded = np.clip(rgb * brightness[..., None], 0, 1)

    # Twilight tint and ocean glint both represent real sunlight arriving
    # at the surface — if a point is inside the Moon's shadow during a
    # solar eclipse, there's less direct sunlight to scatter or reflect,
    # so both need the same eclipse_brightness reduction the base shading
    # already gets. Without this, a point deep in the umbra would still
    # show a bright glint or twilight tint, which is physically backwards
    # — the eclipse is exactly what's blocking the light that creates
    # either effect.
    rgb_shaded = np.clip(
        rgb_shaded + twilight_band[..., None] * twilight_color[None, None, :] * 0.35 * (eclipse_brightness[..., None] ** 3),
        0, 1)
    rgb_shaded = np.clip(
        rgb_shaded + (glint * ocean_mask)[..., None] * glint_color[None, None, :] * 0.5 * (eclipse_brightness[..., None] ** 3),
        0, 1)

    light_add = (night_lights * night_factor)[..., None] * light_color[None, None, :] * 0.9
    rgb_shaded = np.clip(rgb_shaded + light_add, 0, 1)

    # Shadow-lift curve — NOT a naive global gamma. The previous version
    # (rgb**(1/1.9) applied to every pixel) did fix the night side's
    # visibility, but it also brightened the already well-exposed DAY
    # side by the same amount, washing out continent contrast and
    # saturation into the flat, hazy look seen in the animated renders.
    # This curve (1-(1-x)^gamma) lifts dark values substantially while
    # leaving values near 1.0 almost untouched — e.g. a night-side 0.22
    # lifts to ~0.39, while a bright daytime 0.90 only moves to ~0.99,
    # instead of both being pushed upward by the same multiplicative
    # amount.
    rgb_shaded = 1.0 - (1.0 - np.clip(rgb_shaded, 0, 1)) ** 1.9

    R = RE_KM * radius_scale
    X, Y, Z = center[0] + R*R_disp*nx, center[1] + R*R_disp*ny, center[2] + R*R_disp*nz

    vertexcolor = [
        f"rgb({int(rgb_shaded[r,c,0]*255)},{int(rgb_shaded[r,c,1]*255)},{int(rgb_shaded[r,c,2]*255)})"
        for r in range(n_lat) for c in range(n_lon)
    ]

    ii, jj, kk = [], [], []
    for r in range(n_lat - 1):
        for c in range(n_lon - 1):
            v0 = r*n_lon + c; v1 = v0+1; v2 = (r+1)*n_lon + c; v3 = v2+1
            ii += [v0, v1]; jj += [v1, v3]; kk += [v2, v2]

    return go.Mesh3d(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        i=ii, j=jj, k=kk,
        vertexcolor=vertexcolor,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        name="Earth", hoverinfo="skip", showlegend=False,
    )


def _sun_sphere_traces(pos, radius_km, seed=11):
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
    n = 64   # higher again (was 48) for smoother granulation and a rounder limb
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
    ref = np.array([0.4, 0.4, 0.82]); ref /= np.linalg.norm(ref)
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
    for glow_scale, glow_op in [(1.4, 0.12), (1.9, 0.05)]:
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


def plot_globe_orbit_daynight_plotly(a_km, e, inc_deg, raan_deg=0.0, argp_deg=0.0,
                                     nu0_deg=0.0, sat_name="Satellite",
                                     n_orbits=1.0, n_steps=1500, save_path=None):
    t_s, r_eci, T_s = propagate_eci(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg,
                                     n_orbits=n_orbits, n_steps=n_steps)
    sun_hat = sun_direction_eci(t_s)[0]

    orbit_r = np.max(np.linalg.norm(r_eci, axis=1))
    frame_r = max(orbit_r, RE_KM*1.3)
    sun_dist = frame_r * 1.55
    sun_pos = sun_hat * sun_dist
    sun_radius = frame_r * 0.09

    fig = go.Figure()
    fig.add_trace(_earth_mesh(sun_hat))

    # Orbit path, colour-mapped along its length
    n_pts = len(r_eci)
    colors = np.linspace(0, 1, n_pts)
    fig.add_trace(go.Scatter3d(
        x=r_eci[:, 0], y=r_eci[:, 1], z=r_eci[:, 2],
        mode="lines",
        line=dict(color=colors, colorscale="Plasma", width=6),
        name=sat_name, hoverinfo="skip",
    ))

    for tr in _sun_sphere_traces(sun_pos, sun_radius):
        fig.add_trace(tr)

    lim = frame_r * 1.9
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
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        title=dict(text=f"{sat_name} — orbit around Earth, with the Sun shown in frame",
                  x=0.5, font=dict(color="white", size=16)),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, width=1400, height=1000, scale=1)
        print(f"Saved -> {save_path}")
    return fig


if __name__ == "__main__":
    plot_globe_orbit_daynight_plotly(
        a_km=26_560.0, e=0.001, inc_deg=55.0,
        sat_name="GPS-like MEO", n_orbits=1.0, n_steps=1500,
        save_path="/home/claude/demo/outputs/globe_orbit_daynight_plotly.png",
    )
    plot_globe_orbit_daynight_plotly(
        a_km=26_560.0, e=0.001, inc_deg=55.0,
        sat_name="GPS-like MEO", n_orbits=1.0, n_steps=1500,
        save_path="/home/claude/demo/outputs/globe_orbit_daynight_plotly.html",
    )