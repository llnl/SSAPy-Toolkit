import numpy as np
import astropy.units as u

from ..constants import MOON_RADIUS, EARTH_RADIUS  # RGEO was unused; removed


# --- small helpers to avoid circular imports at import-time ---

def _get_angle(a, b, c):
    """Lazy import to break vectors <-> plots cycles."""
    from ..vectors import getAngle  # local import avoids circular import at package import time
    return getAngle(a, b, c)


def _get_body(name):
    """Lazy import of ssapy.get_body to avoid hard dependency at import time."""
    from ssapy import get_body
    return get_body(name)


# ------------------------- flux components -------------------------

def moon_shine(
    r_moon: np.ndarray,
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    radius: float,
    albedo: float,
    albedo_moon: float,
    albedo_back: float,
    albedo_front: float,
    area_panels: float,
) -> dict:
    """
    Flux from Moon-reflected sunlight onto the satellite (bus + panels).
    Returns a dict with 'moon_bus' and 'moon_panels' arrays.
    """
    moon_phase_angle = _get_angle(r_sun, r_moon, r_sat)     # ∠(Sun, Moon, Sat)
    sun_angle = _get_angle(r_sun, r_sat, r_moon)            # ∠(Sun, Sat, Moon)
    moon_to_earth_angle = _get_angle(r_moon, r_sat, r_earth)

    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)

    # Lambertian reflected flux Moon->Sat
    flux_moon_to_sat = (
        2.0 / 3.0 * albedo_moon * MOON_RADIUS**2
        / (np.pi * (r_moon_sat**2))
        * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))
    )

    # Panels: split into back/front depending on illumination geometry
    flux_back = np.zeros_like(sun_angle)
    m = sun_angle > (np.pi / 2.0)
    if np.any(m):
        flux_back[m] = np.abs(
            albedo_back * area_panels
            / (np.pi * (r_earth_sat[m] ** 2))
            * np.cos(np.pi - moon_to_earth_angle[m])
            * flux_moon_to_sat[m]
        )

    flux_front = np.zeros_like(sun_angle)
    m = sun_angle < (np.pi / 2.0)
    if np.any(m):
        flux_front[m] = np.abs(
            albedo_front * area_panels
            / (np.pi * (r_earth_sat[m] ** 2))
            * np.cos(moon_to_earth_angle[m])
            * flux_moon_to_sat[m]
        )

    flux_panels = flux_back + flux_front

    # Bus term
    flux_bus = (
        2.0 / 3.0 * albedo * radius**2
        / (np.pi * (r_earth_sat**2))
        * flux_moon_to_sat
    )

    return {"moon_bus": flux_bus, "moon_panels": flux_panels}


def earth_shine(
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    radius: float,
    albedo: float,
    albedo_earth: float,
    albedo_back: float,
    area_panels: float,
) -> dict:
    """
    Flux from Earth's reflected sunlight (Earthshine) onto the satellite.
    Returns dict with 'earth_bus' and 'earth_panels' (back-side) arrays.
    """
    phase_angle = _get_angle(r_sun, r_sat, r_earth)  # ∠(Sun, Sat, Earth)
    earth_angle = np.pi - phase_angle
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)

    flux_earth_to_sat = (
        2.0 / 3.0 * albedo_earth * EARTH_RADIUS**2
        / (np.pi * (r_earth_sat**2))
        * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))
    )

    # Panels (back) when phase places Sun behind the sat relative to Earth
    flux_back = np.zeros_like(phase_angle)
    m = phase_angle > (np.pi / 2.0)
    if np.any(m):
        flux_back[m] = (
            albedo_back * area_panels
            / (np.pi * (r_earth_sat[m] ** 2))
            * np.cos(np.pi - phase_angle[m])
            * flux_earth_to_sat[m]
        )

    flux_bus = (
        2.0 / 3.0 * albedo * radius**2
        / (np.pi * (r_earth_sat**2))
        * flux_earth_to_sat
    )

    return {"earth_bus": flux_bus, "earth_panels": flux_back}


def sun_shine(
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    radius: float,
    albedo: float,
    albedo_front: float,
    area_panels: float,
) -> dict:
    """
    Direct Sun contribution on bus + front panels (Lambertian geometry).
    Returns dict with 'sun_bus' and 'sun_panels' arrays.
    """
    phase_angle = _get_angle(r_sun, r_sat, r_earth)  # ∠(Sun, Sat, Earth)
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)

    flux_front = np.zeros_like(phase_angle)
    m = phase_angle < (np.pi / 2.0)
    if np.any(m):
        flux_front[m] = (
            albedo_front * area_panels
            / (np.pi * (r_earth_sat[m] ** 2))
            * np.cos(phase_angle[m])
        )

    flux_bus = (
        2.0 / 3.0 * albedo * radius**2
        / (np.pi * (r_earth_sat**2))
        * (np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle))
    )

    return {"sun_bus": flux_bus, "sun_panels": flux_front}


# ------------------------- magnitude wrappers -------------------------

def calc_M_v(
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    r_moon=False,                 # np.ndarray or False; no typing.Union used
    radius: float = 0.4,
    albedo: float = 0.20,
    sun_Mag: float = 4.80,
    albedo_earth: float = 0.30,
    albedo_moon: float = 0.12,
    albedo_back: float = 0.50,
    albedo_front: float = 0.05,
    area_panels: float = 100.0,
    return_components: bool = False,
):
    """
    Apparent magnitude from combined Sun/Earth/Moon reflected fluxes.

    Returns either the magnitude array, or (magnitude, components_dict) if
    return_components is True.
    """
    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)

    f_sun = sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels)
    f_earth = earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels)

    if isinstance(r_moon, np.ndarray) and r_moon.size:
        f_moon = moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels)
    else:
        # zeros with the same shape as the bus term (uses r_sun_sat shape)
        zeros = np.zeros_like(r_sun_sat)
        f_moon = {"moon_bus": zeros, "moon_panels": zeros}

    components = {**f_sun, **f_earth, **f_moon}

    # Sum component arrays explicitly (works for scalar fallback too)
    total_frac_flux = np.sum(list(components.values()), axis=0)

    # Convert 10 pc to meters via astropy for clarity
    ten_pc_in_m = (10 * u.pc).to(u.m).value
    Mag_v = (2.5 * np.log10((r_sun_sat / ten_pc_in_m) ** 2) + sun_Mag) - 2.5 * np.log10(total_frac_flux)

    return (Mag_v, components) if return_components else Mag_v


def M_v_lambertian(
    r_sat: np.ndarray,
    times: np.ndarray,
    radius: float = 1.0,
    albedo: float = 0.20,
    sun_Mag: float = 4.80,
    albedo_earth: float = 0.30,
    albedo_moon: float = 0.12,
    albedo_back: float = 0.50,
    albedo_front: float = 0.05,
    area_panels: float = 100.0,
):
    """
    Visual magnitude time series using Lambertian reflectance for Sun, Earth, Moon.
    """
    # Ephemerides (lazy import keeps import-time light)
    Sun = _get_body("Sun")
    Moon = _get_body("Moon")

    r_sun = Sun.position(times).T
    r_moon = Moon.position(times).T
    r_earth = np.zeros_like(r_sun)

    # Use the same core calculator to compose components and magnitude
    Mag_v = calc_M_v(
        r_sat=r_sat,
        r_earth=r_earth,
        r_sun=r_sun,
        r_moon=r_moon,
        radius=radius,
        albedo=albedo,
        sun_Mag=sun_Mag,
        albedo_earth=albedo_earth,
        albedo_moon=albedo_moon,
        albedo_back=albedo_back,
        albedo_front=albedo_front,
        area_panels=area_panels,
        return_components=False,
    )
    return Mag_v
