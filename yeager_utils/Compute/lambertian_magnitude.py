import numpy as np
from typing import Dict, Union, Tuple
from ..constants import MOON_RADIUS, EARTH_RADIUS, RGEO
from ..vectors import getAngle
import astropy.units as u
from ssapy import get_body


def moon_shine(r_moon: np.ndarray, r_sat: np.ndarray, r_earth: np.ndarray, r_sun: np.ndarray, radius: float,
               albedo: float, albedo_moon: float, albedo_back: float, albedo_front: float, area_panels: float) -> Dict[str, np.ndarray]:
    """
    Calculate the moonshine flux to a satellite.

    Parameters:
    r_moon, r_sat, r_earth, r_sun (np.ndarray): Position vectors for the Moon, Satellite, Earth, and Sun.
    radius (float): The radius of the satellite.
    albedo (float): The albedo of the satellite.
    albedo_moon (float): The albedo of the Moon.
    albedo_back (float): The albedo of the back of the satellite.
    albedo_front (float): The albedo of the front of the satellite.
    area_panels (float): The area of the solar panels.

    Returns:
    dict: A dictionary with keys 'moon_bus' and 'moon_panels', containing the flux values.
    """
    moon_phase_angle = getAngle(r_sun, r_moon, r_sat)
    sun_angle = getAngle(r_sun, r_sat, r_moon)
    moon_to_earth_angle = getAngle(r_moon, r_sat, r_earth)
    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)

    flux_moon_to_sat = 2 / 3 * albedo_moon * MOON_RADIUS**2 / (np.pi * (r_moon_sat)**2) * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))
    flux_back = np.zeros_like(sun_angle)
    flux_back[sun_angle > np.pi / 2] = np.abs(albedo_back * area_panels / (np.pi * r_earth_sat[sun_angle > np.pi / 2]**2) * np.cos(np.pi - moon_to_earth_angle[sun_angle > np.pi / 2]) * flux_moon_to_sat[sun_angle > np.pi / 2])

    flux_front = np.zeros_like(sun_angle)
    flux_front[sun_angle < np.pi / 2] = np.abs(albedo_front * area_panels / (np.pi * r_earth_sat[sun_angle < np.pi / 2]**2) * np.cos(moon_to_earth_angle[sun_angle < np.pi / 2]) * flux_moon_to_sat[sun_angle < np.pi / 2])

    flux_panels = flux_back + flux_front
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_moon_to_sat
    return {'moon_bus': flux_bus, 'moon_panels': flux_panels}


def earth_shine(
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    radius: float,
    albedo: float,
    albedo_earth: float,
    albedo_back: float,
    area_panels: float
) -> Dict[str, np.ndarray]:
    """
    Calculate the flux from Earth's reflected sunlight (Earthshine) onto a satellite.

    Parameters:
    - r_sat: The satellite position in 3D space (array).
    - r_earth: The position of the Earth in 3D space (array).
    - r_sun: The position of the Sun in 3D space (array).
    - radius: The radius of the satellite (meters).
    - albedo: The albedo of the satellite.
    - albedo_earth: The albedo of the Earth.
    - albedo_back: The albedo of the satellite's back panels.
    - area_panels: The total area of the satellite's solar panels (square meters).

    Returns:
    - A dictionary containing flux contributions from Earth's reflection, including 
      flux on the satellite's bus and panels.
    """
    phase_angle = getAngle(r_sun, r_sat, r_earth)
    earth_angle = np.pi - phase_angle
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)

    flux_earth_to_sat = 2 / 3 * albedo_earth * EARTH_RADIUS**2 / (np.pi * r_earth_sat**2) * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))

    flux_back = np.zeros_like(phase_angle)
    flux_back[phase_angle > np.pi / 2] = albedo_back * area_panels / (np.pi * r_earth_sat[phase_angle > np.pi / 2]**2) * np.cos(np.pi - phase_angle[phase_angle > np.pi / 2]) * flux_earth_to_sat[phase_angle > np.pi / 2]

    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_earth_to_sat

    return {'earth_bus': flux_bus, 'earth_panels': flux_back}


def sun_shine(
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    radius: float,
    albedo: float,
    albedo_front: float,
    area_panels: float
) -> Dict[str, np.ndarray]:
    """
    Calculate the flux from the Sun's reflection onto a satellite's panels and bus.

    Parameters:
    - r_sat: The satellite position in 3D space (array).
    - r_earth: The position of the Earth in 3D space (array).
    - r_sun: The position of the Sun in 3D space (array).
    - radius: The radius of the satellite (meters).
    - albedo: The albedo of the satellite.
    - albedo_front: The albedo of the satellite's front panels.
    - area_panels: The total area of the satellite's solar panels (square meters).

    Returns:
    - A dictionary containing flux contributions from the Sun's reflection, including 
      flux on the satellite's bus and panels.
    """
    phase_angle = getAngle(r_sun, r_sat, r_earth)
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)

    flux_front = np.zeros_like(phase_angle)
    flux_front[phase_angle < np.pi / 2] = albedo_front * area_panels / (np.pi * r_earth_sat[phase_angle < np.pi / 2]**2) * np.cos(phase_angle[phase_angle < np.pi / 2])

    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * (r_earth_sat)**2) * (np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle))

    return {'sun_bus': flux_bus, 'sun_panels': flux_front}


def calc_M_v(
    r_sat: np.ndarray,
    r_earth: np.ndarray,
    r_sun: np.ndarray,
    r_moon: Union[np.ndarray, bool] = False,
    radius: float = 0.4,
    albedo: float = 0.20,
    sun_Mag: float = 4.80,
    albedo_earth: float = 0.30,
    albedo_moon: float = 0.12,
    albedo_back: float = 0.50,
    albedo_front: float = 0.05,
    area_panels: float = 100,
    return_components: bool = False
) -> Union[float, Tuple[float, Dict[str, np.ndarray]]]:
    """
    Calculate the apparent magnitude (M_v) of a satellite due to reflections from the Sun, Earth, and optionally the Moon.

    This function computes the apparent magnitude of a satellite based on its reflected light from the Sun, Earth, and optionally the Moon. It uses separate functions to calculate the flux contributions from each of these sources and combines them to determine the overall apparent magnitude.

    Parameters:
    ----------
    r_sat : (n, 3) numpy.ndarray
        Position of the satellite in meters.
    r_earth : (n, 3) numpy.ndarray
        Position of the Earth in meters.
    r_sun : (n, 3) numpy.ndarray
        Position of the Sun in meters.
    r_moon : (n, 3) numpy.ndarray or False, optional
        Position of the Moon in meters. If False, the Moon's contribution is ignored (default is False).
    radius : float, optional
        Radius of the satellite in meters (default is 0.4 m).
    albedo : float, optional
        Albedo of the satellite's surface, representing its reflectivity (default is 0.20).
    sun_Mag : float, optional
        Solar magnitude (apparent magnitude of the Sun) used in magnitude calculations (default is 4.80).
    albedo_earth : float, optional
        Albedo of the Earth, representing its reflectivity (default is 0.30).
    albedo_moon : float, optional
        Albedo of the Moon, representing its reflectivity (default is 0.12).
    albedo_back : float, optional
        Albedo of the back surface of the satellite (default is 0.50).
    albedo_front : float, optional
        Albedo of the front surface of the satellite (default is 0.05).
    area_panels : float, optional
        Area of the satellite's panels in square meters (default is 100 m^2).
    return_components : bool, optional
        If True, returns the magnitude as well as the flux components from the Sun, Earth, and Moon (default is False).

    Returns:
    -------
    float
        The apparent magnitude (M_v) of the satellite as observed from Earth.

    dict, optional
        If `return_components` is True, a dictionary containing the flux components from the Sun, Earth, and Moon.

    Notes:
    ------
    - The function uses separate calculations for flux contributions from the Sun, Earth, and Moon:
        - `sun_shine` calculates the flux from the Sun.
        - `earth_shine` calculates the flux from the Earth.
        - `moon_shine` calculates the flux from the Moon (if applicable).
    - The apparent magnitude is calculated based on the distances between the satellite, Sun, Earth, and optionally the Moon, as well as their respective albedos and other parameters.

    Example usage:
    --------------
    >>> r_sat = np.array([[1e7, 2e7, 3e7]])
    >>> r_earth = np.array([1.496e11, 0, 0])
    >>> r_sun = np.array([0, 0, 0])
    >>> Mag_v = calc_M_v(r_sat, r_earth, r_sun, return_components=True)
    >>> Mag_v
    (15.63, {'sun_bus': 0.1, 'sun_panels': 0.2, 'earth_bus': 0.05, 'earth_panels': 0.1, 'moon_bus': 0.03, 'moon_panels': 0.07})

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)
    frac_flux_sun = sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels)
    frac_flux_earth = earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels)

    frac_flux_moon = {'moon_bus': 0, 'moon_panels': 0}
    if r_moon is not False:
        frac_flux_moon = moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels)

    merged_dict = {**frac_flux_sun, **frac_flux_earth, **frac_flux_moon}
    total_frac_flux = sum(merged_dict.values())

    Mag_v = (2.5 * np.log10((r_sun_sat / (10 * u.Unit('parsec').to(u.Unit('m'))))**2) + sun_Mag) - 2.5 * np.log10(total_frac_flux)

    if return_components:
        return Mag_v, merged_dict
    else:
        return Mag_v


def M_v_lambertian(
    r_sat: np.ndarray,
    times: np.ndarray,
    radius: float = 1.0,
    albedo: float = 0.20,
    sun_Mag: float = 4.80,
    albedo_earth: float = 0.30,
    albedo_moon: float = 0.12,
) -> float:
    """
    Calculate the visual magnitude of a satellite using Lambertian reflectance for Sun, Earth, and Moon fluxes.

    Parameters:
    - r_sat: The satellite positions over time in 3D space (array).
    - times: The times corresponding to the satellite positions (array).
    - radius: The radius of the satellite (meters).
    - albedo: The albedo of the satellite.
    - sun_Mag: The apparent magnitude of the Sun.
    - albedo_earth: The albedo of the Earth.
    - albedo_moon: The albedo of the Moon.
    - plot: If True, plot the satellite's visual magnitude and flux components over time.

    Returns:
    - The visual magnitude of the satellite.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    pc_to_m = 3.085677581491367e+16
    r_sun = get_body('Sun').position(times).T
    r_moon = get_body('Moon').position(times).T
    r_earth = np.zeros_like(r_sun)

    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat, axis=-1)
    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)

    frac_flux_sun = sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels)
    frac_flux_earth = earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels)
    frac_flux_moon = moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels)

    merged_dict = {**frac_flux_sun, **frac_flux_earth, **frac_flux_moon}
    total_frac_flux = sum(merged_dict.values())

    Mag_v = (2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(total_frac_flux)
    return Mag_v
