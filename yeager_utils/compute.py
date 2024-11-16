import numpy as np
from .constants import EARTH_RADIUS, MOON_RADIUS, RGEO
from .utils import divby0
from ssapy.body import get_body
from astropy import units as u
from typing import Tuple, Dict, Union


def find_smallest_bounding_cube(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the smallest bounding cube for a set of 3D coordinates.

    Parameters:
    r (np.ndarray): An array of shape (n, 3) containing the 3D coordinates.

    Returns:
    tuple: A tuple containing the lower and upper bounds of the bounding cube.
    """
    min_coords = np.min(r, axis=0)
    max_coords = np.max(r, axis=0)
    ranges = max_coords - min_coords
    max_range = np.max(ranges)
    center = (max_coords + min_coords) / 2
    half_side_length = max_range / 2
    lower_bound = center - half_side_length
    upper_bound = center + half_side_length

    return lower_bound, upper_bound


def calculate_errors(data: np.ndarray, CI: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the confidence interval errors for a dataset.

    Parameters:
    data (np.ndarray): The input data array.
    CI (float): The confidence interval, default is 0.05.

    Returns:
    tuple: A tuple containing the error bounds and the median of the data.
    """
    data_median = []
    data = np.sort(data)
    median_ = np.nanmedian(data)
    data_median.append(median_)
    err = [median_ - data[int(CI * len(data))], data[int((1 - CI) * len(data))] - median_]
    return err, data_median


def FFT(data: np.ndarray, time_between_samples: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a Fast Fourier Transform on the input data.

    Parameters:
    data (np.ndarray): The input time-series data.
    time_between_samples (float): Time interval between data samples.

    Returns:
    tuple: The frequency array and the FFT result.
    """
    N = len(data)
    k = int(N / 2)
    f = np.linspace(0.0, 1 / (2 * time_between_samples), N // 2)
    Y = np.abs(np.fft.fft(data))[:k]
    return f, Y


def FFTP(data: np.ndarray, time_between_samples: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a Fast Fourier Transform and calculate the period.

    Parameters:
    data (np.ndarray): The input time-series data.
    time_between_samples (float): Time interval between data samples.

    Returns:
    tuple: The period array and the FFT result.
    """
    N = len(data)
    k = int(N / 2)
    f = np.linspace(0.0, 1 / (2 * time_between_samples), N // 2)
    Tp = [divby0(1, float(item), len(data) * time_between_samples) for item in f]
    Y = np.abs(np.fft.fft(data))[:k]
    return Tp, Y


def proper_motion(x: np.ndarray, y: np.ndarray, z: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                  xe: float = 0, ye: float = 0, ze: float = 0, vxe: float = 0, vye: float = 0, vze: float = 0,
                  input_unit: str = 'si') -> Union[float, None]:
    """
    Calculate the proper motion of an object in space relative to Earth.

    Parameters:
    x, y, z (np.ndarray): Position coordinates of the object.
    vx, vy, vz (np.ndarray): Velocity components of the object.
    xe, ye, ze (float): Position of Earth.
    vxe, vye, vze (float): Velocity of Earth.
    input_unit (str): The unit for proper motion ('si' or 'rebound').

    Returns:
    float or None: The proper motion in arcseconds per year, or None if the object is at the Earth's position.
    """
    x_rot = x - xe
    y_rot = y - ye
    z_rot = z - ze
    vx_rot = vx - vxe
    vy_rot = vy - vye
    vz_rot = vz - vze

    d_earth_mag = np.linalg.norm([x_rot, y_rot, z_rot])
    if d_earth_mag == 0:
        return np.nan

    v_ast_earth = np.array([vx_rot, vy_rot, vz_rot])
    los_vector = np.array([x_rot, y_rot, z_rot])

    v_los = np.linalg.norm((np.dot(v_ast_earth, los_vector) / np.linalg.norm(los_vector)))
    v_transverse = np.sqrt(np.linalg.norm(v_ast_earth)**2 - v_los**2)
    
    if input_unit == 'si':
        return v_transverse / d_earth_mag * 206265
    elif input_unit == 'rebound':
        return v_transverse / d_earth_mag * 206265 / (31557600 * 2 * np.pi)
    else:
        print('Error - units provided not available, provide either SI or rebound units.')
        return None


def getAngle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate the angle between vectors a, b, and c, where b is the vertex.

    Parameters:
    a, b, c (np.ndarray): Input vectors.

    Returns:
    np.ndarray: The angle between the vectors in radians.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.arccos(cosine_angle)


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
    Calculate the visual magnitude of a satellite based on its albedo and reflections from Earth, Moon, and Sun.
    
    Parameters:
    - r_sat: The satellite position in 3D space (array).
    - r_earth: The position of the Earth in 3D space (array).
    - r_sun: The position of the Sun in 3D space (array).
    - r_moon: The position of the Moon in 3D space (array, optional).
    - radius: The radius of the satellite (meters).
    - albedo: The albedo of the satellite.
    - sun_Mag: The apparent magnitude of the Sun.
    - albedo_earth: The albedo of the Earth.
    - albedo_moon: The albedo of the Moon.
    - albedo_back: The albedo of the satellite's back panels.
    - albedo_front: The albedo of the satellite's front panels.
    - area_panels: The total area of the satellite's solar panels (square meters).
    - return_components: If True, return the components of the flux along with the magnitude.
    
    Returns:
    - The visual magnitude of the satellite.
    - If `return_components` is True, a tuple containing the magnitude and a dictionary of flux components.
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
    plot: bool = False
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


def megno(r: np.ndarray) -> float:
    """
    Calculate the MEGNO (Mean Exponential Growth of Nearby Orbits) value for a set of orbital states.
    
    The MEGNO is a measure of the chaos in the orbital evolution. It quantifies the exponential 
    divergence of nearby trajectories over time, used to detect chaotic regions in orbital dynamics.
    
    Parameters:
    - r: A 2D numpy array of shape (n_states, 3) representing the initial positions of the orbital states 
         in 3D space (x, y, z).
    
    Returns:
    - A float representing the mean MEGNO value for the given orbital states.
    """
    n_states = len(r)
    perturbed_states = r + 1e-8 * np.random.randn(n_states, 3)
    delta_states = perturbed_states - r
    delta_states_norm = np.linalg.norm(delta_states, axis=1)
    ln_delta_states_norm = np.log(delta_states_norm)

    megno_values = np.zeros(n_states)

    for i in range(1, n_states):
        m = np.mean(ln_delta_states_norm[:i])
        megno = (ln_delta_states_norm[i] + 2 * m) / (i)
        megno_values[i] = megno

    return np.mean(megno_values)


def orbital_period(a: Union[float, np.ndarray], mu_barycenter: float = 3.986004418e14) -> Union[float, np.ndarray]:
    """
    Calculate the orbital period from the semi-major axis (a) using Kepler's third law.
    
    This function computes the orbital period for a satellite orbiting a central body, based on the 
    semi-major axis of the orbit and the gravitational parameter of the body (default is Earth).
    
    Parameters:
    - a: A float or numpy array representing the semi-major axis (in meters) of the orbit.
    - mu_barycenter: The gravitational parameter of the central body (default is Earth's gravitational 
      parameter in m^3/s^2).
    
    Returns:
    - A float or numpy array representing the orbital period(s) in days.
    """
    period_seconds = np.sqrt(4 * np.pi**2 / mu_barycenter * a**3) / 86400  # Convert seconds to days
    return period_seconds
