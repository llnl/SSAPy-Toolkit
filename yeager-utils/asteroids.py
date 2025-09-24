import numpy as np
from random import shuffle
from typing import Tuple, List, Dict


def radius_from_H_albedo(H: np.ndarray, albedo: float = 0.1) -> np.ndarray:
    """
    Calculate asteroid radius from H magnitude and albedo.

    Parameters
    ----------
    H : np.ndarray
        H magnitude values of the asteroid.
    albedo : float, optional
        Albedo value of the asteroid (default is 0.1).

    Returns
    -------
    np.ndarray
        Calculated radius for the asteroid.
    """
    radius = 1329e3 / (2 * np.sqrt(albedo)) * 10 ** (-0.2 * H)  # http://www.physics.sfasu.edu/astro/asteroids/sizemagnitude.html
    return radius


def H_mag(radius: np.ndarray, albedo: float) -> np.ndarray:
    """
    Calculate the H magnitude from the asteroid radius and albedo.

    Parameters
    ----------
    radius : np.ndarray
        Radius values of the asteroid.
    albedo : float
        Albedo value of the asteroid.

    Returns
    -------
    np.ndarray
        H magnitude values of the asteroid.
    """
    return 5 * np.log(664500 / (radius * np.sqrt(albedo))) / np.log(10)


def johnsonV_to_lsst_array(M_app: np.ndarray, filters: List[str], ast_types: np.ndarray) -> np.ndarray:
    """
    Convert apparent magnitude in Johnson V to LSST filters.

    Parameters
    ----------
    M_app : np.ndarray
        Apparent magnitude values.
    filters : List[str]
        List of filter names ('u', 'g', 'r', 'i', 'z', 'y').
    ast_types : np.ndarray
        Array of asteroid types (0 or 1).

    Returns
    -------
    np.ndarray
        Corrected apparent magnitudes.
    """
    corrections = np.zeros(np.shape(M_app))
    corrections[np.where((np.array(filters) == 'u') & (np.array(ast_types) == 0))] = -1.614
    corrections[np.where((np.array(filters) == 'u') & (np.array(ast_types) == 1))] = -1.927
    corrections[np.where((np.array(filters) == 'g') & (np.array(ast_types) == 0))] = -0.302
    corrections[np.where((np.array(filters) == 'g') & (np.array(ast_types) == 1))] = -0.395
    corrections[np.where((np.array(filters) == 'r') & (np.array(ast_types) == 0))] = 0.172
    corrections[np.where((np.array(filters) == 'r') & (np.array(ast_types) == 1))] = 0.255
    corrections[np.where((np.array(filters) == 'i') & (np.array(ast_types) == 0))] = 0.291
    corrections[np.where((np.array(filters) == 'i') & (np.array(ast_types) == 1))] = 0.455
    corrections[np.where((np.array(filters) == 'z') & (np.array(ast_types) == 0))] = 0.298
    corrections[np.where((np.array(filters) == 'z') & (np.array(ast_types) == 1))] = 0.401
    corrections[np.where((np.array(filters) == 'y') & (np.array(ast_types) == 0))] = 0.303
    corrections[np.where((np.array(filters) == 'y') & (np.array(ast_types) == 1))] = 0.406
    return M_app - corrections


def johnsonV_to_ztf_array(M_app: np.ndarray, filters: List[int], ast_types: np.ndarray) -> np.ndarray:
    """
    Convert apparent magnitude in Johnson V to ZTF filters.

    Parameters
    ----------
    M_app : np.ndarray
        Apparent magnitude values.
    filters : List[int]
        List of filter indices (1: 'g', 2: 'r', 3: 'i').
    ast_types : np.ndarray
        Array of asteroid types (0 or 1).

    Returns
    -------
    np.ndarray
        Corrected apparent magnitudes.
    """
    corrections = np.zeros(np.shape(M_app))
    corrections[np.where((np.array(filters) == 1) & (np.array(ast_types) == 0))] = -0.302
    corrections[np.where((np.array(filters) == 1) & (np.array(ast_types) == 1))] = -0.395
    corrections[np.where((np.array(filters) == 2) & (np.array(ast_types) == 0))] = 0.172
    corrections[np.where((np.array(filters) == 2) & (np.array(ast_types) == 1))] = 0.255
    corrections[np.where((np.array(filters) == 3) & (np.array(ast_types) == 0))] = 0.291
    corrections[np.where((np.array(filters) == 3) & (np.array(ast_types) == 1))] = 0.455
    return M_app - corrections


def get_albedo_array(num: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random array of albedo values and asteroid types.

    Parameters
    ----------
    num : int, optional
        Number of samples to generate (default is 1).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Array of albedo values and asteroid types (0 or 1).
    """
    num = int(num)
    albedo_out = []
    ast_type_out = []
    while np.size(albedo_out) != num:
        fd = 0.253
        d = 0.030
        b = 0.168
        albedo = np.random.uniform(0, 1, size=num)
        sample_ys = np.random.uniform(0, 6, size=num)
        c_type = fd * (albedo * np.exp(-albedo**2 / (2 * d**2)) / d**2)
        s_type = (1 - fd) * (albedo * np.exp(-albedo**2 / (2 * b**2)) / b**2)
        c_albedo = albedo[np.where(sample_ys < c_type)]
        s_albedo = albedo[np.where(sample_ys < s_type)]
        c_type = np.zeros(np.size(c_albedo))
        s_type = np.ones(np.size(s_albedo))
        albedo = np.concatenate((c_albedo, s_albedo))
        ast_type = np.concatenate((c_type, s_type))
        if np.size(albedo_out) == 0:
            albedo_out = albedo
            ast_type_out = ast_type
            continue
        albedo_out = np.hstack((albedo_out, albedo))
        ast_type_out = np.hstack((ast_type_out, ast_type))
        if np.size(albedo_out) > num:
            temp = list(zip(albedo, ast_type))
            np.random.shuffle(temp)
            albedo, ast_type = zip(*temp)
            albedo_out = albedo_out[:num]
            ast_type_out = ast_type_out[:num]
    return albedo_out, ast_type_out


def granvik_low_slope(x: np.ndarray) -> np.ndarray:
    """
    Low-slope function for Granvik distribution.

    Parameters
    ----------
    x : np.ndarray
        Input values (e.g., H magnitudes).

    Returns
    -------
    np.ndarray
        Output values after applying the low-slope function.
    """
    return 0.3034 * x - 3.491


def granvik_high_slope(x: np.ndarray) -> np.ndarray:
    """
    High-slope function for Granvik distribution.

    Parameters
    ----------
    x : np.ndarray
        Input values (e.g., H magnitudes).

    Returns
    -------
    np.ndarray
        Output values after applying the high-slope function.
    """
    return 0.7235 * x - 13.12


def get_neo_H_mag_array(num: int = 1, upper_mag: float = 28, min_mag: float = 10) -> np.ndarray:
    """
    Generate a random array of H magnitudes for Near-Earth Objects (NEOs).

    Parameters
    ----------
    num : int, optional
        Number of samples to generate (default is 1).
    upper_mag : float, optional
        Upper limit for the magnitude (default is 28).
    min_mag : float, optional
        Lower limit for the magnitude (default is 10).

    Returns
    -------
    np.ndarray
        Array of H magnitudes.
    """
    num = int(num)
    H_mag_out = []
    while np.size(H_mag_out) != num:
        xs = np.random.uniform(min_mag, upper_mag, size=num)
        ys = np.random.uniform(1, 10**granvik_high_slope(upper_mag), size=num)
        xs_low = xs[np.where(xs < 23)]
        ys_low = ys[np.where(xs < 23)]
        xs_high = xs[np.where(xs >= 23)]
        ys_high = ys[np.where(xs >= 23)]
        index_low = np.where(ys_low < 10**granvik_low_slope(xs_low))
        index_high = np.where(ys_high < 10**granvik_high_slope(xs_high))
        H_mag = np.hstack((xs_low[index_low], xs_high[index_high]))
        if np.size(H_mag_out) == 0:
            H_mag_out = H_mag
            continue
        H_mag_out = np.hstack((H_mag_out, H_mag))
        if np.size(H_mag_out) > num:
            H_mag_out = H_mag_out[:num]
    shuffle(H_mag_out)
    return H_mag_out


def get_eta_radius_albedo_H_array(num: int = 1, upper_mag: float = 28, min_mag: float = 10) -> Dict[str, np.ndarray]:
    """
    Generate radius, albedo, asteroid type, and H magnitude arrays for ETA dataset.

    Parameters
    ----------
    num : int, optional
        Number of samples to generate (default is 1).
    upper_mag : float, optional
        Upper limit for the magnitude (default is 28).
    min_mag : float, optional
        Lower limit for the magnitude (default is 10).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with radius, albedo, asteroid type, and H magnitude arrays.
    """
    albedo, ast_type = get_albedo_array(num=num)
    H = get_neo_H_mag_array(num=num, upper_mag=upper_mag, min_mag=min_mag)
    radius = 1329e3 / (2 * np.sqrt(albedo)) * 10**(-0.2 * H)
    return {'radius': radius, 'albedo': albedo, 'type': ast_type, 'H': H}
