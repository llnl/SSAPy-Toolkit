from typing import Union, List, Tuple
import numpy as np
from astropy.coordinates import Angle


def dms_to_rad(coords: Union[str, List[str], Tuple[str, ...]]) -> Union[float, List[float]]:
    """
    Convert degrees, minutes, and seconds (DMS) to radians.

    Parameters:
    - coords (Union[str, List[str], Tuple[str, ...]]): A single DMS value as a string or a list/tuple of DMS strings to convert. 
      The DMS string should follow the format recognizable by Astropy's Angle class (e.g., '30d30m30s', '45d').

    Returns:
    - Union[float, List[float]]: Converted angle(s) in radians. Returns a single float for a single DMS input,
      or a list of floats for a list/tuple of DMS inputs.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(coords, (list, tuple)):
        return [Angle(coord).radian for coord in coords]
    else:
        return Angle(coords).radian


def dms_to_deg(coords: Union[str, List[str], Tuple[str, ...]]) -> Union[float, List[float]]:
    """
    Convert degrees, minutes, and seconds (DMS) to degrees.

    Parameters:
    - coords (Union[str, List[str], Tuple[str, ...]]): A single DMS value as a string or a list/tuple of DMS strings to convert. 
      The DMS string should follow the format recognizable by Astropy's Angle class (e.g., '30d30m30s', '45d').

    Returns:
    - Union[float, List[float]]: Converted angle(s) in degrees. Returns a single float for a single DMS input,
      or a list of floats for a list/tuple of DMS inputs.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(coords, (list, tuple)):
        return [Angle(coord).deg for coord in coords]
    else:
        return Angle(coords).deg


def rad0to2pi(angles: np.ndarray) -> np.ndarray:
    """
    Normalize angles to the range [0, 2*pi).

    Parameters:
    - angles (np.ndarray): An array of angles in radians.

    Returns:
    - np.ndarray: Angles normalized to the range [0, 2*pi).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return angles % (2 * np.pi)


def deg0to360(array_: Union[np.ndarray, List[float]]) -> Union[np.ndarray, List[float]]:
    """
    Normalize angles to the range [0, 360).

    Parameters:
    - array_ (Union[np.ndarray, List[float]]): An array or list of angles in degrees.

    Returns:
    - Union[np.ndarray, List[float]]: Angles normalized to the range [0, 360).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    try:
        return [i % 360 for i in array_]
    except TypeError:
        return array_ % 360


def deg0to360array(array_: List[float]) -> List[float]:
    """
    Normalize a list of angles to the range [0, 360).

    Parameters:
    - array_ (List[float]): A list of angles in degrees.

    Returns:
    - List[float]: Angles normalized to the range [0, 360).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return [i % 360 for i in array_]


def deg90to90(val_in: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    Normalize angles to the range [-90, 90].

    Parameters:
    - val_in (Union[float, List[float]]): An angle or list of angles in degrees.

    Returns:
    - Union[float, List[float]]: Angles normalized to the range [-90, 90].

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if hasattr(val_in, "__len__"):
        val_out = []
        for i, v in enumerate(val_in):
            while v < -90:
                v += 90
            while v > 90:
                v -= 90
            val_out.append(v)
    else:
        while val_in < -90:
            val_in += 90
        while val_in > 90:
            val_in -= 90
        val_out = val_in
    return val_out


def deg90to90array(array_: List[float]) -> List[float]:
    """
    Normalize a list of angles to the range [-90, 90].

    Parameters:
    - array_ (List[float]): A list of angles in degrees.

    Returns:
    - List[float]: Angles normalized to the range [-90, 90].

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return [i % 90 for i in array_]
