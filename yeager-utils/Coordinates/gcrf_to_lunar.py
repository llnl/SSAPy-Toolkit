import numpy as np
from typing import Tuple, Optional, Union
from ssapy.body import MoonPosition, get_body
from ssapy.utils import normed
from ..Time_Functions import Time
from .v_from_r import v_from_r


def gcrf_to_lunar(r: np.ndarray, t: np.ndarray, v: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert position and velocity vectors from GCRF to Lunar coordinates.

    Parameters:
    - r (np.ndarray): Position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (Optional[np.ndarray]): Velocity vector in GCRF coordinates (meters per second). Optional.

    Returns:
    - Position vector in Lunar coordinates if velocity is not provided, or position and velocity in Lunar coordinates.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    class MoonRotator:
        def __init__(self):
            self.mpm = MoonPosition()

        def __call__(self, r: np.ndarray, t: np.ndarray) -> np.ndarray:
            if isinstance(t, Time):
                t = t.gps
            rmoon = self.mpm(t)
            vmoon = (self.mpm(t + 5.0) - self.mpm(t - 5.0)) / 10.
            xhat = normed(rmoon.T).T
            vpar = np.einsum("ab,ab->b", xhat, vmoon) * xhat
            vperp = vmoon - vpar
            yhat = normed(vperp.T).T
            zhat = np.cross(xhat, yhat, axisa=0, axisb=0).T
            R = np.empty((3, 3, len(t)))
            R[0] = xhat
            R[1] = yhat
            R[2] = zhat
            return np.einsum("abc,cb->ca", R, r)

    rotator = MoonRotator()
    if v is None:
        return rotator(r, t)
    else:
        r_lunar = rotator(r, t)
        v_lunar = v_from_r(r_lunar, t)
        return r_lunar, v_lunar


def gcrf_to_lunar_fixed(r: np.ndarray, t: np.ndarray, v: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert position and velocity vectors from GCRF to Lunar coordinates, with fixed lunar origin.

    Parameters:
    - r (np.ndarray): Position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (Optional[np.ndarray]): Velocity vector in GCRF coordinates (meters per second). Optional.

    Returns:
    - Position vector in Lunar coordinates relative to the fixed lunar origin,
      or position and velocity if velocity is provided.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # print(1, np.shape(r))
    # print(2, np.shape(t))
    # print(3, np.shape(gcrf_to_lunar(r, t)))
    # print(4, np.shape(get_body('moon').position(t).T))
    # print(5, np.shape(gcrf_to_lunar(get_body('moon').position(t).T, t)))
    r_lunar = gcrf_to_lunar(r, t) - gcrf_to_lunar(get_body('moon').position(t).T, t)
    if v is None:
        return r_lunar
    else:
        v = v_from_r(r_lunar, t)
        return r_lunar, v
