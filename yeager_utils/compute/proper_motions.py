import numpy as np
from typing import Union


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

    Author: Travis Yeager (yaeger7@llnl.gov)
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
