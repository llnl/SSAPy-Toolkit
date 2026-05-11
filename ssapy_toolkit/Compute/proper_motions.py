import numpy as np


def proper_motion(x: np.ndarray, y: np.ndarray, z: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                  xe: float = 0, ye: float = 0, ze: float = 0, vxe: float = 0, vye: float = 0, vze: float = 0,
                  input_unit: str = 'si') -> float:
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


def proper_motion_ra_dec(
    r: np.ndarray = None,
    v: np.ndarray = None,
    x: float = None,
    y: float = None,
    z: float = None,
    vx: float = None,
    vy: float = None,
    vz: float = None,
    r_earth: np.ndarray = np.array([0, 0, 0]),
    v_earth: np.ndarray = np.array([0, 0, 0]),
    input_unit: str = 'si'
) -> np.ndarray:
    """
    Calculate the proper motion in right ascension (RA) and declination (DEC) for a given position and velocity in 3D space.

    Parameters:
    - r (np.ndarray): 3D position vector (x, y, z) in SI units (m). Default is None.
    - v (np.ndarray): 3D velocity vector (vx, vy, vz) in SI units (m/s). Default is None.
    - x, y, z (float): Individual coordinates for position (in meters). These are optional if r is provided.
    - vx, vy, vz (float): Individual velocities (in m/s). These are optional if v is provided.
    - r_earth (np.ndarray): 3D position vector of Earth (default is [0, 0, 0]).
    - v_earth (np.ndarray): 3D velocity vector of Earth (default is [0, 0, 0]).
    - input_unit (str): The units for the output. Options are 'si' (SI units) or 'rebound' (rebound units). Default is 'si'.

    Returns:
    - Tuple of proper motion in right ascension and declination (in arcseconds per second) if input_unit is 'si',
      or in rebound units if input_unit is 'rebound'.
    """
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    # Distances to Earth
    d_earth_mag = np.linalg.norm(r, axis=1)

    # RA / DEC calculation
    ra = np.arctan2(r[:, 1], r[:, 0])  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    ra_unit_vector = np.array([-np.sin(ra), np.cos(ra), np.zeros_like(ra)]).T
    dec_unit_vector = -np.array([np.cos(np.pi / 2 - dec) * np.cos(ra), np.cos(np.pi / 2 - dec) * np.sin(ra), -np.sin(np.pi / 2 - dec)]).T
    pmra = (np.einsum('ij,ij->i', v, ra_unit_vector)) / d_earth_mag * 206265  # arcseconds / second
    pmdec = (np.einsum('ij,ij->i', v, dec_unit_vector)) / d_earth_mag * 206265  # arcseconds / second

    if input_unit == 'si':
        return pmra, pmdec
    elif input_unit == 'rebound':
        pmra = pmra / (31557600 * 2 * np.pi)
        pmdec = pmdec / (31557600 * 2 * np.pi)  # arcseconds * (au/sim_time)/au, convert to arcseconds / second
        return pmra, pmdec
    else:
        print('Error - units provided not available, provide either SI or rebound units.')
        return
