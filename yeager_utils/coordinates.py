# flake8: noqa: E501

from .constants import EARTH_RADIUS, WGS84_EARTH_OMEGA
from .time import hms_to_dd, dd_to_hms, dd_to_dms
from ssapy.accel import AccelKepler
from ssapy.body import get_body, MoonPosition
from ssapy.compute import groundTrack, rv
from ssapy.constants import RGEO
from ssapy.orbit import Orbit
from ssapy.propagator import RK78Propagator
from .vectors import normed
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle
from typing import Union, List, Tuple, Optional


def dms_to_rad(coords: Union[str, List[str], Tuple[str, ...]]) -> Union[float, List[float]]:
    """
    Convert degrees, minutes, and seconds (DMS) to radians.

    Parameters:
    - coords (Union[str, List[str], Tuple[str, ...]]): A single DMS value as a string or a list/tuple of DMS strings to convert. 
      The DMS string should follow the format recognizable by Astropy's Angle class (e.g., '30d30m30s', '45d').

    Returns:
    - Union[float, List[float]]: Converted angle(s) in radians. Returns a single float for a single DMS input, 
      or a list of floats for a list/tuple of DMS inputs.
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
    """
    return angles % (2 * np.pi)


def deg0to360(array_: Union[np.ndarray, List[float]]) -> Union[np.ndarray, List[float]]:
    """
    Normalize angles to the range [0, 360).

    Parameters:
    - array_ (Union[np.ndarray, List[float]]): An array or list of angles in degrees.

    Returns:
    - Union[np.ndarray, List[float]]: Angles normalized to the range [0, 360).
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
    """
    return [i % 360 for i in array_]


def deg90to90(val_in: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    Normalize angles to the range [-90, 90].

    Parameters:
    - val_in (Union[float, List[float]]): An angle or list of angles in degrees.

    Returns:
    - Union[float, List[float]]: Angles normalized to the range [-90, 90].
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
    """
    return [i % 90 for i in array_]


def cart2sph_deg(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, radius) in degrees.

    Parameters:
    - x (float): The x-coordinate in Cartesian space.
    - y (float): The y-coordinate in Cartesian space.
    - z (float): The z-coordinate in Cartesian space.

    Returns:
    - Tuple[float, float, float]: Azimuth, elevation, and radius in degrees.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) * (180 / np.pi)
    az = (np.arctan2(y, x)) * (180 / np.pi)
    return az, el, r


def cart_to_cyl(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (radius, angle, z).

    Parameters:
    - x (float): The x-coordinate in Cartesian space.
    - y (float): The y-coordinate in Cartesian space.
    - z (float): The z-coordinate in Cartesian space.

    Returns:
    - Tuple[float, float, float]: Radius, angle, and z in cylindrical coordinates.
    """
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta, z


def inert2rot(x: float, y: float, xe: float, ye: float, xs: float = 0, ys: float = 0) -> Tuple[float, float]:
    """
    Rotate inertial coordinates to a rotated frame defined by the position of Earth.

    Parameters:
    - x (float): The x-coordinate in inertial space.
    - y (float): The y-coordinate in inertial space.
    - xe (float): The x-coordinate of Earth's position.
    - ye (float): The y-coordinate of Earth's position.
    - xs (float, optional): The x-coordinate of the Sun's position. Defaults to 0.
    - ys (float, optional): The y-coordinate of the Sun's position. Defaults to 0.

    Returns:
    - Tuple[float, float]: The rotated x and y coordinates.
    """
    earth_theta = np.arctan2(ye - ys, xe - xs)
    theta = np.arctan2(y - ys, x - xs)
    distance = np.sqrt(np.power((x - xs), 2) + np.power((y - ys), 2))
    xrot = distance * np.cos(np.pi + (theta - earth_theta))
    yrot = distance * np.sin(np.pi + (theta - earth_theta))
    return xrot, yrot


def sim_lonlatrad(x: float, y: float, z: float, xe: float, ye: float, ze: float, xs: float, ys: float, zs: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x, y, z) to geodetic longitude, latitude, and radius, 
    relative to the Sun's position at the given time.

    Parameters:
    - x, y, z (float): Cartesian coordinates of the point of interest.
    - xe, ye, ze (float): Cartesian coordinates of the observer (Earth).
    - xs, ys, zs (float): Cartesian coordinates of the Sun.

    Returns:
    - Tuple[float, float, float]: Longitude, latitude, and radius relative to the Sun in degrees and kilometers.
    """
    # Convert all to geo coordinates
    x = x - xe
    y = y - ye
    z = z - ze
    xs = xs - xe
    ys = ys - ye
    zs = zs - ze
    # Convert x, y, z to lon, lat, radius
    longitude, latitude, radius = cart2sph_deg(x, y, z)
    slongitude, slatitude, sradius = cart2sph_deg(xs, ys, zs)
    # Correct so that Sun is at (0,0)
    longitude = deg0to360(slongitude - longitude)
    latitude = latitude - slatitude
    return longitude, latitude, radius


def sun_ra_dec(time_: Union[int, float, str]) -> Tuple[float, float]:
    """
    Calculate the Right Ascension and Declination of the Sun for a given time.

    Parameters:
    - time_ (Union[int, float, str]): The time for which to calculate the Sun's position (in MJD or string format).

    Returns:
    - Tuple[float, float]: Right Ascension and Declination of the Sun in radians.
    """
    out = get_body(Time(time_, format='mjd'))
    return out.ra.to('rad').value, out.dec.to('rad').value


def ra_dec(
    r: np.ndarray = None, v: np.ndarray = None,
    x: float = None, y: float = None, z: float = None,
    vx: float = None, vy: float = None, vz: float = None,
    r_earth: np.ndarray = np.array([0, 0, 0]),
    v_earth: np.ndarray = np.array([0, 0, 0]),
    input_unit: str = 'si'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Right Ascension (RA) and Declination (Dec) of a position and velocity vector,
    relative to Earth's position and velocity.

    Parameters:
    - r (np.ndarray): Position vector of the object in 3D space.
    - v (np.ndarray): Velocity vector of the object.
    - x, y, z (float, optional): Individual Cartesian coordinates of the object.
    - vx, vy, vz (float, optional): Individual velocity components of the object.
    - r_earth (np.ndarray, optional): Earth's position vector. Default is the origin.
    - v_earth (np.ndarray, optional): Earth's velocity vector. Default is zero velocity.
    - input_unit (str, optional): The unit system used. Defaults to SI units.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: RA and Dec in radians.
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

    d_earth_mag = np.linalg.norm(r, axis=1)
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    return ra, dec


def lonlat_distance(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.

    Parameters:
    - lat1, lat2 (float): Latitude of the two points in radians.
    - lon1, lon2 (float): Longitude of the two points in radians.

    Returns:
    - float: Distance between the two points in kilometers.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of Earth in kilometers
    return c * EARTH_RADIUS


def altitude2zenithangle(altitude: float, deg: bool = True) -> float:
    """
    Convert altitude to zenith angle.

    Parameters:
    - altitude (float): The altitude of the object in degrees or radians.
    - deg (bool, optional): If True, the output is in degrees. If False, in radians. Default is True.

    Returns:
    - float: Zenith angle in degrees or radians.
    """
    if deg:
        return 90 - altitude
    else:
        return np.pi / 2 - altitude


def zenithangle2altitude(zenith_angle: float, deg: bool = True) -> float:
    """
    Convert zenith angle to altitude.

    Parameters:
    - zenith_angle (float): The zenith angle in degrees or radians.
    - deg (bool, optional): If True, the input is in degrees. If False, in radians. Default is True.

    Returns:
    - float: Altitude in degrees or radians.
    """
    if deg:
        return 90 - zenith_angle
    else:
        return np.pi / 2 - zenith_angle


def rightasension2hourangle(right_ascension: Union[str, float], local_time: Union[str, float]) -> str:
    """
    Convert right ascension and local time to hour angle.

    Parameters:
    - right_ascension (Union[str, float]): The right ascension of the object in HH:MM:SS format or decimal degrees.
    - local_time (Union[str, float]): The local time in HH:MM:SS format or decimal hours.

    Returns:
    - str: The corresponding hour angle in HH:MM:SS format.
    """
    if type(right_ascension) is not str:
        right_ascension = dd_to_hms(right_ascension)
    if type(local_time) is not str:
        local_time = dd_to_dms(local_time)
    
    _ra = float(right_ascension.split(':')[0])
    _lt = float(local_time.split(':')[0])
    
    if _ra > _lt:
        __ltm, __lts = local_time.split(':')[1:]
        local_time = f'{24 + _lt}:{__ltm}:{__lts}'

    return dd_to_dms(hms_to_dd(local_time) - hms_to_dd(right_ascension))


def equatorial_to_horizontal(
    observer_latitude: float, declination: float,
    right_ascension: Union[str, float] = None, hour_angle: Union[str, float] = None,
    local_time: Union[str, float] = None
) -> Tuple[float, float]:
    """
    Convert equatorial coordinates (right ascension, declination) to horizontal coordinates (azimuth, altitude).

    Parameters:
    - observer_latitude (float): Latitude of the observer in degrees.
    - declination (float): Declination of the object in degrees.
    - right_ascension (Union[str, float], optional): Right ascension in HH:MM:SS format or decimal degrees.
    - hour_angle (Union[str, float], optional): Hour angle in HH:MM:SS format or decimal degrees.
    - local_time (Union[str, float], optional): Local time in HH:MM:SS format or decimal hours.

    Returns:
    - Tuple[float, float]: Azimuth and altitude in degrees.
    """
    if right_ascension is not None:
        hour_angle_dd = rightasension2hourangle(right_ascension, local_time)
        hour_angle_dd = hms_to_dd(hour_angle_dd)
    elif hour_angle is not None:
        if isinstance(hour_angle, str):
            hour_angle_dd = hms_to_dd(hour_angle)
        else:
            hour_angle_dd = hour_angle
    elif right_ascension is not None and hour_angle is not None:
        print('Both right_ascension and hour_angle parameters are provided.\nUsing hour_angle for calculations.')
        if hms:
            hour_angle_dd = hms_to_dd(hour_angle)
    else:
        print('Either right_ascension or hour_angle must be provided.')

    observer_latitude, hour_angle_rad, declination = np.radians([observer_latitude, hour_angle_dd, declination])

    zenith_angle = np.arccos(np.sin(observer_latitude) * np.sin(declination) + np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle_rad))

    altitude = zenithangle2altitude(zenith_angle, deg=False)

    _num = np.sin(declination) - np.sin(observer_latitude) * np.cos(zenith_angle)
    _den = np.cos(observer_latitude) * np.sin(zenith_angle)
    azimuth = np.arccos(_num / _den)

    if observer_latitude < 0:
        azimuth = np.pi - azimuth
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude


def horizontal_to_equatorial(observer_latitude: float, azimuth: float, altitude: float) -> Tuple[float, float]:
    """
    Convert horizontal coordinates (azimuth, altitude) to equatorial coordinates (right ascension, declination).

    Parameters:
    - observer_latitude (float): Latitude of the observer in degrees.
    - azimuth (float): Azimuth in degrees.
    - altitude (float): Altitude in degrees.

    Returns:
    - Tuple[float, float]: Hour angle and declination in degrees.
    """
    # Convert inputs to radians
    altitude_rad, azimuth_rad, latitude_rad = np.radians([altitude, azimuth, observer_latitude])
    
    # Calculate zenith angle
    zenith_angle_rad = np.pi / 2 - altitude_rad
    
    # Calculate declination
    declination_rad = np.arcsin(
        np.sin(latitude_rad) * np.cos(zenith_angle_rad) +
        np.cos(latitude_rad) * np.sin(zenith_angle_rad) * np.cos(azimuth_rad)
    )
    
    # Calculate hour angle
    cos_hour_angle = (np.cos(zenith_angle_rad) - np.sin(latitude_rad) * np.sin(declination_rad)) / \
                     (np.cos(latitude_rad) * np.cos(declination_rad))
    
    # Ensure the hour angle is in the range [0, 2π)
    hour_angle_rad = np.arccos(np.clip(cos_hour_angle, -1, 1))
    
    # Adjust hour angle based on azimuth quadrant
    if azimuth_rad > np.pi:  # Azimuth in 3rd or 4th quadrant
        hour_angle_rad = 2 * np.pi - hour_angle_rad

    # Convert results back to degrees
    declination, hour_angle = np.degrees([declination_rad, hour_angle_rad])
    
    return hour_angle, declination



# Constants for ecliptic inclination and sine/cosine of it
_ecliptic = 0.409092601  # np.radians(23.43927944)
cos_ec = 0.9174821430960974
sin_ec = 0.3977769690414367


def equatorial_xyz_to_ecliptic_xyz(xq: float, yq: float, zq: float) -> Tuple[float, float, float]:
    """
    Convert equatorial Cartesian coordinates to ecliptic Cartesian coordinates.

    Parameters:
    - xq, yq, zq (float): Equatorial Cartesian coordinates.

    Returns:
    - Tuple[float, float, float]: Ecliptic Cartesian coordinates (xc, yc, zc).
    """
    xc = xq
    yc = cos_ec * yq + sin_ec * zq
    zc = -sin_ec * yq + cos_ec * zq
    return xc, yc, zc


def ecliptic_xyz_to_equatorial_xyz(xc: float, yc: float, zc: float) -> Tuple[float, float, float]:
    """
    Convert ecliptic Cartesian coordinates to equatorial Cartesian coordinates.

    Parameters:
    - xc, yc, zc (float): Ecliptic Cartesian coordinates.

    Returns:
    - Tuple[float, float, float]: Equatorial Cartesian coordinates (xq, yq, zq).
    """
    xq = xc
    yq = cos_ec * yc - sin_ec * zc
    zq = sin_ec * yc + cos_ec * zc
    return xq, yq, zq


def xyz_to_ecliptic(xc: float, yc: float, zc: float, xe: float = 0, ye: float = 0, ze: float = 0, degrees: bool = False) -> Tuple[float, float]:
    """
    Convert a point in space from equatorial coordinates to ecliptic latitude and longitude.

    Parameters:
    - xc, yc, zc (float): Cartesian coordinates of the point in space.
    - xe, ye, ze (float, optional): Cartesian coordinates of the Earth. Default is (0, 0, 0).
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - Tuple[float, float]: Ecliptic longitude and latitude (in radians or degrees).
    """
    x_ast_to_earth = xc - xe
    y_ast_to_earth = yc - ye
    z_ast_to_earth = zc - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ec_longitude = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    ec_latitude = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ec_longitude), np.degrees(ec_latitude)
    else:
        return ec_longitude, ec_latitude


def xyz_to_equatorial(xq: float, yq: float, zq: float, xe: float = 0, ye: float = 0, ze: float = 0, degrees: bool = False) -> Tuple[float, float]:
    """
    Convert a point in space from Cartesian coordinates to equatorial right ascension and declination.

    Parameters:
    - xq, yq, zq (float): Cartesian coordinates of the point.
    - xe, ye, ze (float, optional): Cartesian coordinates of the Earth. Default is (0, 0, 0).
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - Tuple[float, float]: Right ascension and declination (in radians or degrees).
    """
    # RA / DEC calculation - assumes XY plane to be celestial equator, and -x axis to be vernal equinox
    x_ast_to_earth = xq - xe
    y_ast_to_earth = yq - ye
    z_ast_to_earth = zq - ze
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def ecliptic_xyz_to_equatorial(xc: float, yc: float, zc: float, xe: float = 0, ye: float = 0, ze: float = 0, degrees: bool = False) -> Tuple[float, float]:
    """
    Convert ecliptic Cartesian coordinates to equatorial right ascension and declination.

    Parameters:
    - xc, yc, zc (float): Ecliptic Cartesian coordinates of the point.
    - xe, ye, ze (float, optional): Cartesian coordinates of the Earth. Default is (0, 0, 0).
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - Tuple[float, float]: Right ascension and declination (in radians or degrees).
    """
    x_ast_to_earth, y_ast_to_earth, z_ast_to_earth = ecliptic_xyz_to_equatorial_xyz(xc - xe, yc - ye, zc - ze)
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def equatorial_to_ecliptic(right_ascension: float, declination: float, degrees: bool = False) -> Tuple[float, float]:
    """
    Convert equatorial coordinates (right ascension and declination) to ecliptic coordinates (longitude and latitude).

    Parameters:
    - right_ascension (float): Right ascension in radians or degrees.
    - declination (float): Declination in radians or degrees.
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - Tuple[float, float]: Ecliptic longitude and latitude (in radians or degrees).
    """
    ra, dec = np.radians(right_ascension), np.radians(declination)
    ec_latitude = np.arcsin(cos_ec * np.sin(dec) - sin_ec * np.cos(dec) * np.sin(ra))
    ec_longitude = np.arctan((cos_ec * np.cos(dec) * np.sin(ra) + sin_ec * np.sin(dec)) / (np.cos(dec) * np.cos(ra)))
    if degrees:
        return deg0to360(np.degrees(ec_longitude)), np.degrees(ec_latitude)
    else:
        return rad0to2pi(ec_longitude), ec_latitude


def ecliptic_to_equatorial(lon: float, lat: float, degrees: bool = False) -> Tuple[float, float]:
    """
    Convert ecliptic coordinates (longitude and latitude) to equatorial coordinates (right ascension and declination).

    Parameters:
    - lon (float): Ecliptic longitude in radians or degrees.
    - lat (float): Ecliptic latitude in radians or degrees.
    - degrees (bool, optional): If True, return the results in degrees. Default is False (radians).

    Returns:
    - Tuple[float, float]: Right ascension and declination (in radians or degrees).
    """
    lon, lat = np.radians(lon), np.radians(lat)
    ra = np.arctan((cos_ec * np.cos(lat) * np.sin(lon) - sin_ec * np.sin(lat)) / (np.cos(lat) * np.cos(lon)))
    dec = np.arcsin(cos_ec * np.sin(lat) + sin_ec * np.cos(lat) * np.sin(lon))
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def proper_motion_ra_dec(
    r: Optional[np.ndarray] = None, 
    v: Optional[np.ndarray] = None, 
    x: Optional[float] = None, 
    y: Optional[float] = None, 
    z: Optional[float] = None, 
    vx: Optional[float] = None, 
    vy: Optional[float] = None, 
    vz: Optional[float] = None, 
    r_earth: np.ndarray = np.array([0, 0, 0]), 
    v_earth: np.ndarray = np.array([0, 0, 0]), 
    input_unit: str = 'si'
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
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


def gcrf_to_lunar(r: np.ndarray, t: np.ndarray, v: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert position and velocity vectors from GCRF to Lunar coordinates.

    Parameters:
    - r (np.ndarray): Position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (Optional[np.ndarray]): Velocity vector in GCRF coordinates (meters per second). Optional.

    Returns:
    - Position vector in Lunar coordinates if velocity is not provided, or position and velocity in Lunar coordinates.
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


def gcrf_to_radec(gcrf_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GCRF Cartesian coordinates to right ascension and declination.

    Parameters:
    - gcrf_coords (np.ndarray): 3D position vector in GCRF coordinates (x, y, z).

    Returns:
    - Tuple of right ascension and declination in degrees.
    """
    x, y, z = gcrf_coords
    ra = np.arctan2(y, x)
    ra_deg = np.degrees(ra)
    ra_deg = ra_deg % 360
    dec_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    dec_deg = np.degrees(dec_rad)
    return ra_deg, dec_deg


def gcrf_to_ecef_bad(r_gcrf: np.ndarray, t: Time) -> np.ndarray:
    """
    Convert GCRF coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates (approximated).

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates.
    - t (Time): Time at which the conversion is performed.

    Returns:
    - 3D position vector in ECEF coordinates (meters).
    """
    if isinstance(t, Time):
        t = t.gps
    r_gcrf = np.atleast_2d(r_gcrf)
    rotation_angles = WGS84_EARTH_OMEGA * (t - Time("1980-3-20T11:06:00", format='isot').gps)
    cos_thetas = np.cos(rotation_angles)
    sin_thetas = np.sin(rotation_angles)

    Rz = np.array([[cos_thetas, -sin_thetas, np.zeros_like(cos_thetas)],
                   [sin_thetas, cos_thetas, np.zeros_like(cos_thetas)],
                   [np.zeros_like(cos_thetas), np.zeros_like(cos_thetas), np.ones_like(cos_thetas)]]).T

    r_ecef = np.einsum('ijk,ik->ij', Rz, r_gcrf)
    return r_ecef


def gcrf_to_lat_lon(r: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert GCRF coordinates to latitude, longitude, and height above Earth's surface.

    Parameters:
    - r (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.

    Returns:
    - Tuple of longitude, latitude, and height above the Earth's surface.
    """
    lon, lat, height = groundTrack(r, t)
    return lon, lat, height


def gcrf_to_itrf(r_gcrf: np.ndarray, t: np.ndarray, v: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert GCRF coordinates to ITRF coordinates.

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - v (Optional[np.ndarray]): Velocity vector in GCRF coordinates (meters per second). Optional.

    Returns:
    - Position in ITRF coordinates, or position and velocity in ITRF coordinates if velocity is provided.
    """
    x, y, z = groundTrack(r_gcrf, t, format='cartesian')
    _ = np.array([x, y, z]).T
    if v is None:
        return _
    else:
        return _, v_from_r(_, t)


def gcrf_to_sim_geo(r_gcrf: np.ndarray, t: np.ndarray, h: float = 10) -> np.ndarray:
    """
    Convert GCRF coordinates to simulated geodetic coordinates (latitude, longitude, height).

    Parameters:
    - r_gcrf (np.ndarray): 3D position vector in GCRF coordinates (meters).
    - t (np.ndarray): Time array for conversion.
    - h (float): Step size in seconds for time propagation.

    Returns:
    - Simulated geodetic coordinates in meters.
    """
    if np.min(np.diff(t.gps)) < h:
        h = np.min(np.diff(t.gps))
    r_gcrf = np.atleast_2d(r_gcrf)
    r_geo, v_geo = rv(Orbit.fromKeplerianElements(*[RGEO, 0, 0, 0, 0, 0], t=t[0]), t, propagator=RK78Propagator(AccelKepler(), h=h))
    angle_geo_to_x = np.arctan2(r_geo[:, 1], r_geo[:, 0])
    c = np.cos(angle_geo_to_x)
    s = np.sin(angle_geo_to_x)
    rotation = np.array([[c, -s, np.zeros_like(c)], [s, c, np.zeros_like(c)], [np.zeros_like(c), np.zeros_like(c), np.ones_like(c)]]).T
    return np.einsum('ijk,ik->ij', rotation, r_gcrf)


def gcrf_to_itrf_astropy(state_vectors: np.ndarray, t: Time) -> np.ndarray:
    """
    Convert GCRF state vectors to ITRF using Astropy.

    Parameters:
    - state_vectors (np.ndarray): Position and velocity vectors in GCRF coordinates (meters).
    - t (Time): Time of conversion.

    Returns:
    - Position and velocity vectors in ITRF coordinates (meters).
    """
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, SkyCoord, get_body_barycentric, solar_system_ephemeris, ICRS

    sc = SkyCoord(x=state_vectors[:, 0] * u.m, y=state_vectors[:, 1] * u.m, z=state_vectors[:, 2] * u.m, representation_type='cartesian', frame=GCRS(obstime=t))
    sc_itrs = sc.transform_to(ITRS(obstime=t))
    with solar_system_ephemeris.set('de430'):
        earth = get_body_barycentric('earth', t)
    earth_center_itrs = SkyCoord(earth.x, earth.y, earth.z, representation_type='cartesian', frame=ICRS()).transform_to(ITRS(obstime=t))
    itrs_coords = SkyCoord(
        sc_itrs.x.value - earth_center_itrs.x.to_value(u.m),
        sc_itrs.y.value - earth_center_itrs.y.to_value(u.m),
        sc_itrs.z.value - earth_center_itrs.z.to_value(u.m),
        representation_type='cartesian',
        frame=ITRS(obstime=t)
    )
    itrs_coords_meters = np.array([itrs_coords.x,
                                  itrs_coords.y,
                                  itrs_coords.z]).T
    return itrs_coords_meters


def v_from_r(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate velocity from position using numerical differentiation.

    Parameters:
    - r (np.ndarray): 3D position array (meters).
    - t (np.ndarray): Time array corresponding to the position data.

    Returns:
    - 3D velocity array (meters per second).
    """
    if isinstance(t[0], Time):
        t = t.gps
    delta_r = np.diff(r, axis=0)
    delta_t = np.diff(t)
    v = delta_r / delta_t[:, np.newaxis]
    v = np.vstack((v, v[-1]))
    return v


def get_lunar_rv(t):
    """
    Calculate the position and velocity of the Moon at a given time or times.

    Parameters:
    t (Time or array-like): A `Time` object or an array of time points for which to calculate the position and velocity. 
                            If a single time is passed, the position and velocity are calculated at that time. 
                            If multiple times are passed, the position and velocity are calculated for each time.

    Returns:
    tuple: A tuple containing two elements:
        - r (ndarray): The position of the Moon(s) at the given time(s), in kilometers.
        - v (ndarray): The velocity of the Moon(s) at the given time(s), in kilometers per second.

    Author:
    Travis Yeager (yeager7@llnl.gov)
    """
    if np.size(t) > 1:
        if isinstance(t[0], Time):
            t = t.gps
    else:
        if isinstance(t, Time):
            t = t.gps
    r = get_body("moon").position(t).T
    if np.size(t) > 1:
        v = v_from_r(r, t)
    else:
        v = (r - get_body("moon").position(t + 1).T) / 2
    return np.atleast_2d(r), np.atleast_2d(v)