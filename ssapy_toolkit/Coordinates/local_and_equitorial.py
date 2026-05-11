import numpy as np
from .sky_angles import zenithangle2altitude
from ..Time_Functions import hms_to_dd, dd_to_hms, dd_to_dms


def rightasension2hourangle(right_ascension, local_time):
    """
    Convert right ascension and local time to hour angle.

    Parameters:
    - right_ascension (str or float): The right ascension of the object in HH:MM:SS format or decimal degrees.
    - local_time (str or float): The local time in HH:MM:SS format or decimal hours.

    Returns:
    - str: The corresponding hour angle in HH:MM:SS format.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if not isinstance(right_ascension, str):
        right_ascension = dd_to_hms(right_ascension)
    if not isinstance(local_time, str):
        local_time = dd_to_dms(local_time)

    _ra = float(right_ascension.split(':')[0])
    _lt = float(local_time.split(':')[0])

    if _ra > _lt:
        __ltm, __lts = local_time.split(':')[1:]
        local_time = f'{24 + _lt}:{__ltm}:{__lts}'

    return dd_to_dms(hms_to_dd(local_time) - hms_to_dd(right_ascension))


def equatorial_to_horizontal(
    observer_latitude,
    declination,
    right_ascension=None,
    hour_angle=None,
    local_time=None,
    hms=False
):
    """
    Convert equatorial coordinates (right ascension, declination) to horizontal coordinates (azimuth, altitude).

    Parameters:
    - observer_latitude (float): Latitude of the observer in degrees.
    - declination (float): Declination of the object in degrees.
    - right_ascension (str or float, optional): Right ascension in HH:MM:SS format or decimal degrees.
    - hour_angle (str or float, optional): Hour angle in HH:MM:SS format or decimal degrees.
    - local_time (str or float, optional): Local time in HH:MM:SS format or decimal hours.
    - hms (bool): If True, interpret inputs as HH:MM:SS strings.

    Returns:
    - (float, float): Azimuth and altitude in degrees.

    Author: Travis Yeager (yeager7@llnl.gov)
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

    observer_latitude, hour_angle_rad, declination = np.radians(
        [observer_latitude, hour_angle_dd, declination]
    )

    zenith_angle = np.arccos(
        np.sin(observer_latitude) * np.sin(declination) +
        np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle_rad)
    )

    altitude = zenithangle2altitude(zenith_angle, deg=False)

    _num = np.sin(declination) - np.sin(observer_latitude) * np.cos(zenith_angle)
    _den = np.cos(observer_latitude) * np.sin(zenith_angle)
    azimuth = np.arccos(_num / _den)

    if observer_latitude < 0:
        azimuth = np.pi - azimuth
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude


def horizontal_to_equatorial(observer_latitude, azimuth, altitude):
    """
    Convert horizontal coordinates (azimuth, altitude) to equatorial coordinates (hour angle, declination).

    Parameters:
    - observer_latitude (float): Latitude of the observer in degrees.
    - azimuth (float): Azimuth in degrees.
    - altitude (float): Altitude in degrees.

    Returns:
    - (float, float): Hour angle and declination in degrees.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    altitude_rad, azimuth_rad, latitude_rad = np.radians([altitude, azimuth, observer_latitude])

    zenith_angle_rad = np.pi / 2 - altitude_rad

    declination_rad = np.arcsin(
        np.sin(latitude_rad) * np.cos(zenith_angle_rad) +
        np.cos(latitude_rad) * np.sin(zenith_angle_rad) * np.cos(azimuth_rad)
    )

    cos_hour_angle = (
        (np.cos(zenith_angle_rad) - np.sin(latitude_rad) * np.sin(declination_rad)) /
        (np.cos(latitude_rad) * np.cos(declination_rad))
    )

    hour_angle_rad = np.arccos(np.clip(cos_hour_angle, -1, 1))

    if azimuth_rad > np.pi:  # 3rd or 4th quadrant
        hour_angle_rad = 2 * np.pi - hour_angle_rad

    declination, hour_angle = np.degrees([declination_rad, hour_angle_rad])

    return hour_angle, declination
