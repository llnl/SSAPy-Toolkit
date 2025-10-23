import numpy as np
from ..Coordinates import gcrf_to_itrf, v_from_r
from ..vectors import angle_between_vectors
from ..Time_Functions import Time


def calc_gamma(r, t):
    """
    Calculate the gamma angle between position and velocity vectors in the ITRF frame.

    Parameters
    ----------
    r : numpy.ndarray
        The position vectors in the GCRF frame, shaped (n, 3), where n is the number of vectors.
    t : numpy.ndarray or astropy.time.Time
        The times corresponding to the position vectors. Can be:
        - An array of GPS seconds (numpy.ndarray)
        - An Astropy Time object

    Returns
    -------
    numpy.ndarray
        An array of gamma angles (in degrees) between the position and velocity vectors for each time point.

    Notes
    -----
    - This function first converts the given position vectors from the GCRF (Geocentric Celestial Reference Frame)
      to the ITRF (International Terrestrial Reference Frame) using the provided time information.
    - The gamma angle is defined as the angle between the position vector and the velocity vector in the ITRF frame.
    - If the input time array is an Astropy Time object, it is converted to GPS time before processing.

    Calculation Steps
    -----------------
    1. Transform the position and velocity vectors from the GCRF frame to the ITRF frame using the time information.
    2. Compute the gamma angle as the angle between the position and velocity vectors in the ITRF frame.
    3. Return the gamma angle in degrees for each time point.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True)
    if isinstance(t[0], Time):
        t = t.gps
    gamma = np.degrees(np.apply_along_axis(lambda x: angle_between_vectors(x[:3], x[3:]), 1, np.concatenate((r_itrf, v_itrf), axis=1))) - 90
    return gamma


def calc_heading_itrf(r_itrf, v_itrf):
    """
    Calculate the heading of the flight path in the ITRF frame.

    Parameters
    ----------
    r_itrf : numpy.ndarray
        The position vectors in the ITRF frame, shaped (n, 3), where n is the number of vectors.
    v_itrf : numpy.ndarray
        The velocity vectors in the ITRF frame, shaped (n, 3), where n is the number of vectors.

    Returns
    -------
    numpy.ndarray
        An array of headings in degrees, measured clockwise from North, for each time point.

    Notes
    -----
    - The heading is calculated in the normal plane of the position vector (r_itrf).
    - The heading is measured clockwise from North, with North being aligned with the positive y-axis in the ITRF frame.
    - The calculation involves projecting the velocity vector into the plane perpendicular to the position vector (r_itrf),
      and then determining the angle between this projected vector and the North direction in the ITRF frame.

    Calculation Steps
    -----------------
    1. Normalize the position vector (r_itrf) to get the radial direction.
    2. Compute the normal plane by projecting the velocity vector (v_itrf) onto the plane perpendicular to r_itrf.
    3. Define the North direction as the positive y-axis in the ITRF frame.
    4. Calculate the heading as the angle between the projected velocity vector and the North direction, measured clockwise.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """

    # Normalize position and velocity vectors
    r_unit = r_itrf / np.linalg.norm(r_itrf, axis=1, keepdims=True)
    v_unit = v_itrf / np.linalg.norm(v_itrf, axis=1, keepdims=True)

    # Compute the flight path direction in the tangent plane of r_itrf
    v_tangential = v_unit - np.einsum('ij,ij->i', v_unit, r_unit)[:, None] * r_unit
    v_tangential /= np.linalg.norm(v_tangential, axis=1, keepdims=True)

    # Extract x (East) and y (North) components in the local tangent plane
    east = v_tangential[:, 0]
    north = v_tangential[:, 1]

    # Calculate heading in degrees clockwise from North
    heading = np.degrees(np.arctan2(east, north))
    heading = (heading + 360) % 360  # Ensure heading is in range [0, 360)

    return heading


def calc_gamma_and_heading(r, t):
    """
    Calculate both the gamma angle and the heading of the flight path in the ITRF frame.

    Parameters
    ----------
    r : numpy.ndarray
        The position vectors in the GCRF frame, shaped (n, 3), where n is the number of vectors.

    t : numpy.ndarray or astropy.time.Time
        The times corresponding to the position vectors. Can be:
        - An array of GPS seconds (numpy.ndarray)
        - An Astropy Time object

    Returns
    -------
    tuple
        - numpy.ndarray: An array of gamma angles (in degrees) between the position and velocity vectors for each time point.
        - numpy.ndarray: An array of headings (in degrees), measured clockwise from North, for each time point.

    Notes
    -----
    - This function transforms position and velocity vectors from the GCRF (Geocentric Celestial Reference Frame)
      to the ITRF (International Terrestrial Reference Frame) using the provided time information.
    - Gamma is defined as the angle between the position vector and the velocity vector, offset by 90 degrees.
    - Heading is measured clockwise from North (aligned with the positive y-axis) in the normal plane of the position vector (r_itrf).
    - If the input time array is an Astropy Time object, it is converted to GPS time before processing.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Transform position and velocity vectors from GCRF to ITRF
    r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True)

    # Convert time to GPS seconds if necessary
    if isinstance(t[0], Time):
        t = t.gps

    # Normalize position and velocity vectors
    r_unit = r_itrf / np.linalg.norm(r_itrf, axis=1, keepdims=True)
    v_unit = v_itrf / np.linalg.norm(v_itrf, axis=1, keepdims=True)

    # Calculate gamma (angle between r_itrf and v_itrf, offset by 90 degrees)
    gamma = np.degrees(np.apply_along_axis(
        lambda x: angle_between_vectors(x[:3], x[3:]),
        1,
        np.concatenate((r_itrf, v_itrf), axis=1)
    )) - 90

    # Compute the flight path direction in the tangent plane of r_itrf
    v_tangential = v_unit - np.einsum('ij,ij->i', v_unit, r_unit)[:, None] * r_unit
    v_tangential /= np.linalg.norm(v_tangential, axis=1, keepdims=True)

    # Extract x (East) and y (North) components in the local tangent plane
    east = v_tangential[:, 0]
    north = v_tangential[:, 1]

    # Calculate heading in degrees clockwise from North
    heading = np.degrees(np.arctan2(east, north))
    heading = (heading + 360) % 360  # Ensure heading is in range [0, 360)

    return gamma, heading


def calc_gamma_and_heading_itrf(r_itrf, t):
    """
    Calculate both the gamma angle and the heading of the flight path in the ITRF frame.

    Parameters
    ----------
    r : numpy.ndarray
        The position vectors in the ITRF frame, shaped (n, 3), where n is the number of vectors.

    t : numpy.ndarray or astropy.time.Time
        The times corresponding to the position vectors. Can be:
        - An array of GPS seconds (numpy.ndarray)
        - An Astropy Time object

    Returns
    -------
    tuple
        - numpy.ndarray: An array of gamma angles (in degrees) between the position and velocity vectors for each time point.
        - numpy.ndarray: An array of headings (in degrees), measured clockwise from North, for each time point.

    Notes
    -----
    - This function transforms position and velocity vectors from the GCRF (Geocentric Celestial Reference Frame)
      to the ITRF (International Terrestrial Reference Frame) using the provided time information.
    - Gamma is defined as the angle between the position vector and the velocity vector, offset by 90 degrees.
    - Heading is measured clockwise from North (aligned with the positive y-axis) in the normal plane of the position vector (r_itrf).
    - If the input time array is an Astropy Time object, it is converted to GPS time before processing.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """

    v_itrf = v_from_r(r_itrf, t)

    # Convert time to GPS seconds if necessary
    if isinstance(t[0], Time):
        t = t.gps

    # Normalize position and velocity vectors
    r_unit = r_itrf / np.linalg.norm(r_itrf, axis=1, keepdims=True)
    v_unit = v_itrf / np.linalg.norm(v_itrf, axis=1, keepdims=True)

    # Calculate gamma (angle between r_itrf and v_itrf, offset by 90 degrees)
    gamma = np.degrees(np.apply_along_axis(
        lambda x: angle_between_vectors(x[:3], x[3:]),
        1,
        np.concatenate((r_itrf, v_itrf), axis=1)
    )) - 90

    # Compute the flight path direction in the tangent plane of r_itrf
    v_tangential = v_unit - np.einsum('ij,ij->i', v_unit, r_unit)[:, None] * r_unit
    v_tangential /= np.linalg.norm(v_tangential, axis=1, keepdims=True)

    # Extract x (East) and y (North) components in the local tangent plane
    east = v_tangential[:, 0]
    north = v_tangential[:, 1]

    # Calculate heading in degrees clockwise from North
    heading = np.degrees(np.arctan2(east, north))
    heading = (heading + 360) % 360  # Ensure heading is in range [0, 360)

    return gamma, heading
