import numpy as np
from astropy.time import Time
from astropy.coordinates import GCRS, CartesianRepresentation
from astropy import units as u


def j2000_to_gcrf(pos_j2000, obstime):
    """
    Convert n x 3 array of J2000 (ECI) positions to GCRF coordinates.

    Parameters:
    -----------
    pos_j2000 : ndarray
        n x 3 array of x, y, z positions in J2000 coordinates (meters).
    obstime : float or astropy.time.Time or str
        Observation time for GCRF frame. Can be GPS seconds (float, seconds since
        1980-01-06 00:00:00 UTC), an astropy.time.Time object, or an ISO string.

    Returns:
    --------
    pos_gcrf : ndarray
        n x 3 array of x, y, z positions in GCRF coordinates (meters).

    Raises:
    -------
    ValueError
        If pos_j2000 is not an n x 3 array or obstime is invalid.
    """
    # Validate and process input positions
    pos_j2000 = np.asarray(pos_j2000)
    if pos_j2000.ndim != 2 or pos_j2000.shape[1] != 3:
        raise ValueError("Input must be an n x 3 array of positions.")

    # Handle observation time
    if isinstance(obstime, (int, float)):
        t = Time(obstime, format='gps', scale='utc')
    elif isinstance(obstime, Time):
        t = obstime
    elif isinstance(obstime, str):
        obstime_clean = obstime.replace(' ', 'T')
        if obstime_clean.endswith('Z'):
            obstime_clean = obstime_clean.rstrip('Z')
        if '.' in obstime_clean:
            obstime_clean = obstime_clean.split('.')[0]
        try:
            t = Time(obstime_clean, format='isot', scale='utc')
        except ValueError as e:
            raise ValueError(f"Invalid obstime string format: {obstime}. Expected 'YYYY-MM-DDTHH:MM:SS'. Error: {e}")
    else:
        raise ValueError("obstime must be GPS seconds (float/int), an astropy.time.Time object, or a string")

    # Define J2000.0 epoch in Terrestrial Time (TT)
    j2000_time = Time("2000-01-01T12:00:00", scale='tt')

    # Create GCRS coordinates at J2000.0
    j2000_cart = CartesianRepresentation(x=pos_j2000[:, 0] * u.m,
                                         y=pos_j2000[:, 1] * u.m,
                                         z=pos_j2000[:, 2] * u.m)
    gcrs_j2000 = GCRS(j2000_cart, obstime=j2000_time)

    # Transform to GCRS at observation time
    gcrs_t = gcrs_j2000.transform_to(GCRS(obstime=t))

    # Extract the rotation matrix by transforming basis vectors
    # Define unit vectors in J2000 frame
    basis_vectors_j2000 = np.eye(3)  # [x, y, z] unit vectors
    basis_cart_j2000 = CartesianRepresentation(x=basis_vectors_j2000[:, 0] * u.m,
                                               y=basis_vectors_j2000[:, 1] * u.m,
                                               z=basis_vectors_j2000[:, 2] * u.m)
    basis_gcrs_j2000 = GCRS(basis_cart_j2000, obstime=j2000_time)
    basis_gcrs_t = basis_gcrs_j2000.transform_to(GCRS(obstime=t))

    # Construct rotation matrix from transformed basis vectors
    rotation_matrix = np.vstack((basis_gcrs_t.cartesian.x.value,
                                 basis_gcrs_t.cartesian.y.value,
                                 basis_gcrs_t.cartesian.z.value)).T

    # Apply rotation to input positions
    pos_gcrf = np.dot(rotation_matrix, pos_j2000.T).T
    return pos_gcrf
