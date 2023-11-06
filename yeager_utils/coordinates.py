from .constants import EARTH_RADIUS, WGS84_EARTH_OMEGA
from .time import hms_to_dd, dd_to_hms, dd_to_dms
from ssapy.body import get_body, MoonPosition
from ssapy.utils import normed, einsum_norm
import numpy as np
from astropy.time import Time


def rad0to2pi(angles):
    return (2 * np.pi + angles) * (angles < 0) + angles * (angles > 0)


def deg0to360(array_):
    try:
        return [i % 360 for i in array_]
    except TypeError:
        return array_ % 360


def deg0to360array(array_):
    return [i % 360 for i in array_]


def deg90to90(val_in):
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


def deg90to90array(array_):
    return [i % 90 for i in array_]


def cart2sph_deg(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) * (180 / np.pi)
    az = (np.arctan2(y, x)) * (180 / np.pi)
    return az, el, r


def cart_to_cyl(x, y, z):
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta, z


def inert2rot(x, y, xe, ye, xs=0, ys=0):  # Places Earth at (-1,0)
    earth_theta = np.arctan2(ye - ys, xe - xs)
    theta = np.arctan2(y - ys, x - xs)
    distance = np.sqrt(np.power((x - xs), 2) + np.power((y - ys), 2))
    xrot = distance * np.cos(np.pi + (theta - earth_theta))
    yrot = distance * np.sin(np.pi + (theta - earth_theta))
    return xrot, yrot


def sim_lonlatrad(x, y, z, xe, ye, ze, xs, ys, zs):
    # convert all to geo coordinates
    x = x - xe
    y = y - ye
    z = z - ze
    xs = xs - xe
    ys = ys - ye
    zs = zs - ze
    # convert x y z to lon lat radius
    longitude, latitude, radius = cart2sph_deg(x, y, z)
    slongitude, slatitude, sradius = cart2sph_deg(xs, ys, zs)
    # correct so that Sun is at (0,0)
    longitude = deg0to360(slongitude - longitude)
    latitude = latitude - slatitude
    return longitude, latitude, radius


def sun_ra_dec(time_):
    out = get_body(Time(time_, format='mjd'))
    return out.ra.to('rad').value, out.dec.to('rad').value


def ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    d_earth_mag = einsum_norm(r, 'ij,ij->i')
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    return ra, dec


def lonlat_distance(lat1, lat2, lon1, lon2):
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles
    # calculate the result
    return (c * EARTH_RADIUS)


def altitude2zenithangle(altitude, deg=True):
    if deg:
        out = 90 - altitude
    else:
        out = np.pi / 2 - altitude
    return out


def zenithangle2altitude(zenith_angle, deg=True):
    if deg:
        out = 90 - zenith_angle
    else:
        out = np.pi / 2 - zenith_angle
    return out


def rightasension2hourangle(right_ascension, local_time):
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


def equatorial_to_horizontal(observer_latitude, declination, right_ascension=None, hour_angle=None, local_time=None, hms=False):
    if right_ascension is not None:
        hour_angle = rightasension2hourangle(right_ascension, local_time)
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif hour_angle is not None:
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    elif right_ascension is not None and hour_angle is not None:
        print('Both right_ascension and hour_angle parameters are provided.\nUsing hour_angle for calculations.')
        if hms:
            hour_angle = hms_to_dd(hour_angle)
    else:
        print('Either right_ascension or hour_angle must be provided.')

    observer_latitude, hour_angle, declination = np.radians([observer_latitude, hour_angle, declination])

    zenith_angle = np.arccos(np.sin(observer_latitude) * np.sin(declination) + np.cos(observer_latitude) * np.cos(declination) * np.cos(hour_angle))

    altitude = zenithangle2altitude(zenith_angle, deg=False)

    _num = np.sin(declination) - np.sin(observer_latitude) * np.cos(zenith_angle)
    _den = np.cos(observer_latitude) * np.sin(zenith_angle)
    azimuth = np.arccos(_num / _den)

    if observer_latitude < 0:
        azimuth = np.pi - azimuth
    altitude, azimuth = np.degrees([altitude, azimuth])

    return azimuth, altitude


def horizontal_to_equatorial(observer_latitude, azimuth, altitude):
    altitude, azimuth, latitude = np.radians([altitude, azimuth, observer_latitude])
    zenith_angle = zenithangle2altitude(altitude)

    zenith_angle = [-zenith_angle if latitude < 0 else zenith_angle][0]

    declination = np.sin(latitude) * np.cos(zenith_angle)
    declination = declination + (np.cos(latitude) * np.sin(zenith_angle) * np.cos(azimuth))
    declination = np.arcsin(declination)

    _num = np.cos(zenith_angle) - np.sin(latitude) * np.sin(declination)
    _den = np.cos(latitude) * np.cos(declination)
    hour_angle = np.arccos(_num / _den)

    if (latitude > 0 > declination) or (latitude < 0 < declination):
        hour_angle = 2 * np.pi - hour_angle

    declination, hour_angle = np.degrees([declination, hour_angle])

    return hour_angle, declination


_ecliptic = 0.409092601  # np.radians(23.43927944)
cos_ec = 0.9174821430960974
sin_ec = 0.3977769690414367


def equatorial_xyz_to_ecliptic_xyz(xq, yq, zq):
    xc = xq
    yc = cos_ec * yq + sin_ec * zq
    zc = -sin_ec * yq + cos_ec * zq
    return xc, yc, zc


def ecliptic_xyz_to_equatorial_xyz(xc, yc, zc):
    xq = xc
    yq = cos_ec * yc - sin_ec * zc
    zq = sin_ec * yc + cos_ec * zc
    return xq, yq, zq


def xyz_to_ecliptic(xc, yc, zc, xe=0, ye=0, ze=0, degrees=False):
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


def xyz_to_equatorial(xq, yq, zq, xe=0, ye=0, ze=0, degrees=False):
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


def ecliptic_xyz_to_equatorial(xc, yc, zc, xe=0, ye=0, ze=0, degrees=False):
    # Convert ecliptic cartesian into equitorial cartesian
    x_ast_to_earth, y_ast_to_earth, z_ast_to_earth = ecliptic_xyz_to_equatorial_xyz(xc - xe, yc - ye, zc - ze)
    d_earth_mag = np.sqrt(np.power(x_ast_to_earth, 2) + np.power(y_ast_to_earth, 2) + np.power(z_ast_to_earth, 2))
    ra = rad0to2pi(np.arctan2(y_ast_to_earth, x_ast_to_earth))  # in radians
    dec = np.arcsin(z_ast_to_earth / d_earth_mag)
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def equatorial_to_ecliptic(right_ascension, declination, degrees=False):
    ra, dec = np.radians(right_ascension), np.radians(declination)
    ec_latitude = np.arcsin(cos_ec * np.sin(dec) - sin_ec * np.cos(dec) * np.sin(ra))
    ec_longitude = np.arctan((cos_ec * np.cos(dec) * np.sin(ra) + sin_ec * np.sin(dec)) / (np.cos(dec) * np.cos(ra)))
    if degrees:
        return deg0to360(np.degrees(ec_longitude)), np.degrees(ec_latitude)
    else:
        return rad0to2pi(ec_longitude), ec_latitude


def ecliptic_to_equatorial(lon, lat, degrees=False):
    lon, lat = np.radians(lon), np.radians(lat)
    ra = np.arctan((cos_ec * np.cos(lat) * np.sin(lon) - sin_ec * np.sin(lat)) / (np.cos(lat) * np.cos(lon)))
    dec = np.arcsin(cos_ec * np.sin(lat) + sin_ec * np.cos(lat) * np.sin(lon))
    if degrees:
        return np.degrees(ra), np.degrees(dec)
    else:
        return ra, dec


def proper_motion_ra_dec(r=None, v=None, x=None, y=None, z=None, vx=None, vy=None, vz=None, r_earth=np.array([0, 0, 0]), v_earth=np.array([0, 0, 0]), input_unit='si'):
    if r is None or v is None:
        if x is not None and y is not None and z is not None and vx is not None and vy is not None and vz is not None:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])
        else:
            raise ValueError("Either provide r and v arrays or individual coordinates (x, y, z) and velocities (vx, vy, vz)")

    # Subtract Earth's position and velocity from the input arrays
    r = r - r_earth
    v = v - v_earth

    # Distances to Earth and Sun
    d_earth_mag = einsum_norm(r, 'ij,ij->i')

    # RA / DEC calculation
    ra = rad0to2pi(np.arctan2(r[:, 1], r[:, 0]))  # in radians
    dec = np.arcsin(r[:, 2] / d_earth_mag)
    ra_unit_vector = np.array([-np.sin(ra), np.cos(ra), np.zeros(np.shape(ra))]).T
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


# Counter-clockwise direction
def rotate_axis(vector, axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(rotation_matrix, vector)


def rotate_via_numpy(x, y, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return float(m.T[0]), float(m.T[1])


def rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = -x * np.sin(radians) + y * np.cos(radians)
    return xx, yy


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])


def Ry(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])


def Rz(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])


# Rotate 2d - theta is a counterclockwise rotation
def rotate_2d(x, y, theta_to_rotate, x_origin=0, y_origin=0):
    theta = np.arctan2(y - x_origin, x - y_origin)
    distance = np.sqrt(np.power((x - x_origin), 2) + np.power((y - y_origin), 2))
    xrot = distance * np.cos(np.pi + (theta - theta_to_rotate))
    yrot = distance * np.sin(np.pi + (theta - theta_to_rotate))
    return xrot, yrot


# Using clockwise direction
def rotate_3d(vector, xtheta, ytheta, ztheta):
    vector = np.array(vector).flatten()
    return np.dot(vector, np.dot(Rz(ztheta), np.dot(Ry(ytheta), Rx(xtheta))))


def rotation_matrix_from_vectors(vec1, vec2=np.array([1, 0, 0])):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def rotate_points_3d(points, axis=np.array([0, 0, 1]), theta=-np.pi / 2):
    """
    Rotate a set of 3D points about a 3D axis by an angle theta in radians.

    Args:
        points (np.ndarray): The set of 3D points to rotate, as an Nx3 array.
        axis (np.ndarray): The 3D axis to rotate about, as a length-3 array. Default is the z-axis.
        theta (float): The angle to rotate by, in radians. Default is pi/2.

    Returns:
        np.ndarray: The rotated set of 3D points, as an Nx3 array.
    """
    # Normalize the axis to be a unit vector
    axis = axis / np.linalg.norm(axis)

    # Compute the quaternion representing the rotation
    qw = np.cos(theta / 2)
    qx, qy, qz = axis * np.sin(theta / 2)

    # Construct the rotation matrix from the quaternion
    R = np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

    # Apply the rotation matrix to the set of points
    rotated_points = np.dot(R, points.T).T

    return rotated_points


def perpendicular_vectors(v):
    """Returns two vectors that are perpendicular to v and each other."""
    # Check if v is the zero vector
    if np.allclose(v, np.zeros_like(v)):
        raise ValueError("Input vector cannot be the zero vector.")

    # Choose an arbitrary non-zero vector w that is not parallel to v
    w = np.array([1., 0., 0.])
    if np.allclose(v, w) or np.allclose(v, -w):
        w = np.array([0., 1., 0.])
    u = np.cross(v, w)
    if np.allclose(u, np.zeros_like(u)):
        w = np.array([0., 0., 1.])
        u = np.cross(v, w)
    w = np.cross(v, u)

    return u, w


def points_on_circle(r, v, rad, num_points=4):
    # Convert inputs to numpy arrays
    r = np.array(r)
    v = np.array(v)

    # Find the perpendicular vectors to the given vector v
    if np.all(v[:2] == 0):
        if np.all(v[2] == 0):
            raise ValueError("The given vector v must not be the zero vector.")
        else:
            u = np.array([1, 0, 0])
    else:
        u = np.array([-v[1], v[0], 0])
    u = u / np.linalg.norm(u)
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-15:
        # v is parallel to z-axis
        w = np.array([0, 1, 0])
    else:
        w = w / w_norm
    # Generate a sequence of angles for equally spaced points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Compute the x, y, z coordinates of each point on the circle
    x = rad * np.cos(angles) * u[0] + rad * np.sin(angles) * w[0]
    y = rad * np.cos(angles) * u[1] + rad * np.sin(angles) * w[1]
    z = rad * np.cos(angles) * u[2] + rad * np.sin(angles) * w[2]

    # Apply rotation about z-axis by 90 degrees
    rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotated_points = np.dot(rot_matrix, np.column_stack((x, y, z)).T).T

    # Translate the rotated points to the center point r
    points_rotated = rotated_points + r.reshape(1, 3)

    return points_rotated


def gcrf_to_lunar(r, times):
    class MoonRotator:
        def __init__(self):
            self.mpm = MoonPosition()

        def __call__(self, r, t):
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
    return rotator(r, times)


def gcrf_to_stationary_lunar(r, times):
    from .body import get_body
    return gcrf_to_lunar(r, times) - gcrf_to_lunar(get_body('moon').position(times).T, times)


def gcrf_to_ecef(r_gcrf, t):
    if isinstance(t, Time):
        t = t.gps
    rotation_angles = WGS84_EARTH_OMEGA * (t - t[0])
    cos_thetas = np.cos(rotation_angles)
    sin_thetas = np.sin(rotation_angles)

    # Create an array of 3x3 rotation matrices
    Rz = np.array([[cos_thetas, -sin_thetas, np.zeros_like(cos_thetas)],
                  [sin_thetas, cos_thetas, np.zeros_like(cos_thetas)],
                  [np.zeros_like(cos_thetas), np.zeros_like(cos_thetas), np.ones_like(cos_thetas)]]).T

    # Apply the rotation matrices to all rows of r_gcrf simultaneously
    r_ecef = np.einsum('ijk,ik->ij', Rz, r_gcrf)
    return r_ecef


def gcrf_to_radec(gcrf_coords):
    x, y, z = gcrf_coords
    # Calculate right ascension in radians
    ra = np.arctan2(y, x)
    # Convert right ascension to degrees
    ra_deg = np.degrees(ra)
    # Normalize right ascension to the range [0, 360)
    ra_deg = ra_deg % 360
    # Calculate declination in radians
    dec_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    # Convert declination to degrees
    dec_deg = np.degrees(dec_rad)
    return (ra_deg, dec_deg)
