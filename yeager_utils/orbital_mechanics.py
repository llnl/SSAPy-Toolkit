from .constants import EARTH_MU, au_to_m
from .coordinates import deg90to90, deg0to360
from .utils import nby3shape
import rebound
from rebound import hash as h
from astropy.time import Time
import numpy as np


def mean_longitude(longitude_of_ascending_node=True, argument_of_periapsis=True, mean_anomaly=True):
    return longitude_of_ascending_node + argument_of_periapsis + mean_anomaly


def true_anomaly(eccentricity=True, eccentric_anomaly=True, mean_anomaly=True, true_longitude=True, argument_of_periapsis=True, longitude_of_ascending_node=True):
    if eccentricity is not True and eccentric_anomaly is not True:
        beta = eccentricity / (1 + np.sqrt(1 - eccentricity**2))
        true_anomaly = eccentric_anomaly + 2 * np.arctan2(beta * np.sin(eccentric_anomaly) / (1 - beta * np.cos(eccentric_anomaly)))
    elif eccentricity is not True and mean_anomaly is not True:
        true_anomaly = mean_anomaly + (2 * eccentricity - 1 / 4 * eccentricity**3) * np.sin(mean_anomaly) + 5 / 4 * eccentricity ** 2 * np.sin(2 * mean_anomaly) + 13 / 12 * eccentricity ** 3 * np.sin(3 * mean_anomaly)
    elif true_longitude is not True and longitude_of_ascending_node is not True and argument_of_periapsis is not True:
        true_anomaly = true_longitude - longitude_of_ascending_node - argument_of_periapsis
    else:
        return print('Not enough information provided to calculate true anomaly.')
    return true_anomaly


def rebound_orbital_elements(planet='earth', time="2000-1-1", format='utc'):
    if not isinstance(time, Time):
        time = Time(time, format=format)
    jd = time.jd
    try:
        if isinstance(jd, float) or isinstance(jd, int):
            jd = str(jd)
        if jd[0:2] == 'JD':
            pass
        else:
            jd = f'JD{jd}'
    except IndexError:
        print('Error with the date provided. Give a year/month/day or JD.')
        return
    sim = rebound.Simulation()
    sim.add("sun", date=jd, hash=0)
    sim.add("mercury", date=jd, hash=1)
    sim.add("venus", date=jd, hash=2)
    sim.add("earth", date=jd, hash=3)
    sim.add("mars", date=jd, hash=4)
    sim.add("jupiter", date=jd, hash=5)
    sim.add("saturn", date=jd, hash=6)
    sim.add("uranus", date=jd, hash=7)
    sim.add("neptune", date=jd, hash=8)
    sim.move_to_com()
    if planet.lower() == "all":
        orbitals = {}
        for i, planet_str in enumerate(['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']):
            orbitals[planet_str] = {
                'a': sim.particles[h(i + 1)].a,
                'e': sim.particles[h(i + 1)].e,
                'i': sim.particles[h(i + 1)].inc,
                'true_longitude': sim.particles[h(i + 1)].theta,
                'argument_of_pericenter': sim.particles[h(i + 1)].omega,
                'longitude_of_ascending_node': sim.particles[h(i + 1)].Omega,
                'true_anomaly': sim.particles[h(i + 1)].f,
                'longitude_of_pericenter': sim.particles[h(i + 1)].pomega,
                'mean_longitude': sim.particles[h(i + 1)].l,
                'mean_anomaly': sim.particles[h(i + 1)].M,
                'eccentric_anomaly': rebound.M_to_E(sim.particles[h(i + 1)].e, sim.particles[h(i + 1)].M)
            }
        return orbitals
    elif planet.lower() == "sun":
        index = 0
    elif planet.lower() == "mercury":
        index = 1
    elif planet.lower() == "venus":
        index = 2
    elif planet.lower() == "earth":
        index = 3
    elif planet.lower() == "mars":
        index = 4
    elif planet.lower() == "jupiter":
        index = 5
    elif planet.lower() == "saturn":
        index = 6
    elif planet.lower() == "uranus":
        index = 7
    elif planet.lower() == "neptune":
        index = 8
    return {
        'a': sim.particles[h(index)].a,
        'e': sim.particles[h(index)].e,
        'i': sim.particles[h(index)].inc,
        'true_longitude': sim.particles[h(index)].theta,
        'argument_of_pericenter': sim.particles[h(index)].omega,
        'longitude_of_ascending_node': sim.particles[h(index)].Omega,
        'true_anomaly': sim.particles[h(index)].f,
        'longitude_of_pericenter': sim.particles[h(index)].pomega,
        'mean_longitude': sim.particles[h(index)].l,
        'mean_anomaly': sim.particles[h(index)].M,
        'eccentric_anomaly': rebound.M_to_E(sim.particles[h(index)].e, sim.particles[h(index)].M)
    }


######################################################################
# Calculate JPL orbital elements https://ssd.jpl.nasa.gov/planets/approx_pos.html
######################################################################
def j2000_orbitals(planet='earth', Teph=2451545.0):  # date input is in jd
    # ecliptic and equinox of J2000, valid for the time-interval 1800 AD - 2050 AD table 1-https://ssd.jpl.nasa.gov/planets/approx_pos.html
    if planet.lower() == 'mercury':
        anaut = 0.38709927
        enaut = 0.20563593
        inaut = 7.00497902
        Lnaut = 252.25032350
        onaut = 77.45779628
        Onaut = 48.33076593
        arate = 0.00000037
        erate = 0.00001906
        irate = -0.00594749
        Lrate = 149472.67411175
        orate = 0.16047689
        Orate = -0.12534081
    elif planet.lower() == 'venus':
        anaut = 0.72333566
        enaut = 0.00677672
        inaut = 3.39467605
        Lnaut = 181.97909950
        onaut = 131.60246718
        Onaut = 76.67984255
        arate = 0.00000390
        erate = -0.00004107
        irate = -0.00078890
        Lrate = 58517.81538729
        orate = 0.00268329
        Orate = -0.27769418
    elif planet.lower() == 'earth':
        anaut = 1.00000261
        enaut = 0.01671123
        inaut = -0.00001531
        Lnaut = 100.46457166
        onaut = 102.93768193
        Onaut = 0.0
        arate = 0.00000562
        erate = -0.00004392
        irate = -0.01294668
        Lrate = 35999.37244981
        orate = 0.32327364
        Orate = 0.0
    elif planet.lower() == 'mars':
        anaut = 1.52371034
        enaut = 0.09339410
        inaut = 1.84969142
        Lnaut = -4.55343205
        onaut = -23.94362959
        Onaut = 49.55953891
        arate = 0.00001847
        erate = 0.00007882
        irate = -0.00813131
        Lrate = 19140.30268499
        orate = 0.44441088
        Orate = -0.29257343
    elif planet.lower() == 'jupiter':
        anaut = 5.20288700
        enaut = 0.04838624
        inaut = 1.30439695
        Lnaut = 34.39644051
        onaut = 14.72847983
        Onaut = 100.47390909
        arate = -0.00011607
        erate = -0.00013253
        irate = -0.00183714
        Lrate = 3034.74612775
        orate = 0.21252668
        Orate = 0.20469106
    elif planet.lower() == 'saturn':
        anaut = 9.53667594
        enaut = 0.05386179
        inaut = 2.48599187
        Lnaut = 49.95424423
        onaut = 92.59887831
        Onaut = 113.66242448
        arate = -0.00125060
        erate = -0.00050991
        irate = 0.00193609
        Lrate = 1222.49362201
        orate = -0.41897216
        Orate = -0.28867794
    elif planet.lower() == 'uranus':
        anaut = 19.18916464
        enaut = 0.04725744
        inaut = 0.77263783
        enaut = 313.23810451
        onaut = 170.95427630
        Onaut = 74.01692503
        arate = -0.00196176
        erate = -0.00004397
        irate = -0.00242939
        Lrate = 428.48202785
        orate = 0.40805281
        Orate = 0.04240589
    elif planet.lower() == 'neptune':
        anaut = 30.06992276
        enaut = 0.00859048
        inaut = 1.77004347
        Lnaut = -55.12002969
        onaut = 44.96476227
        Onaut = 131.78422574
        arate = 0.00026291
        erate = 0.00005105
        irate = 0.00035372
        Lrate = 218.45945325
        orate = -0.32241464
        Orate = -0.00508664

    # number of centuries after J2000
    T = (Teph - 2451545.0) / 36525
    a = anaut + arate * T
    e = enaut + erate * T
    i = inaut + irate * T
    mean_longitude = Lnaut + Lrate * T
    longitude_of_perihelion = onaut + orate * T
    longitude_of_the_ascending_node = Onaut + Orate * T

    return {'a': a, 'e': e, 'i': deg90to90(i), 'mean_longitude': deg0to360(mean_longitude), 'longitude_of_perihelion': deg0to360(longitude_of_perihelion), 'longitude_of_the_ascending_node': deg0to360(longitude_of_the_ascending_node)}


def kepler_to_state(a=1, e=0, i=0, raan=0, w=0, nu=0, mu=EARTH_MU):
    """
    Converts keplerian orbital elements to a state vector.

    Parameters:
    - a: semi-major axis (in meters)
    - e: eccentricity
    - i: inclination (in radians)
    - raan: right ascension of the ascending node (in radians)
    - w: argument of perigee (in radians)
    - nu: true anomaly (in radians)
    - mu: gravitational parameter of the central body (in meters^3/s^2)

    Returns:
    - r: position vector in the inertial frame (in meters)
    - v: velocity vector in the inertial frame (in meters/s)

    Can pass a 2D array of shape (N, 6) where N is the number of sets of keplerian orbital elements you want to convert.
    The function will return two arrays of shape (N, 3) with the corresponding position and velocity vectors.
    """

    # Compute the position vector in the perifocal frame
    r_pf = a * (1 - e**2) / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])

    # Compute the velocity vector in the perifocal frame
    v_pf = np.sqrt(mu / a) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    # Create the rotation matrix from the perifocal to the inertial frame
    R_pf_i = np.array([[np.cos(raan) * np.cos(w) - np.sin(raan) * np.cos(i) * np.sin(w),
                        -np.cos(raan) * np.sin(w) - np.sin(raan) * np.cos(i) * np.cos(w),
                        np.sin(raan) * np.sin(i)],
                       [np.sin(raan) * np.cos(w) + np.cos(raan) * np.cos(i) * np.sin(w),
                        -np.sin(raan) * np.sin(w) + np.cos(raan) * np.cos(i) * np.cos(w),
                        -np.cos(raan) * np.sin(i)],
                       [np.sin(i) * np.sin(w), np.sin(i) * np.cos(w), np.cos(i)]])

    # Transform the position and velocity vectors to the inertial frame
    r = np.dot(R_pf_i, r_pf)
    v = np.dot(R_pf_i, v_pf)

    return r, v


def state_to_kepler(r, v, mu=EARTH_MU):
    """
    Converts a state vector to keplerian orbital elements.

    Parameters:
    - r: position vector in the inertial frame (in meters)
    - v: velocity vector in the inertial frame (in meters/s)
    - mu: gravitational parameter of the central body (in meters^3/s^2)

    Returns:
    - a: semi-major axis (in meters)
    - e: eccentricity
    - i: inclination (in radians)
    - raan: right ascension of the ascending node (in radians)
    - w: argument of perigee (in radians)
    - nu: true anomaly (in radians)

    Can pass two arrays of shape (N, 3) where N is the number of state vectors you want to convert.
    The function will return an array of shape (N, 6) with the corresponding keplerian orbital elements.
    """

    # Compute the angular momentum vector
    h = np.cross(r, v)

    # Compute the eccentricity vector
    e_vec = (np.cross(v, h) / mu) - r / np.linalg.norm(r)

    # Compute the semi-major axis
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v)**2 / mu)

    # Compute the eccentricity
    e = np.linalg.norm(e_vec)

    # Compute the inclination
    i = np.arccos(h[2] / np.linalg.norm(h))

    # Compute the right ascension of the ascending node
    if h[0] >= 0:
        raan = np.arccos(h[0] / np.linalg.norm(h[:2]))
    else:
        raan = 2 * np.pi - np.arccos(h[0] / np.linalg.norm(h[:2]))

    # Compute the argument of perigee
    if e_vec[2] >= 0:
        w = np.arccos(np.dot(h, e_vec) / (np.linalg.norm(h) * e))
    else:
        w = 2 * np.pi - np.arccos(np.dot(h, e_vec) / (np.linalg.norm(h) * e))

    # Compute the true anomaly
    if np.dot(r, v) >= 0:
        nu = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
    else:
        nu = 2 * np.pi - np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))

    return a, e, i, raan, w, nu


def kepler_to_parametric(a, e, i, omega, w, theta):
    # Convert to radians
    i = np.radians(i)
    omega = np.radians(omega)
    w = np.radians(w)
    theta = np.radians(theta)

    # Compute the semi-major and semi-minor axes
    b = a * np.sqrt(1 - e**2)
    # Compute the parametric coefficients
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = 0
    # Rotate the ellipse about the x-axis
    x_prime = x
    y_prime = y * np.cos(i) - z * np.sin(i)
    z_prime = y * np.sin(i) + z * np.cos(i)

    # Rotate the ellipse about the z-axis
    x_prime_prime = x_prime * np.cos(omega) - y_prime * np.sin(omega)
    y_prime_prime = x_prime * np.sin(omega) + y_prime * np.cos(omega)
    z_prime_prime = z_prime

    # Translate the ellipse
    x_final = x_prime_prime + w
    y_final = y_prime_prime
    z_final = z_prime_prime
    return x_final, y_final, z_final


def calculate_orbital_elements(r_, v_, mu_barycenter=EARTH_MU, frame='gcrf'):
    # mu_barycenter - all bodies interior to Earth
    # 1.0013415732186798 #All bodies of solar system
    mu_ = mu_barycenter
    rarr = nby3shape(r_)
    varr = nby3shape(v_)
    aarr = []
    earr = []
    incarr = []
    true_longitudearr = []
    argument_of_periapsisarr = []
    longitude_of_ascending_nodearr = []
    true_anomalyarr = []
    hmagarr = []
    for r, v in zip(rarr, varr):
        r = np.array(r)  # print(f'r: {r}')
        v = np.array(v)  # print(f'v: {v}')

        rmag = np.sqrt(r.dot(r))
        vmag = np.sqrt(v.dot(v))

        h = np.cross(r, v)
        hmag = np.sqrt(h.dot(h))
        n = np.cross(np.array([0, 0, 1]), h)

        a = 1 / ((2 / rmag) - (vmag ** 2) / mu_)

        evector = np.cross(v, h) / (mu_) - r / rmag
        e = np.sqrt(evector.dot(evector))

        inc = np.arccos(h[2] / hmag)

        if np.dot(r, v) > 0:
            true_anomaly = np.arccos(np.dot(evector, r) / (e * rmag))
        else:
            true_anomaly = 2 * np.pi - np.arccos(np.dot(evector, r) / (e * rmag))
        if evector[2] >= 0:
            argument_of_periapsis = np.arccos(np.dot(n, evector) / (e * np.sqrt(n.dot(n))))
        else:
            argument_of_periapsis = 2 * np.pi - np.arccos(np.dot(n, evector) / (e * np.sqrt(n.dot(n))))
        if n[1] >= 0:
            longitude_of_ascending_node = np.arccos(n[0] / np.sqrt(n.dot(n)))
        else:
            longitude_of_ascending_node = 2 * np.pi - np.arccos(n[0] / np.sqrt(n.dot(n)))

        true_longitude = true_anomaly + argument_of_periapsis + longitude_of_ascending_node
        aarr.append(a)
        earr.append(e)
        incarr.append(inc)
        true_longitudearr.append(true_longitude)
        argument_of_periapsisarr.append(argument_of_periapsis)
        longitude_of_ascending_nodearr.append(longitude_of_ascending_node)
        true_anomalyarr.append(true_anomaly)
        hmagarr.append(hmag)
    return {
        'a': aarr,
        'e': earr,
        'i': incarr,
        'tl': true_longitudearr,
        'ap': argument_of_periapsisarr,
        'raan': longitude_of_ascending_nodearr,
        'ta': true_anomalyarr,
        'L': hmagarr
    }


def periapsis(a, e):
    return (1 - e) * a


def apoapsis(a, e):
    return (1 + e) * a


def peri_apo_from_rv(perigee, apogee):
    # Semi-major axis
    a = (perigee + apogee) / 2
    # Eccentricity
    e = (apogee - perigee) / (apogee + perigee)
    return {"a": a, "e": e}


def peri_apo_apsis_from_rv(r, v):
    temp = calculate_orbital_elements(r, v, EARTH_MU)
    return {"periapsis": periapsis(temp['a'], temp['e']), "apoapsis": apoapsis(temp['a'], temp['e'])}


def vcirc(r=au_to_m, mu_=1.32712440018e20 + 2.2032e13 + 3.24859e14):
    return np.sqrt(mu_ / r)
