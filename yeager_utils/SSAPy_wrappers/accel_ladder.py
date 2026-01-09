def ssapy_accel_ladder():
    """
    Returns a dict: name -> ssapy Accel object.

    Ladder (add effects in typical strength order, then upgrade gravity fidelity):
      1) kepler_earth_point_mass
      2) kepler_earth_plus_moon
      3) + earth_j2
      4) + sun
      5) + planets
      6) upgrade earth gravity -> full earth harmonics (140x140)
      7) add full lunar harmonics (20x20) on top of moon point-mass
      8) full earth harmonics + (moon point + moon harmonics) + sun + planets
    """
    from ssapy.accel import AccelKepler
    from ssapy.body import get_body
    from ssapy.gravity import AccelHarmonic, AccelThirdBody

    earth = get_body("Earth", model="EGM2008")
    moon = get_body("moon")
    sun = get_body("Sun")
    planets = [get_body(p) for p in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")]

    a_kepler = AccelKepler()

    # Earth gravity (non-central terms; Kepler provides the central term)
    a_earth_j2 = AccelHarmonic(earth, 2, 0)
    a_earth_full = AccelHarmonic(earth, 140, 140)

    # Moon gravity: point-mass as third-body + optional non-central harmonics
    a_moon_point = AccelThirdBody(moon)
    a_moon_full = AccelHarmonic(moon, 20, 20)

    # Sun + planets
    a_sun = AccelThirdBody(sun)

    a_planets = None
    for p in planets:
        term = AccelThirdBody(p)
        a_planets = term if a_planets is None else a_planets + term

    suite = {}

    suite["kepler_earth_point_mass"] = a_kepler

    kepler_moon = a_kepler + a_moon_point
    suite["kepler_earth_plus_moon"] = kepler_moon

    kepler_moon_earthj2 = kepler_moon + a_earth_j2
    suite["kepler_earth_plus_moon_plus_earth_j2"] = kepler_moon_earthj2

    kepler_moon_earthj2_sun = kepler_moon_earthj2 + a_sun
    suite["kepler_earth_plus_moon_plus_earth_j2_plus_sun"] = kepler_moon_earthj2_sun

    kepler_moon_earthj2_sun_planets = kepler_moon_earthj2_sun + a_planets
    suite["kepler_earth_plus_moon_plus_earth_j2_plus_sun_plus_planets"] = kepler_moon_earthj2_sun_planets

    earth_full_only = a_kepler + a_earth_full
    suite["kepler_plus_earth_full_harmonics"] = earth_full_only

    earth_full_plus_moon_full = earth_full_only + a_moon_point + a_moon_full
    suite["kepler_plus_earth_full_harmonics_plus_moon_full_harmonics"] = earth_full_plus_moon_full

    suite["kepler_plus_earth_full_harmonics_plus_moon_full_harmonics_plus_sun_plus_planets"] = (
        earth_full_plus_moon_full + a_sun + a_planets
    )

    return suite
