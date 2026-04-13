"""Helpers for constructing common SSAPy orbit initial conditions."""

import numpy as np
import ssapy


class OrbitInitialize:
    """Factory for SSAPy orbits with common initial conditions."""

    def __init__(self):
        pass

    @staticmethod
    def DRO(t, delta_r=7.52064e7, delta_v=344):
        """Construct a distant retrograde orbit (DRO) around the Moon.

        Parameters
        ----------
        t :
            SSAPy time object specifying the epoch.
        delta_r : float, optional
            Radial offset from the Moon in meters, by default 7.52064e7.
        delta_v : float, optional
            Velocity offset in m/s, by default 344.

        Returns
        -------
        ssapy.Orbit
            Initialized DRO orbit at the requested epoch.
        """
        moon = ssapy.get_body("moon")

        unit_vector_moon = moon.position(t) / np.linalg.norm(moon.position(t))
        moon_v = (moon.position(t.gps) - moon.position(t.gps - 1)) / 1
        unit_vector_moon_velocity = moon_v / np.linalg.norm(moon_v)

        r = (np.linalg.norm(moon.position(t)) - delta_r) * unit_vector_moon
        v = (np.linalg.norm(moon_v) + delta_v) * unit_vector_moon_velocity

        orbit = ssapy.Orbit(r=r, v=v, t=t)
        return orbit

    def Lunar_L4(t, delta_r=7.52064e7, delta_v=344):
        """Construct an orbit near the lunar L4 point.

        Parameters
        ----------
        t :
            SSAPy time object specifying the epoch.
        delta_r : float, optional
            Radial offset from the Moon in meters, by default 7.52064e7.
        delta_v : float, optional
            Velocity offset in m/s, by default 344.

        Returns
        -------
        ssapy.Orbit
            Initialized orbit near the L4 Lagrange point.
        """
        moon = ssapy.get_body("moon")

        unit_vector_moon = moon.position(t) / np.linalg.norm(moon.position(t))
        moon_v = (moon.position(t.gps) - moon.position(t.gps - 1)) / 1
        unit_vector_moon_velocity = moon_v / np.linalg.norm(moon_v)

        r = (np.linalg.norm(moon.position(t)) - delta_r) * unit_vector_moon
        v = (np.linalg.norm(moon_v) + delta_v) * unit_vector_moon_velocity

        orbit = ssapy.Orbit(r=r, v=v, t=t)
        return orbit
# Usage example:
# orbit = OrbitInitialize.DRO(t)
