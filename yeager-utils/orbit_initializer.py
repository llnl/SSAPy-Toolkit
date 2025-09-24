import ssapy
import numpy as np


class OrbitInitialize:
    def __init__(self):
        pass

    @staticmethod
    def DRO(t, delta_r=7.52064e7, delta_v=344):
        moon = ssapy.get_body("moon")

        unit_vector_moon = moon.position(t) / np.linalg.norm(moon.position(t))
        moon_v = (moon.position(t.gps) - moon.position(t.gps - 1)) / 1
        unit_vector_moon_velocity = moon_v / np.linalg.norm(moon_v)

        r = (np.linalg.norm(moon.position(t)) - delta_r) * unit_vector_moon
        v = (np.linalg.norm(moon_v) + delta_v) * unit_vector_moon_velocity

        orbit = ssapy.Orbit(r=r, v=v, t=t)
        return orbit

    def Lunar_L4(t, delta_r=7.52064e7, delta_v=344):
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
