from types import SimpleNamespace
import importlib

import numpy as np
import pytest


class FakeTime:
    def __init__(self, value=0.0, *args, **kwargs):
        if isinstance(value, (list, tuple)):
            self.gps = np.zeros(len(value), dtype=float)
        elif hasattr(value, "gps"):
            self.gps = value.gps
        else:
            self.gps = float(value)


class FakeMoon:
    period = 2.3605915968e6
    radius = 384_400_000.0

    def position(self, t):
        values = np.atleast_1d(np.asarray(t, dtype=float))
        theta = 2.0 * np.pi * values / self.period
        coords = np.vstack((self.radius * np.cos(theta), self.radius * np.sin(theta), np.zeros_like(theta)))
        return coords[:, 0] if np.asarray(t).ndim == 0 else coords


def test_lagrange_points_and_lunar_frame_helpers(monkeypatch):
    import ssapy_toolkit.orbital_mechanics.lagrange_points as module

    monkeypatch.setattr(module, "Time", FakeTime)
    monkeypatch.setattr(module, "get_body", lambda name: FakeMoon())
    monkeypatch.setattr(module, "gcrf_to_lunar", lambda r, t: np.asarray(r) + 1.0)
    monkeypatch.setattr(module, "gcrf_to_lunar_fixed", lambda r, t: np.asarray(r) - 1.0)

    normal = module.moon_normal_vector(FakeTime(0.0))
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0)
    assert normal[2] > 0.0

    points = module.lunar_lagrange_points(FakeTime(0.0))
    assert set(points) == {"L1", "L2", "L3", "L4", "L5"}
    assert points["L1"].shape == (3,)
    np.testing.assert_allclose(points["L3"], [-FakeMoon.radius, 0.0, 0.0], atol=1e-6)

    circular = module.lunar_lagrange_points_circular(FakeTime(0.0))
    assert circular["L4"].shape == (3,)
    assert circular["L5"].shape == (3,)

    lunar = module.lagrange_points_lunar_frame()
    lunar_fixed = module.lagrange_points_lunar_fixed_frame()
    assert set(lunar) == set(points)
    assert set(lunar_fixed) == set(points)


def test_bbox_and_lonlat_perigee_helpers(monkeypatch):
    from ssapy_toolkit.coordinates.lon_lat_bbox import bbox_min
    perigee = importlib.import_module("ssapy_toolkit.coordinates.lonlat_perigee")

    assert bbox_min([10], [20]) == (20, 20, 10.0, 10.0, 0.0)
    lat_min, lat_max, lon_left, lon_right, span = bbox_min([170, -170, 175], [-5, 10, 0])
    assert (lat_min, lat_max) == (-5, 10)
    assert lon_left > lon_right
    assert span == pytest.approx(20.0)
    with pytest.raises(ValueError, match="provided"):
        bbox_min(None, [1])
    with pytest.raises(ValueError, match="same nonzero"):
        bbox_min([1, 2], [1])

    class FakeOrbit:
        def __init__(self, r, v, t, mu):
            self.r = np.asarray(r)
            self.v = np.asarray(v)
            self.t = t
            self.mu = mu

    monkeypatch.setattr(perigee, "astropy_surface_rv", lambda lon, lat, t: (np.array([1.0, 0.0, 0.0]), np.zeros(3)))
    monkeypatch.setattr(perigee, "Orbit", FakeOrbit)
    orbit = perigee.lonlat_perigee(lon=0.0, lat=0.0, t="2025-01-01", alt=1_000.0, e=0.1, i=90.0)
    assert orbit.r[0] == pytest.approx(perigee.EARTH_RADIUS + 1_000.0)
    assert np.linalg.norm(orbit.v) == pytest.approx(np.sqrt(perigee.EARTH_MU * 1.1 / np.linalg.norm(orbit.r)))
    assert orbit.v[2] > 0.0


def test_gcrf_to_itrf_and_llh_helpers(monkeypatch):
    itrf = importlib.import_module("ssapy_toolkit.coordinates.gcrf_to_itrf")
    from ssapy_toolkit.coordinates.gcrf_to_llh import gcrf_to_llh
    from astropy.time import Time

    monkeypatch.setattr(itrf, "to_gps", lambda t: np.asarray(t, dtype=float))
    monkeypatch.setattr(itrf, "groundTrack", lambda r, t, format="cartesian": (np.asarray(r)[:, 0] + 1.0, np.asarray(r)[:, 1] + 2.0, np.asarray(r)[:, 2] + 3.0))
    monkeypatch.setattr(itrf, "v_from_r", lambda pos, t: np.ones_like(pos) * 4.0)

    r = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pos = itrf.gcrf_to_itrf(r, [0.0, 1.0])
    np.testing.assert_allclose(pos, [[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])
    pos, vel = itrf.gcrf_to_itrf(r, [0.0, 1.0], v=np.zeros_like(r))
    np.testing.assert_allclose(vel, np.ones_like(r) * 4.0)

    with pytest.raises(ValueError, match="shape"):
        gcrf_to_llh(np.ones((2, 2)), Time("2025-01-01T00:00:00", scale="utc"))
    lon, lat, height = gcrf_to_llh(np.array([6_378_137.0, 0.0, 0.0]), Time("2025-01-01T00:00:00", scale="utc"))
    assert isinstance(lon, float)
    assert isinstance(lat, float)
    assert isinstance(height, float)


def test_gcrf_to_lunar_and_fixed_helpers(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.coordinates.gcrf_to_lunar")

    class FakeMoonPosition:
        def __call__(self, t):
            values = np.atleast_1d(np.asarray(t, dtype=float))
            return np.vstack((np.ones_like(values), values + 2.0, np.zeros_like(values)))

    class FakeMoonBody:
        def position(self, t):
            values = np.atleast_1d(np.asarray(t, dtype=float))
            return np.vstack((np.ones_like(values), values + 2.0, np.zeros_like(values)))

    monkeypatch.setattr(module, "MoonPosition", FakeMoonPosition)
    monkeypatch.setattr(module, "get_body", lambda name: FakeMoonBody())
    monkeypatch.setattr(module, "v_from_r", lambda r, t: np.ones_like(r) * 5.0)

    t = np.array([0.0, 1.0, 2.0])
    r = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]])
    lunar = module.gcrf_to_lunar(r, t)
    assert lunar.shape == (3, 3)
    lunar_pos, lunar_vel = module.gcrf_to_lunar(r, t, v=np.zeros_like(r))
    np.testing.assert_allclose(lunar_pos, lunar)
    np.testing.assert_allclose(lunar_vel, np.ones_like(lunar) * 5.0)

    fixed = module.gcrf_to_lunar_fixed(r, t)
    assert fixed.shape == (3, 3)
    fixed_pos, fixed_vel = module.gcrf_to_lunar_fixed(r, t, v=np.zeros_like(r))
    np.testing.assert_allclose(fixed_pos, fixed)
    np.testing.assert_allclose(fixed_vel, np.ones_like(fixed) * 5.0)


def test_v_from_r_validation_and_time_inputs():
    from astropy.time import Time
    from ssapy_toolkit.coordinates.v_from_r import v_from_r

    positions = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    times = Time([0.0, 2.0, 4.0], format="gps", scale="utc")

    velocities = v_from_r(positions, times)

    np.testing.assert_allclose(velocities, np.array([[1.0, 2.0, 3.0]] * 3))
    with pytest.raises(ValueError, match="shape"):
        v_from_r(np.ones(3), [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="1D"):
        v_from_r(positions, np.ones((3, 1)))
    with pytest.raises(ValueError, match="same length"):
        v_from_r(positions, [0.0, 1.0])
    with pytest.raises(ValueError, match="at least two"):
        v_from_r(positions[:1], [0.0])


def test_surface_rv_wrappers_forward_arguments(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.coordinates.surface_rv")
    astropy_calls = []

    def fake_astropy_surface_rv(**kwargs):
        astropy_calls.append(kwargs)
        return np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])

    class FakeEarthObserver:
        def __init__(self, lon, lat, elevation=0.0, fast=False):
            self.lon = lon
            self.lat = lat
            self.elevation = elevation
            self.fast = fast

        def getRV(self, t):
            return np.array([self.lon, self.lat, self.elevation]), np.array([t, float(self.fast), 0.0])

    monkeypatch.setattr(module, "astropy_surface_rv", fake_astropy_surface_rv)
    monkeypatch.setattr(module, "EarthObserver", FakeEarthObserver)
    monkeypatch.setattr(module, "to_gps", lambda value: 42.0)

    r, v = module.surface_rv(lon=1.0, lat=2.0, elevation=3.0, t="time")
    np.testing.assert_allclose(r, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(v, [0.1, 0.2, 0.3])
    assert astropy_calls == [{"lon": 1.0, "lat": 2.0, "elevation": 3.0, "t": "time"}]

    r_ssapy, v_ssapy = module.surface_rv_ssapy(lon=4.0, lat=5.0, elevation=6.0, t="time", fast=True)
    np.testing.assert_allclose(r_ssapy, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(v_ssapy, [42.0, 1.0, 0.0])


def test_lunar_rv_scalar_and_time_sequence(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.coordinates.lunar_position")

    class FakeTimeValue:
        def __init__(self, gps):
            self.gps = float(gps)

    class FakeMoonBody:
        def position(self, t):
            values = np.asarray(t, dtype=float)
            return np.vstack((values, 2.0 * values, 3.0 * values)) if values.ndim else np.array([values, 2.0 * values, 3.0 * values])

    monkeypatch.setattr(module, "Time", FakeTimeValue)
    monkeypatch.setattr(module, "get_body", lambda name: FakeMoonBody())

    r_scalar, v_scalar = module.get_lunar_rv(10.0)
    np.testing.assert_allclose(r_scalar, [[10.0, 20.0, 30.0]])
    np.testing.assert_allclose(v_scalar, [[1.0, 2.0, 3.0]])

    r_vector, v_vector = module.get_lunar_rv([FakeTimeValue(0.0), FakeTimeValue(1.0), FakeTimeValue(2.0)])
    np.testing.assert_allclose(r_vector, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(v_vector, [[1.0, 2.0, 3.0]] * 3)


def test_j2000_validation_and_time_parsing():
    from ssapy_toolkit.coordinates.j2000_to_gcrf import j2000_to_gcrf
    from astropy.time import Time

    positions = np.eye(3)
    assert j2000_to_gcrf(positions, 0.0).shape == (3, 3)
    assert j2000_to_gcrf(positions, Time("2025-01-01T00:00:00", scale="utc")).shape == (3, 3)
    assert j2000_to_gcrf(positions, "2025-01-01 00:00:00.123Z").shape == (3, 3)
    with pytest.raises(ValueError, match="n x 3"):
        j2000_to_gcrf(np.ones(3), 0.0)
    with pytest.raises(ValueError, match="Invalid obstime"):
        j2000_to_gcrf(positions, "not-a-time")
    with pytest.raises(ValueError, match="obstime must"):
        j2000_to_gcrf(positions, object())


def test_local_equatorial_and_earth_trojan_helpers():
    from ssapy_toolkit.coordinates import local_and_equatorial, local_and_equitorial as local

    assert local_and_equatorial.horizontal_to_equatorial is local.horizontal_to_equatorial
    from ssapy_toolkit.coordinates.earth_trojan_sim import inert2rot, sim_lonlatrad

    assert local.rightasension2hourangle("23:00:00", "01:00:00") == "30:0:0"
    assert local.rightascension2hourangle("23:00:00", "01:00:00") == "30:0:0"
    assert isinstance(local.rightasension2hourangle(15.0, 2.0), str)
    az, alt = local.equatorial_to_horizontal(30.0, 10.0, hour_angle="01:00:00")
    assert np.isfinite(az)
    assert np.isfinite(alt)
    az_south, alt_south = local.equatorial_to_horizontal(-30.0, 10.0, right_ascension="01:00:00", local_time="02:00:00")
    assert np.isfinite(az_south)
    assert np.isfinite(alt_south)
    with pytest.warns(UserWarning, match="Both right_ascension"):
        local.equatorial_to_horizontal(30.0, 10.0, right_ascension="01:00:00", hour_angle="02:00:00")
    with pytest.raises(ValueError, match="must be provided"):
        local.equatorial_to_horizontal(30.0, 10.0)

    hour_angle, dec = local.horizontal_to_equatorial(30.0, 270.0, 45.0)
    assert 0.0 <= hour_angle <= 360.0
    assert -90.0 <= dec <= 90.0

    xrot, yrot = inert2rot(1.0, 0.0, xe=0.0, ye=1.0)
    assert np.isfinite(xrot)
    assert np.isfinite(yrot)
    lon, lat, radius = sim_lonlatrad(2.0, 0.0, 0.0, xe=1.0, ye=0.0, ze=0.0, xs=0.0, ys=1.0, zs=0.0)
    assert 0.0 <= lon < 360.0
    assert np.isfinite(lat)
    assert radius > 0.0


def test_point_mass_accelerations_with_fake_ephemerides(monkeypatch):
    moon = importlib.import_module("ssapy_toolkit.accelerations.accel_moon")
    sun = importlib.import_module("ssapy_toolkit.accelerations.accel_sun")

    class FakeContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeXYZ:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

        def to(self, unit):
            return self

    class FakeCartesian:
        def __init__(self, value):
            self.xyz = FakeXYZ(value)

    class FakeBody:
        def __init__(self, value):
            self.cartesian = FakeCartesian(value)

        def transform_to(self, frame):
            return self

    monkeypatch.setattr(moon, "to_gps", lambda t: 0.0)
    monkeypatch.setattr(sun, "to_gps", lambda t: 0.0)
    monkeypatch.setattr(moon.solar_system_ephemeris, "set", lambda name: FakeContext())
    monkeypatch.setattr(sun.solar_system_ephemeris, "set", lambda name: FakeContext())
    monkeypatch.setattr(moon, "get_body", lambda name, t: FakeBody([10.0, 0.0, 0.0]))
    monkeypatch.setattr(sun, "get_body", lambda name, t: FakeBody([20.0, 0.0, 0.0]))

    np.testing.assert_allclose(moon.accel_point_moon([10.0, 0.0, 0.0], 0.0), np.zeros(3))
    assert moon.accel_point_moon([0.0, 0.0, 0.0], 0.0)[0] > 0.0
    assert sun.accel_point_sun(np.array([0.0, 0.0, 0.0]), 0.0)[0] > 0.0
