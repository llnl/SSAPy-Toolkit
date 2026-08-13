import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU, MOON_RADIUS, RGEO
from ssapy_toolkit.orbital_mechanics import keplerian


def test_hkoe_period_longitude_and_anomaly_helpers(capsys):
    np.testing.assert_allclose(keplerian.hkoe([RGEO, 0.1, 30.0, 40.0, 50.0, 60.0]), [RGEO, 0.1, np.pi / 6, np.deg2rad(40), np.deg2rad(50), np.pi / 3])
    np.testing.assert_allclose(keplerian.hkoe(RGEO, 0.1, 30.0, 40.0, 50.0, 60.0), [RGEO, 0.1, np.pi / 6, np.deg2rad(40), np.deg2rad(50), np.pi / 3])

    with pytest.raises(ValueError, match="6-element"):
        keplerian.hkoe([1, 2, 3])
    with pytest.raises(ValueError, match="Must provide"):
        keplerian.hkoe(RGEO, 0.1)

    assert np.isclose(keplerian.period(RGEO), 2 * np.pi * np.sqrt(RGEO**3 / EARTH_MU))
    assert keplerian.mean_longitude(1.0, 2.0, 3.0) == 6.0
    assert np.isclose(keplerian.true_anomaly(eccentricity=0.0, eccentric_anomaly=0.5), 0.5)
    assert np.isfinite(keplerian.true_anomaly(eccentricity=0.1, mean_anomaly=0.5))
    assert np.isclose(keplerian.true_anomaly(true_longitude=3.0, longitude_of_ascending_node=1.0, argument_of_periapsis=0.5), 1.5)
    assert keplerian.true_anomaly() is None
    assert "Not enough information" in capsys.readouterr().out


def test_kepler_state_conversions_for_circular_equatorial_case():
    r, v = keplerian.kepler_to_state(a=RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, nu=0.0)
    np.testing.assert_allclose(r, [RGEO, 0.0, 0.0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(v, [0.0, np.sqrt(EARTH_MU / RGEO), 0.0], rtol=1e-12)

    r_loop, v_loop = keplerian.kepler_to_state_loop(a=RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, nu=0.0)
    np.testing.assert_allclose(r_loop, r)
    np.testing.assert_allclose(v_loop, v)

    r_many, v_many = keplerian.kepler_to_state_loop(
        a=np.array([RGEO, RGEO * 1.1]),
        e=np.array([0.0, 0.1]),
        i=np.array([0.0, 0.1]),
        pa=np.array([0.0, 0.2]),
        raan=np.array([0.0, 0.3]),
        nu=np.array([0.0, 0.4]),
    )
    assert r_many.shape == v_many.shape == (2, 3)

    with pytest.raises(ValueError, match="Semi-major"):
        keplerian.kepler_to_state_loop(a=-1.0)
    with pytest.raises(ValueError, match="Eccentricity"):
        keplerian.kepler_to_state_loop(a=RGEO, e=1.0)


def test_apsis_and_velocity_formula_helpers():
    rp = 7_000_000.0
    ra = 42_000_000.0
    a = (rp + ra) / 2.0
    e = (ra - rp) / (ra + rp)

    assert keplerian.a_from_periap(rp, ra) == a
    assert keplerian.e_from_periap(rp, ra) == e
    assert keplerian.ae_from_periap(rp, ra) == (a, e)
    assert keplerian.periapsis(a, e) == rp
    assert keplerian.apoapsis(a, e) == ra
    assert keplerian.peri_apo_from_rv(rp, ra) == {"a": a, "e": e}
    assert keplerian.apapsis_from_a_rp(a, rp) == ra
    assert np.isclose(keplerian.vcircular(rp, mu_=EARTH_MU), np.sqrt(EARTH_MU / rp))
    assert np.isclose(keplerian.vis_viva(a, rp, EARTH_MU), np.sqrt(EARTH_MU * (2.0 / rp - 1.0 / a)))
    assert np.isclose(keplerian.v_periapsis(a, rp, EARTH_MU), keplerian.vis_viva(a, rp, EARTH_MU))


def test_parametric_and_chance_radius_helpers():
    x, y, z = keplerian.kepler_to_parametric(a=2.0, e=0.0, i=0.0, omega=0.0, pa=0.0, theta=90.0)
    assert np.isclose(x, 0.0, atol=1e-12)
    assert np.isclose(y, 2.0)
    assert np.isclose(z, 0.0)

    v = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert keplerian.get_chance_radius(v, time_step=2.0) == 10.0 * 2.0 * 4.0 + 2.0 * MOON_RADIUS


def test_rebound_orbital_elements_with_fake_rebound(monkeypatch):
    class FakeTime:
        def __init__(self, value, format="utc"):
            self.jd = 2451545.0 if not str(value).startswith("JD") else str(value)

    class FakeParticle:
        def __init__(self, idx):
            self.a = 1.0 + idx
            self.e = 0.01 * idx
            self.inc = 0.1 * idx
            self.theta = 0.2 * idx
            self.omega = 0.3 * idx
            self.Omega = 0.4 * idx
            self.f = 0.5 * idx
            self.pomega = 0.6 * idx
            self.l = 0.7 * idx
            self.M = 0.8 * idx

    class FakeSimulation:
        def __init__(self):
            self.particles = {hash(idx): FakeParticle(idx) for idx in range(9)}
            self.added = []

        def add(self, *args, **kwargs):
            self.added.append((args, kwargs))

        def move_to_com(self):
            self.moved = True

    monkeypatch.setattr(keplerian, "Time", FakeTime)
    monkeypatch.setattr(keplerian.rebound, "Simulation", FakeSimulation)
    monkeypatch.setattr(keplerian.rebound, "M_to_E", lambda e, mean: mean + e)

    earth = keplerian.rebound_orbital_elements("earth", time="2000-01-01")
    assert earth["a"] == 4.0
    assert earth["eccentric_anomaly"] == pytest.approx(2.4 + 0.03)

    all_planets = keplerian.rebound_orbital_elements("all", time="JD2451545.0", format="jd")
    assert set(all_planets) == {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}
    assert all_planets["neptune"]["a"] == 9.0
    with pytest.raises(ValueError, match="Unknown planet"):
        keplerian.rebound_orbital_elements("pluto")


def test_j2000_and_orbital_element_calculators():
    earth = keplerian.j2000_orbitals("earth", Teph=2451545.0 + 36525.0)
    assert set(earth) == {"a", "e", "i", "mean_longitude", "longitude_of_perihelion", "longitude_of_the_ascending_node"}
    assert 0.0 <= earth["mean_longitude"] < 360.0
    with pytest.raises(ValueError, match="Unknown planet"):
        keplerian.j2000_orbitals("pluto")

    elements = keplerian.hkoe(RGEO, 0.1, 30.0, 40.0, 50.0, 60.0)
    r, v = keplerian.kepler_to_state(*elements)
    a, e, inc, pa, raan, nu = keplerian.state_to_kepler(r, v)
    assert np.isfinite([a, e, inc, pa, raan, nu]).all()
    assert a > 0.0

    r2, v2 = keplerian.kepler_to_state(*keplerian.hkoe(RGEO * 1.1, 0.2, 50.0, 20.0, 100.0, 210.0))
    calculated = keplerian.calculate_orbital_elements(np.vstack([r, r2]), np.vstack([v, v2]))
    assert calculated["a"].shape == (2,)
    assert np.all(calculated["e"] > 0.0)
    apsides = keplerian.peri_apo_apsis_from_rv(r, v)
    assert apsides["apoapsis"] > apsides["periapsis"]


def test_apsis_validation_errors():
    with pytest.raises(ValueError, match="positive"):
        keplerian.a_from_periap(0.0, 1.0)
    with pytest.raises(ValueError, match=">= perigee"):
        keplerian.a_from_periap(2.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        keplerian.e_from_periap(1.0, 0.0)
    with pytest.raises(ValueError, match=">= perigee"):
        keplerian.e_from_periap(2.0, 1.0)
