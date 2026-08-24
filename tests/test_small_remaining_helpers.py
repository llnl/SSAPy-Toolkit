import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_set_axes_equal_balances_3d_ranges():
    from ssapy_toolkit.plots.set_axes_equal import set_axes_equal

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(5, 6)
    set_axes_equal(ax)
    ranges = [abs(lims[1] - lims[0]) for lims in (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())]
    assert ranges[0] == pytest.approx(ranges[1])
    assert ranges[1] == pytest.approx(ranges[2])
    plt.close(fig)


def test_engine_fuel_gravity_turn_and_finite_burn():
    fuel = importlib.import_module("ssapy_toolkit.engines.fuel_usage")
    gravity = importlib.import_module("ssapy_toolkit.launch.gravity_turn")
    finite = importlib.import_module("ssapy_toolkit.orbital_mechanics.calculate_finite_burn_acceleration")

    positions = np.array([[7_000_000.0, 0.0, 0.0], [7_100_000.0, 0.0, 0.0]])
    used = fuel.estimate_fuel_usage(
        np.array([0.01, 0.02]),
        10.0,
        positions,
        engine="hall_effect_small",
        initial_mass_kg=500.0,
    )
    first_step = 500.0 * 0.01 / (1604.0 * fuel.G0) * 10.0
    second_step = (500.0 - first_step) * 0.02 / (1604.0 * fuel.G0) * 10.0
    assert used == pytest.approx(first_step + second_step)
    assert fuel.estimate_fuel_usage(
        np.array([0.01]),
        10.0,
        positions[:1],
        engine="SPT-100",
        initial_mass_kg=1000.0,
    ) == pytest.approx(1000.0 * 0.01 / (1604.0 * fuel.G0) * 10.0)
    with pytest.raises(KeyError, match="Unknown thruster spec"):
        fuel.estimate_fuel_usage(np.array([0.01]), 1.0, positions[:1], engine="missing", initial_mass_kg=500.0)
    with pytest.raises(ValueError, match="positive"):
        fuel.estimate_fuel_usage(np.array([0.01]), 0.0, positions[:1], engine="hall_effect_small", initial_mass_kg=500.0)
    with pytest.raises(TypeError, match="initial_mass_kg"):
        fuel.estimate_fuel_usage(np.array([0.01]), 1.0, positions[:1], engine="hall_effect_small")
    with pytest.raises(ValueError, match="initial_mass_kg"):
        fuel.estimate_fuel_usage(np.array([0.01]), 1.0, positions[:1], engine="hall_effect_small", initial_mass_kg=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        fuel.estimate_fuel_usage(np.array([-0.01]), 1.0, positions[:1], engine="hall_effect_small", initial_mass_kg=500.0)
    with pytest.raises(ValueError, match="must match"):
        fuel.estimate_fuel_usage(np.array([0.01]), 1.0, positions, engine="hall_effect_small", initial_mass_kg=500.0)
    with pytest.raises(ValueError, match="shape"):
        fuel.estimate_fuel_usage(np.array([0.01]), 1.0, np.ones((1, 2)), engine="hall_effect_small", initial_mass_kg=500.0)

    accel_start = gravity.accel_gravity_turn(np.array([7_000_000.0, 0.0, 0.0]), 0, np.array([0.0, 10.0]), np.array([1.0, 1.0]), turn_time=10.0)
    accel_end = gravity.accel_gravity_turn(np.array([7_000_000.0, 0.0, 0.0]), 1, np.array([0.0, 10.0]), np.array([1.0, 1.0]), turn_time=10.0, launch_az=np.pi / 2)
    assert accel_start[2] > 0.0
    assert accel_end[1] > 0.0

    a_vec, t_burn, t_start, t_end = finite.calculate_finite_burn_acceleration(np.array([3.0, 4.0, 0.0]), 100.0, 0.5)
    np.testing.assert_allclose(a_vec, [0.3, 0.4, 0.0])
    assert t_burn == pytest.approx(10.0)
    assert (t_start, t_end) == pytest.approx((95.0, 105.0))
    with pytest.raises(ValueError, match="zero"):
        finite.calculate_finite_burn_acceleration(np.zeros(3), 0.0, 1.0)


def test_list_files_natural_sort_deduplicates_and_expands_home(tmp_path, monkeypatch):
    from ssapy_toolkit.io.listfiles import list_files

    (tmp_path / "frame10.png").write_text("x")
    (tmp_path / "frame2.png").write_text("x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "frame1.png").write_text("x")
    monkeypatch.setenv("HOME", str(tmp_path))

    files = [path.split("/")[-1] for path in list_files(str(tmp_path / "frame*.png"), str(tmp_path / "frame2.png"))]
    assert files == ["frame2.png", "frame10.png"]
    unsorted_files = list_files(str(tmp_path / "frame*.png"), sort=False)
    assert len(unsorted_files) == 2
    recursive = list_files("~/nested/**/*.png")
    assert recursive[0].endswith("frame1.png")


def test_gamma_heading_helpers_with_fake_transforms(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.orbital_mechanics.gamma_and_heading")
    r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    velocity = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    monkeypatch.setattr(module, "gcrf_to_itrf", lambda r_in, t, v=True: (r, velocity))
    monkeypatch.setattr(module, "v_from_r", lambda r_in, t: velocity)

    gamma = module.calc_gamma(r, np.array([0.0, 1.0]))
    heading = module.calc_heading_itrf(r, velocity)
    gamma2, heading2 = module.calc_gamma_and_heading(r, np.array([0.0, 1.0]))
    gamma3, heading3 = module.calc_gamma_and_heading_itrf(r, np.array([0.0, 1.0]))

    np.testing.assert_allclose(gamma, [0.0, 0.0])
    np.testing.assert_allclose(gamma2, gamma)
    np.testing.assert_allclose(gamma3, gamma)
    assert np.all((0.0 <= heading) & (heading < 360.0))
    np.testing.assert_allclose(heading2, heading)
    np.testing.assert_allclose(heading3, heading)


def test_sky_angle_helpers_with_fake_orbit_and_groundtrack(monkeypatch):
    sky = importlib.import_module("ssapy_toolkit.coordinates.sky")

    class FakeAngle:
        def __init__(self, value):
            self.value = value

        def to(self, unit):
            return self

    class FakeSun:
        ra = FakeAngle(1.0)
        dec = FakeAngle(0.5)

    class FakeOrbit:
        @classmethod
        def fromKeplerianElements(cls, *args, **kwargs):
            return cls()

    class FakePropagator:
        def __init__(self, accel):
            self.accel = accel

    class FakeTime:
        def __init__(self):
            self.gps = np.array([0.0, 5.0, 10.0])

        def __getitem__(self, idx):
            return self.gps[idx]

    monkeypatch.setattr(sky, "get_body", lambda time: FakeSun())
    monkeypatch.setattr(sky, "groundTrack", lambda r, t: (np.array([1.0]), np.array([2.0]), np.array([3.0])))
    monkeypatch.setattr(sky, "Orbit", FakeOrbit)
    monkeypatch.setattr(sky, "SciPyPropagator", FakePropagator)
    monkeypatch.setattr(sky, "AccelKepler", lambda: "accel")
    monkeypatch.setattr(
        sky,
        "rv",
        lambda orbit, t, propagator=None: (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.zeros((3, 3)),
        ),
    )

    assert sky.sun_ra_dec(60000.0) == (1.0, 0.5)
    ra, dec = sky.ra_dec(r=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]), v=np.zeros((2, 3)))
    np.testing.assert_allclose(ra, [0.0, np.pi / 2])
    assert dec.shape == (2,)
    with pytest.raises(ValueError, match="Either provide"):
        sky.ra_dec()
    assert sky.gcrf_to_radec(np.array([0.0, 1.0, 1.0])) == pytest.approx((90.0, 45.0))
    lon, lat, height = sky.gcrf_to_lat_lon(np.zeros((1, 3)), np.array([0.0]))
    np.testing.assert_allclose([lon[0], lat[0], height[0]], [1.0, 2.0, 3.0])
    sim = sky.gcrf_to_sim_geo(np.eye(3), FakeTime(), h=10.0)
    assert sim.shape == (3, 3)
    assert sky.altitude2zenithangle(30.0) == 60.0
    assert sky.zenithangle2altitude(np.pi / 3, deg=False) == pytest.approx(np.pi / 6)


def test_break_plot_line_preserves_containers_and_inserts_nans():
    from ssapy_toolkit.plots.break_plot_lines import break_plot_line

    lon, lat = break_plot_line([170, 179, -179, -170], [0, 1, 2, 3])
    assert isinstance(lon, list)
    assert np.isnan(lon[2])
    assert np.isnan(lat[2])

    lon_arr, lat_arr = break_plot_line(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert isinstance(lon_arr, np.ndarray)
    np.testing.assert_array_equal(lon_arr, [1, 2, 3])
    np.testing.assert_array_equal(lat_arr, [4, 5, 6])

    single_lon, single_lat = break_plot_line((1,), (2,))
    assert isinstance(single_lon, np.ndarray)
    np.testing.assert_array_equal(single_lat, [2])

    with pytest.raises(ValueError, match="same shape"):
        break_plot_line([1, 2], [1])


def test_leapfrog_extra_accel_signatures_and_impact(monkeypatch, capsys):
    module = importlib.import_module("ssapy_toolkit.propagators_orbit.leap_frog")
    monkeypatch.setattr(module, "to_gps", lambda t: np.asarray(t, dtype=float))
    monkeypatch.setattr(module, "build_profile", lambda spec, t: np.zeros_like(np.asarray(t, dtype=float)))
    monkeypatch.setattr(module, "accel_point_earth", lambda r: np.zeros(3))
    monkeypatch.setattr(module, "accel_radial", lambda r, magnitude: np.zeros(3))
    monkeypatch.setattr(module, "accel_velocity", lambda v, thrust_mag: np.zeros(3))
    monkeypatch.setattr(module, "accel_inclination", lambda r, v, magnitude: np.zeros(3))

    def accel_rvt(r, v, t):
        return np.array([1.0, 0.0, 0.0])

    def accel_rt(r, t):
        return np.array([0.0, 1.0, 0.0])

    def accel_rv(r, v):
        return np.array([0.0, 0.0, 1.0])

    def accel_r(r):
        return np.array([1.0, 1.0, 1.0])

    r, v = module.leapfrog(
        [7_000_000.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        accels=[accel_rvt, accel_rt, accel_rv, accel_r],
    )
    assert r.shape == (3, 3)
    assert v[-1, 0] > 0.0
    assert v[-1, 1] > 0.0
    assert v[-1, 2] > 0.0

    r2, v2 = module.leapfrog(
        [7_000_000.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0],
        accels=accel_rvt,
    )
    assert r2.shape == (2, 3)

    impacted_r, impacted_v = module.leapfrog(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0],
        stop_altitude_m=0.0,
        verbose=True,
    )
    assert impacted_r.shape == impacted_v.shape == (1, 3)
    assert "Impact" in capsys.readouterr().out
