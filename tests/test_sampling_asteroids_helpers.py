import random

import numpy as np
import pytest

from ssapy_toolkit import asteroids
from ssapy_toolkit.compute import sampling


def test_asteroid_size_magnitude_and_filter_conversions():
    H = np.array([17.0, 20.0])
    radius = asteroids.radius_from_H_albedo(H, albedo=0.1)
    np.testing.assert_allclose(asteroids.H_mag(radius, albedo=0.1), H)

    mags = np.full(12, 20.0)
    filters = np.array(list("uuggrriizzyy"))
    types = np.array([0, 1] * 6)
    expected_corrections = np.array([-1.614, -1.927, -0.302, -0.395, 0.172, 0.255, 0.291, 0.455, 0.298, 0.401, 0.303, 0.406])
    np.testing.assert_allclose(asteroids.johnsonV_to_lsst_array(mags, filters, types), mags - expected_corrections)

    ztf_filters = np.array([1, 1, 2, 2, 3, 3])
    ztf_types = np.array([0, 1, 0, 1, 0, 1])
    ztf_expected = np.array([-0.302, -0.395, 0.172, 0.255, 0.291, 0.455])
    np.testing.assert_allclose(asteroids.johnsonV_to_ztf_array(np.full(6, 19.0), ztf_filters, ztf_types), 19.0 - ztf_expected)

    assert np.isclose(asteroids.granvik_low_slope(10.0), 0.3034 * 10.0 - 3.491)
    assert np.isclose(asteroids.granvik_high_slope(20.0), 0.7235 * 20.0 - 13.12)


def test_asteroid_random_generators_are_shape_bounded(monkeypatch):
    np.random.seed(1)
    albedo, ast_type = asteroids.get_albedo_array(num=5)
    assert albedo.shape == ast_type.shape == (5,)
    assert np.all((albedo >= 0.0) & (albedo <= 1.0))
    assert set(ast_type).issubset({0, 1})

    np.random.seed(2)
    H = asteroids.get_neo_H_mag_array(num=4, upper_mag=25, min_mag=20)
    assert H.shape == (4,)
    assert np.all((H >= 20.0) & (H <= 25.0))

    monkeypatch.setattr(asteroids, "get_albedo_array", lambda num: (np.full(num, 0.25), np.ones(num, dtype=int)))
    monkeypatch.setattr(asteroids, "get_neo_H_mag_array", lambda num, upper_mag, min_mag: np.linspace(min_mag, upper_mag, num))
    eta = asteroids.get_eta_radius_albedo_H_array(num=3, upper_mag=13, min_mag=11)
    assert set(eta) == {"radius", "albedo", "type", "H"}
    np.testing.assert_allclose(eta["albedo"], 0.25)
    np.testing.assert_array_equal(eta["type"], [1, 1, 1])
    np.testing.assert_allclose(eta["H"], [11, 12, 13])


def test_sampling_basic_distribution_helpers(tmp_path, monkeypatch):
    np.random.seed(0)
    random.seed(0)
    assert sampling.sample_from_sequence(["a", "b", "c"], 2).shape == (2,)
    assert 2.0 <= sampling.rand_num(2.0, 3.0) < 3.0

    values = [1, 2, 3]
    sampling.shuffle(values)
    assert sorted(values) == [1, 2, 3]

    assert sampling.random_arr(0, 5, size=(2, 2), dtype="int64").dtype == np.int64
    assert sampling.random_arr(0, 1, size=(2,), dtype="float32").dtype == np.float32
    assert isinstance(sampling.uniform_scalar(1, 2), float)
    assert sampling.uniform_array(0, 1, size=(2, 3)).shape == (2, 3)
    assert isinstance(sampling.normal_scalar(0, 1), float)
    assert sampling.normal_array(0, 1, size=(2,), dtype="float32").dtype == np.float32
    assert sampling.lognormal_array(size=(2,)).shape == (2,)
    assert sampling.exponential_array(size=(2,)).shape == (2,)
    assert sampling.poisson_array(lam=2, size=(2,)).shape == (2,)
    assert sampling.binomial_array(n=3, p=0.5, size=(2,)).shape == (2,)
    assert sampling.dirichlet_array([1, 2, 3], size=2).shape == (2, 3)
    assert sampling.multivariate_normal_array([0, 0], np.eye(2), size=2).shape == (2, 2)

    sig_path = tmp_path / "sigmas.npy"
    sigmas = sampling.get_sigmas(n=3, path=sig_path)
    assert sigmas.shape == (3, 6)
    assert sig_path.exists()
    loaded = sampling.get_sigmas(n=3, path=sig_path)
    np.testing.assert_array_equal(loaded, sigmas)
    np.save(sig_path, np.zeros((1, 6)))
    assert sampling.get_sigmas(n=3, path=sig_path).shape == (3, 6)
    sig_path.write_text("not npy")
    assert sampling.get_sigmas(n=2, path=sig_path).shape == (2, 6)

    env_dir = tmp_path / "cache"
    monkeypatch.setenv(sampling.ENV_VAR, str(env_dir))
    assert sampling.get_sigmas(n=1).shape == (1, 6)


def test_sampling_3d_offsets_and_state_perturbations():
    assert np.all(sampling._sample_3d_offset(0.0) == 0.0)
    for dist in ["uniform", "normal", "gaussian", "shell", "surface", "laplace"]:
        rng = np.random.default_rng(123)
        offset = sampling._sample_3d_offset(2.0, distribution=dist, rng=rng)
        assert offset.shape == (3,)
        if dist in {"shell", "surface"}:
            assert np.isclose(np.linalg.norm(offset), 2.0)

    with pytest.raises(ValueError, match="Unknown distribution"):
        sampling._sample_3d_offset(1.0, distribution="bad", rng=np.random.default_rng(1))

    rng = np.random.default_rng(5)
    r, v = sampling.perturb_state_3d(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.2, 0.3]),
        pos_scale=0.0,
        vel_scale=0.0,
        rng=rng,
    )
    np.testing.assert_array_equal(r, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(v, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="3-vectors"):
        sampling.perturb_state_3d([1, 2], [1, 2, 3])
