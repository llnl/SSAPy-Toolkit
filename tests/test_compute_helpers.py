import numpy as np
import pytest

from ssapy_toolkit.compute.calculate_errors import calculate_errors
from ssapy_toolkit.compute.fft import FFT, FFTP
from ssapy_toolkit.compute.generate_sphere_of_vectors import _summarize, generate_sphere_vectors
from ssapy_toolkit.compute.mengos import megno


def test_calculate_errors_returns_percentile_offsets_and_median():
    err, medians = calculate_errors(np.array([5.0, 1.0, 3.0, 2.0, 4.0]), CI=0.2)

    assert medians == [3.0]
    assert err == [1.0, 2.0]


def test_fft_helpers_return_expected_peak_and_periods():
    data = np.array([0.0, 1.0, 0.0, -1.0])

    frequencies, amplitudes = FFT(data, time_between_samples=0.5)
    periods, period_amplitudes = FFTP(data, time_between_samples=0.5)

    np.testing.assert_allclose(frequencies, [0.0, 1.0])
    np.testing.assert_allclose(amplitudes, [0.0, 2.0])
    np.testing.assert_allclose(period_amplitudes, amplitudes)
    assert periods[0] == pytest.approx(len(data) * 0.5)
    assert periods[1] == pytest.approx(1.0)


def test_generate_sphere_vectors_distribution_modes_and_summary(capsys):
    uniform = generate_sphere_vectors(8, magnitude=3.0, seed=1, distribution="uniform")
    random = generate_sphere_vectors(8, magnitude=2.0, seed=1, distribution="random")

    assert uniform.shape == (8, 3)
    assert random.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(uniform, axis=1), 3.0)
    np.testing.assert_allclose(np.linalg.norm(random, axis=1), 2.0)
    np.testing.assert_allclose(generate_sphere_vectors(0, magnitude=5.0), np.zeros((0, 3)))

    with pytest.raises(ValueError, match="distribution"):
        generate_sphere_vectors(1, magnitude=1.0, distribution="bad")

    _summarize(uniform[:2], label="sample")
    captured = capsys.readouterr().out
    assert "sample: shape=(2, 3)" in captured
    assert "first few rows" in captured

    _summarize(np.zeros((0, 3)), label="empty")
    assert "empty: shape=(0, 3)" in capsys.readouterr().out


def test_megno_uses_perturbed_state_norms(monkeypatch):
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape))
    states = np.zeros((4, 3))

    value = megno(states)


    assert np.isfinite(value)
    assert value < 0.0
