from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import ncx2

from ssapy_toolkit.ssa import (
    ClosestApproach,
    ConjunctionCandidate,
    coarse_conjunction_screen,
    encounter_frame,
    probability_of_collision,
    refine_closest_approach,
    relative_encounter_covariance,
)


def _trajectory(times, positions, velocities=None):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if velocities is None:
        velocities = np.gradient(positions, times, axis=0)
    return SimpleNamespace(t=times, r=positions, v=np.asarray(velocities, dtype=float))


def _hermite_position(position_start, position_end, velocity_start, velocity_end, time):
    h00 = 2.0 * time**3 - 3.0 * time**2 + 1.0
    h10 = time**3 - 2.0 * time**2 + time
    h01 = -2.0 * time**3 + 3.0 * time**2
    h11 = time**3 - time**2
    return (
        h00 * np.asarray(position_start)
        + h10 * np.asarray(velocity_start)
        + h01 * np.asarray(position_end)
        + h11 * np.asarray(velocity_end)
    )


def test_screen_and_refine_constant_relative_motion():
    first = _trajectory([0.0, 10.0], [[0, 0, 0], [0, 0, 0]])
    second = _trajectory([0.0, 10.0], [[5, 0, 0], [-5, 0, 0]], [[-1, 0, 0], [-1, 0, 0]])
    candidates = coarse_conjunction_screen(first, second, 0.1)
    assert len(candidates) == 1
    assert candidates[0].t_min == pytest.approx(5.0)
    refined = refine_closest_approach(first, second, candidates[0])
    assert isinstance(refined, ClosestApproach)
    assert refined.tca == pytest.approx(5.0, abs=1e-8)
    assert refined.miss_distance == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(refined.relative_position, 0.0, atol=1e-8)
    np.testing.assert_allclose(refined.relative_velocity, [-1, 0, 0])


def test_screen_endpoint_no_overlap_and_invalid_trajectories():
    first = _trajectory([0.0, 1.0], [[0, 0, 0], [0, 0, 0]])
    endpoint = _trajectory([0.0, 1.0], [[1, 0, 0], [2, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    assert coarse_conjunction_screen(first, endpoint, 1.0)[0].t_min == pytest.approx(0.0)
    outside = _trajectory([2.0, 3.0], [[0, 0, 0], [0, 0, 0]])
    assert coarse_conjunction_screen(first, outside, 10.0) == ()
    exact = _trajectory([1.0, 2.0], [[8, 0, 0], [7, 0, 0]], [[-1, 0, 0], [-1, 0, 0]])
    singleton_candidate = coarse_conjunction_screen(first, exact, 8.0)
    assert len(singleton_candidate) == 1
    assert singleton_candidate[0].bracket == (1.0, 1.0)
    assert refine_closest_approach(first, exact, singleton_candidate[0]).tca == pytest.approx(1.0)
    assert coarse_conjunction_screen(first, exact, 7.9) == ()
    with pytest.raises(ValueError, match="strictly increasing"):
        coarse_conjunction_screen(
            SimpleNamespace(t=[0, 0], r=[[0, 0, 0], [1, 0, 0]], v=[[1, 0, 0], [1, 0, 0]]),
            first,
            1.0,
        )
    with pytest.raises(ValueError, match="shape"):
        coarse_conjunction_screen(SimpleNamespace(t=[0, 1], r=[[0, 0]], v=[[0, 0, 0]]), first, 1.0)


def test_encounter_frame_and_zero_speed():
    basis = encounter_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(basis, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(basis.T @ basis, np.eye(2))
    with pytest.raises(ValueError, match="relative speed"):
        encounter_frame([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_relative_encounter_covariance_projection_and_cross_covariance():
    basis = np.eye(3)[:, :2]
    covariance_a = np.diag([1.0, 2.0, 3.0])
    covariance_b = np.diag([4.0, 5.0, 6.0])
    np.testing.assert_allclose(
        relative_encounter_covariance(covariance_a, covariance_b, basis), np.diag([5.0, 7.0])
    )
    np.testing.assert_allclose(
        relative_encounter_covariance(
            covariance_a, covariance_b, basis, cross_covariance=np.diag([0.5, 0.5, 0.5])
        ),
        np.diag([4.0, 6.0]),
    )
    with pytest.raises(ValueError, match="orthonormal"):
        relative_encounter_covariance(covariance_a, covariance_b, np.ones((3, 2)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        relative_encounter_covariance(-np.eye(3), covariance_b, basis)
    with pytest.raises(ValueError, match="joint covariance"):
        relative_encounter_covariance(covariance_a, covariance_b, basis, cross_covariance=3.0 * np.eye(3))


def test_refinement_splits_internal_knots_and_preserves_b_minus_a_sign():
    first = _trajectory(
        [0.0, 1.0, 2.0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0]] * 3
    )
    second = _trajectory(
        [0.0, 1.0, 2.0], [[3, 0, 0], [1, 0, 0], [3, 0, 0]], [[-2, 0, 0], [-2, 0, 0], [3, 0, 0]]
    )
    refined = refine_closest_approach(first, second, (0.0, 2.0))
    assert 1.0 < refined.tca < 1.3
    assert refined.miss_distance < 1.0
    np.testing.assert_allclose(refined.relative_velocity[1:], 0.0, atol=1e-7)
    with pytest.raises(ValueError, match="two times"):
        refine_closest_approach(first, second, (0.0, 1.0, 2.0))


def test_refinement_finds_global_stationary_minimum_on_one_cubic_segment():
    first = _trajectory([0.0, 1.0], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]])
    position_start = [2.73956049, -0.6112156, 3.5859792]
    position_end = [1.97368029, -4.05822652, 4.75622352]
    velocity_start = [10.44558808, 11.44257221, -14.87545469]
    velocity_end = [-1.98456248, -5.16807903, 17.07059955]
    second = _trajectory(
        [0.0, 1.0], [position_start, position_end], [velocity_start, velocity_end]
    )
    refined = refine_closest_approach(first, second, (0.0, 1.0))
    dense_times = np.linspace(0.0, 1.0, 100001)
    dense_positions = np.array(
        [
            _hermite_position(position_start, position_end, velocity_start, velocity_end, time)
            for time in dense_times
        ]
    )
    dense_index = np.argmin(np.linalg.norm(dense_positions, axis=1))
    assert refined.tca == pytest.approx(dense_times[dense_index], abs=2e-5)
    assert refined.miss_distance == pytest.approx(np.linalg.norm(dense_positions[dense_index]), abs=2e-8)


def test_candidate_validation():
    with pytest.raises(ValueError, match="inside"):
        ConjunctionCandidate(0.0, 1.0, 2.0, 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        ConjunctionCandidate(0.0, 1.0, 0.5, -1.0)


def test_probability_centered_offset_and_anisotropic_rotation():
    sigma = 2.0
    radius = 1.0
    centered = probability_of_collision([0.0, 0.0], np.eye(2) * sigma**2, radius)
    assert centered == pytest.approx(1.0 - np.exp(-radius**2 / (2.0 * sigma**2)), abs=1e-12)

    mean = np.array([1.5, 0.0])
    offset = probability_of_collision(mean, np.eye(2) * sigma**2, radius)
    oracle = ncx2.cdf(radius**2 / sigma**2, 2, np.dot(mean, mean) / sigma**2)
    assert offset == pytest.approx(oracle, abs=1e-10)

    covariance = np.diag([1.0, 4.0])
    mean = np.array([0.4, -0.7])
    anisotropic = probability_of_collision(mean, covariance, 1.2)
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = probability_of_collision(rotation @ mean, rotation @ covariance @ rotation.T, 1.2)
    assert anisotropic == pytest.approx(rotated, abs=1e-10)


def test_probability_validation_and_refinement_validation():
    with pytest.raises(ValueError, match="positive-definite"):
        probability_of_collision([0, 0], [[1, 0], [0, 0]], 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        probability_of_collision([0, 0], np.eye(2), -1.0)
    first = _trajectory([0.0, 1.0], [[0, 0, 0], [0, 0, 0]])
    second = _trajectory([0.0, 1.0], [[1, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="within"):
        refine_closest_approach(first, second, (-1.0, 0.5))
