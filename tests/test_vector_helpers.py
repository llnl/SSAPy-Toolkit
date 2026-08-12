import numpy as np
import pytest
import ssapy_toolkit.plots as plots

from ssapy_toolkit.vectors import (
    angle_between_vectors,
    einsum_norm,
    extend_vector,
    getAngle,
    norm,
    normSq,
    normed,
    perpendicular_vectors,
    points_on_circle,
    rotate_points_3d,
    rotate_vector,
    rotation_matrix_from_vectors,
    unit_vector,
)


def test_basic_vector_norm_helpers():
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])

    np.testing.assert_allclose(unit_vector(np.array([0.0, 3.0, 4.0])), [0.0, 0.6, 0.8])
    np.testing.assert_allclose(norm(vectors), [5.0, 5.0])
    np.testing.assert_allclose(normSq(vectors), [25.0, 25.0])
    np.testing.assert_allclose(einsum_norm(vectors, indices="ij,ij->i"), [5.0, 5.0])
    np.testing.assert_allclose(normed(vectors), [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])


def test_extend_vector_and_angles():
    np.testing.assert_allclose(extend_vector(np.array([3.0, 4.0, 0.0]), 5.0), [6.0, 8.0, 0.0])

    with pytest.raises(ValueError, match="zero vector"):
        extend_vector(np.zeros(3), 1.0)

    assert np.isclose(angle_between_vectors([1, 0, 0], [0, 1, 0]), np.pi / 2)
    np.testing.assert_allclose(
        getAngle(
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.zeros((2, 3)),
            np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        ),
        [np.pi / 2, np.pi],
    )


def test_rotation_helpers_preserve_lengths_and_align_vectors():
    matrix = rotation_matrix_from_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)

    identity = rotation_matrix_from_vectors(np.array([2.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)

    opposite = rotation_matrix_from_vectors(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(opposite @ np.array([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(opposite.T @ opposite, np.eye(3), atol=1e-12)

    with pytest.raises(ValueError, match="zero-length"):
        rotation_matrix_from_vectors(np.zeros(3), np.array([1.0, 0.0, 0.0]))

    rotated = rotate_vector(np.array([0.0, 0.0, 1.0]), theta=90.0, phi=0.0)
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.isclose(np.linalg.norm(rotated), 1.0)

    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(rotate_points_3d(points, axis=np.array([0.0, 0.0, 1.0]), theta=np.pi / 2), expected, atol=1e-12)


def test_rotate_vector_save_path_and_alternate_perpendicular_branch(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(plots, "save_plot", lambda fig, save_path=None, **kwargs: saved.append(save_path))

    rotated = rotate_vector(np.array([1.0, 1.0, 0.0]), theta=20.0, phi=30.0, save_path=tmp_path / "vector.png")

    assert saved == [tmp_path / "vector.png"]
    assert np.isclose(np.linalg.norm(rotated), 1.0)


def test_perpendicular_vectors_and_circle_points():
    vector = np.array([1.0, 2.0, 3.0])
    u, w = perpendicular_vectors(vector)
    assert np.isclose(np.dot(u, vector), 0.0)
    assert np.isclose(np.dot(w, vector), 0.0)
    assert np.isclose(np.dot(u, w), 0.0)

    u_axis, w_axis = perpendicular_vectors(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(u_axis, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(w_axis, [0.0, -1.0, 0.0])

    with pytest.raises(ValueError, match="zero vector"):
        perpendicular_vectors(np.zeros(3))

    center = np.array([1.0, 2.0, 3.0])
    points = points_on_circle(center, np.array([0.0, 0.0, 2.0]), rad=2.0, num_points=4)
    assert points.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(points - center, axis=1), 2.0)
    np.testing.assert_allclose(points[:, 2], 3.0)

    off_axis = points_on_circle(center, np.array([1.0, 1.0, 1.0]), rad=1.5, num_points=5)
    assert off_axis.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(off_axis - center, axis=1), 1.5, atol=1e-12)

    with pytest.raises(ValueError, match="must not be the zero vector"):
        points_on_circle(center, np.zeros(3), rad=1.0)

    with pytest.raises(ValueError, match="must not be the zero vector"):
        points_on_circle(center, np.zeros(3), rad=1.0)
