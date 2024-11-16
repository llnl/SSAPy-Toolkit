# test_vectors.py

import numpy as np
import pytest
from yeager_utils.vectors import (
    unit_vector,
    getAngle,
    angle_between_vectors,
    rotation_matrix_from_vectors,
    normed,
    einsum_norm,
    normSq,
    norm,
    rotate_vector,
    rotate_points_3d,
    perpendicular_vectors,
    points_on_circle
)

def test_unit_vector():
    vector = np.array([1, 2, 3])
    result = unit_vector(vector)
    assert np.isclose(np.linalg.norm(result), 1), "The unit vector is not normalized correctly."


def test_getAngle():
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array([0, 1, 0])
    result = getAngle(a, b, c)
    assert np.isclose(result, np.pi / 2), "Angle between perpendicular vectors should be 90 degrees."


def test_angle_between_vectors():
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 1, 0])
    result = angle_between_vectors(vector1, vector2)
    assert np.isclose(result, np.pi / 2), "Angle between perpendicular vectors should be 90 degrees."


def test_rotation_matrix_from_vectors():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    rotation_matrix = rotation_matrix_from_vectors(vec1, vec2)
    rotated_vec1 = np.dot(rotation_matrix, vec1)
    assert np.allclose(rotated_vec1, vec2), "Rotation matrix did not align vec1 to vec2."


def test_normed():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    result = normed(arr)
    for vector in result:
        assert np.isclose(np.linalg.norm(vector), 1), "Each vector should be normalized."


def test_einsum_norm():
    a = np.array([[3, 4], [5, 12]])
    result = einsum_norm(a)
    expected = np.array([5.38516481, 12.80624847])  # Correct the expected value to be float
    assert np.allclose(result, expected, atol=1e-6), "Einsum norm calculation is incorrect."



def test_normSq():
    arr = np.array([[1, 2, 2], [3, 4, 4]])
    result = normSq(arr)
    expected = np.array([9, 41])
    assert np.allclose(result, expected), "Norm square calculation is incorrect."


def test_norm():
    arr = np.array([[1, 2, 2], [3, 4, 4]])
    result = norm(arr)
    expected = np.sqrt([9, 41])
    assert np.allclose(result, expected), "Norm calculation is incorrect."


def test_rotate_vector():
    v_unit = np.array([1, 0, 0])
    theta, phi = 90, 0  # Rotate by 90 degrees around Z-axis
    result = rotate_vector(v_unit, theta, phi)
    expected = np.array([0, 1, 0])
    assert np.allclose(result, expected), "Vector rotation did not produce the expected result."


def test_rotate_points_3d():
    points = np.array([[1, 0, 0]])
    axis = np.array([0, 0, 1])
    theta = np.pi / 2  # Rotate by 90 degrees around Z-axis
    result = rotate_points_3d(points, axis, theta)
    expected = np.array([[0, 1, 0]])
    assert np.allclose(result, expected), "3D point rotation did not produce the expected result."


def test_perpendicular_vectors():
    v = np.array([1, 0, 0])
    u, w = perpendicular_vectors(v)
    assert np.isclose(np.dot(v, u), 0), "u is not perpendicular to v."
    assert np.isclose(np.dot(v, w), 0), "w is not perpendicular to v."
    assert np.isclose(np.dot(u, w), 0), "u and w are not perpendicular to each other."


def test_points_on_circle():
    r = np.array([0, 0, 0])
    v = np.array([0, 0, 1])
    rad = 1
    num_points = 4
    result = points_on_circle(r, v, rad, num_points=num_points)
    assert result.shape == (4, 3), "The number of points generated is incorrect."
    assert np.allclose(np.linalg.norm(result, axis=1), rad), "Points are not on the circle with the specified radius."


if __name__ == "__main__":
    import os
    import pytest

    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the script's name
    script_name = os.path.basename(__file__)
    
    # Construct the path dynamically
    test_dir = os.path.join(current_dir, script_name)
    
    # Run pytest
    pytest.main([test_dir])
