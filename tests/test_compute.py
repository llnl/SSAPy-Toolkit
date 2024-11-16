import pytest
import numpy as np
from yeager_utils.compute import (
    find_smallest_bounding_cube, 
    calculate_errors, 
    FFT, 
    FFTP, 
    proper_motion, 
    getAngle, 
    moon_shine, 
    earth_shine, 
    sun_shine, 
    calc_M_v, 
    M_v_lambertian, 
    megno, 
    orbital_period
)


# Test find_smallest_bounding_cube
def test_find_smallest_bounding_cube():
    coords = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
    lower, upper = find_smallest_bounding_cube(coords)
    assert np.all(lower <= upper)

# Test calculate_errors
def test_calculate_errors():
    data = np.array([1, 2, 3, 4, 5])
    err, median = calculate_errors(data)
    assert len(err) == 2  # Should return two errors (lower and upper)
    assert np.isclose(np.median(data), median[0])

# Test FFT
def test_FFT():
    data = np.array([1, 2, 3, 4, 5])
    f, Y = FFT(data)
    assert len(f) == len(Y)

# Test FFTP
def test_FFTP():
    data = np.array([1, 2, 3, 4, 5])
    Tp, Y = FFTP(data)
    assert len(Tp) == len(Y)

# Test proper_motion
def test_proper_motion():
    x, y, z = 1, 2, 3
    vx, vy, vz = 0.1, 0.2, 0.3
    pm = proper_motion(x, y, z, vx, vy, vz)
    assert pm > 0

# Test getAngle
def test_getAngle():
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array([0, 1, 0])
    angle = getAngle(a, b, c)
    assert np.isclose(angle, np.pi / 2)

# Test calc_M_v
def test_calc_M_v():
    r_sat = np.array([[-1, 0, 0]])
    r_earth = np.array([[0, 0, 0]])
    r_sun = np.array([[1, 0, 0]])
    Mag_v = calc_M_v(r_sat, r_earth, r_sun, radius=0.4)
    Mag_v2 = calc_M_v(r_sat, r_earth, r_sun, radius=0.4, r_moon=[0.5, 0, 0])
    # Assert that Mag_v is a numpy.ndarray
    assert isinstance(Mag_v, np.ndarray), f"Expected Mag_v to be a numpy.ndarray, got {type(Mag_v)}"
    
    # Assert that all elements in Mag_v are finite and of floating point type
    assert np.all(np.isfinite(Mag_v)) and np.issubdtype(Mag_v.dtype, np.floating), \
        f"Expected all elements in Mag_v to be of floating-point type, got {Mag_v.dtype}"

    assert np.all(Mag_v2 < Mag_v), f"Expected Mag_v2 to be greater than Mag_v, got Mag_v2={Mag_v2}, Mag_v={Mag_v}"


# Test megno
def test_megno():
    r = np.random.rand(10, 3)  # 10 random states
    value = megno(r)
    
    # Assert that the result is a float
    assert isinstance(value, float), f"Expected value to be a float, got {type(value)}"


# Test orbital_period
def test_orbital_period():
    a = 1.0  # Semi-major axis in AU
    period = orbital_period(a)
    assert period > 0


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
