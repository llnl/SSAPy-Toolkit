from ..constants import EARTH_MASS, EARTH_RADIUS, EARTH_MU
from ..engines import thrusters
import numpy as np
from astropy import units as u

def estimate_fuel_usage(
    accels: np.ndarray,
    dt: float,
    positions: np.ndarray,
    engine: str = "Mira",
) -> float:
    """
    Estimate total propellant mass used given an acceleration profile,
    with gravity computed at each time step based on spacecraft position.

    Parameters
    ----------
    accels : np.ndarray
        Array of acceleration magnitudes (m/s²) at each time step.
    dt : float
        Time step duration (s).
    positions : np.ndarray
        Array of position vectors (x, y, z) in meters, shape (n, 3), in Earth-centered inertial frame.
    engine : str, optional
        Name of the thruster or spacecraft to use (must be in engines.thrusters). Default is "Mira".

    Returns
    -------
    total_fuel : float
        Total fuel mass consumed across all steps (kg).

    Raises
    ------
    KeyError
        If the specified engine is not found in the thrusters dictionary or lacks a mass specification.
    ValueError
        If positions array shape does not match accels or is not 3D.
    """
    if engine not in thrusters:
        raise KeyError(f"Engine '{engine}' not found in thrusters dictionary. Available: {list(thrusters.keys())}")
    
    if "mass" not in thrusters[engine]:
        thrusters[engine]["mass"] = 500

    if positions.shape[0] != accels.size:
        raise ValueError(f"Positions array size ({positions.shape[0]}) must match accels size ({accels.size})")
    if positions.shape[1] != 3:
        raise ValueError(f"Positions array must have shape (n, 3), got {positions.shape}")

    isp = thrusters[engine]["ISP"]  # Specific impulse in seconds
    mass0 = thrusters[engine]["mass"]  # Initial spacecraft mass in kg
    total_fuel = 0.0  # Total fuel used
    mass = mass0

    # Compute gravitational acceleration at each step
    for a, pos in zip(accels, positions):
        # Calculate radial distance r = sqrt(x^2 + y^2 + z^2)
        r = np.sqrt(np.sum(pos**2))
        # Compute g = mu / r^2
        g = EARTH_MU / (r**2)
        # Compute force and mass flow rate
        force = mass * a
        mdot = force / (isp * g)
        delta_m = mdot * dt
        # Accumulate total fuel used
        total_fuel += delta_m
        # Update spacecraft mass
        mass -= delta_m

    return total_fuel