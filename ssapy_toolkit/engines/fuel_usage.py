import numpy as np

from ..constants import G0
from .catalog import thruster_spec


def estimate_fuel_usage(
    accels: np.ndarray,
    dt: float,
    positions: np.ndarray,
    *,
    engine: str,
    initial_mass_kg: float,
    g0: float = G0,
) -> float:
    """
    Estimate total propellant mass used by an acceleration profile.

    Parameters
    ----------
    accels : np.ndarray
        Array of acceleration magnitudes (m/s²) at each time step.
    dt : float
        Time step duration (s).
    positions : np.ndarray
        Array of position vectors in meters, shape (n, 3). The values are only
        used to validate that one position exists per acceleration sample.
    engine : str
        Name or alias of a thruster preset from :mod:`ssapy_toolkit.engines`.
    initial_mass_kg : float
        Vehicle mass used to convert acceleration to thrust.
    g0 : float, optional
        Standard gravity for Isp conversion. Specific impulse is defined using
        standard gravity, not local gravity.

    Returns
    -------
    total_fuel : float
        Total fuel mass consumed across all steps (kg).

    Raises
    ------
    KeyError
        If the specified engine is not found in the thruster catalog.
    ValueError
        If positions array shape does not match accels or is not 3D.
    """
    accels = np.asarray(accels, dtype=float)
    positions = np.asarray(positions, dtype=float)
    dt = float(dt)
    g0 = float(g0)
    spec = thruster_spec(engine)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if g0 <= 0.0:
        raise ValueError("g0 must be positive.")

    if positions.shape[0] != accels.size:
        raise ValueError(f"Positions array size ({positions.shape[0]}) must match accels size ({accels.size})")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Positions array must have shape (n, 3), got {positions.shape}")
    if np.any(accels < 0.0):
        raise ValueError("accels must be non-negative magnitudes.")

    isp = spec.nominal_isp_s
    mass0 = float(initial_mass_kg)
    if mass0 <= 0.0:
        raise ValueError("initial_mass_kg must be positive.")
    total_fuel = 0.0  # Total fuel used
    mass = mass0

    for a, _pos in zip(accels, positions):
        force = mass * a
        mdot = force / (isp * g0)
        delta_m = mdot * dt
        # Accumulate total fuel used
        total_fuel += delta_m
        # Update spacecraft mass
        mass -= delta_m

    return total_fuel
