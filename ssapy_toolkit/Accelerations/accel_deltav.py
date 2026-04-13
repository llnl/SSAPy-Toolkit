import numpy as np

def accel_deltav(dv: float, thrust_accel: float, center_idx: int, dt: float = 1.0) -> dict:
    """
    Build a Δv-based along-track accel profile centered at a given step.

    Parameters
    ----------
    dv : float
        Signed total Δv to deliver (m/s). (Only its sign matters.)
    thrust_accel : float
        Available constant acceleration magnitude (m/s²). Must be > 0.
    center_idx : int
        Index in your time array where the burn is centered.
    dt : float, optional
        Uniform timestep between your indices (s). Defaults to 1 s.

    Returns
    -------
    profile : dict
        A dict with keys:
          - 'thrust' : signed acceleration (m/s²)
          - 'start'  : start index (int)
          - 'end'    : end index (int)
        which you can pass directly to `leapfrog(..., velocity=profile)`.
    """
    if thrust_accel <= 0:
        raise ValueError("`thrust_accel` must be positive")
    if dt <= 0:
        raise ValueError("`dt` must be positive")

    # total burn time (s) required for |Δv|
    burn_time = abs(dv) / thrust_accel

    # number of steps (round to nearest int, minimum 1)
    steps = max(1, int(round(burn_time / dt)))

    # center the burn on center_idx
    half = steps // 2
    start = center_idx - half
    end = start + steps  # ensures end - start == steps

    # signed acceleration to match dv's sign
    signed_accel = thrust_accel * np.sign(dv)

    return {
        "thrust": signed_accel,
        "start": start,
        "end": end,
    }
