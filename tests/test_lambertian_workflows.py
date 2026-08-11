import numpy as np
from astropy.time import Time

from ssapy_toolkit.compute.lambertian_magnitude import lambertian_reflection


def test_lambertian_reflection_vectorized_topocentric_magnitudes():
    time = Time("2024-01-01T00:00:00", scale="utc")
    positions = np.array(
        [
            [42_164_000.0, 0.0, 0.0],
            [0.0, 42_164_000.0, 0.0],
            [-42_164_000.0, 0.0, 0.0],
        ]
    )

    out = lambertian_reflection(
        positions,
        time=time,
        lon=-156.26,
        lat=20.71,
        elevation=3055.0,
        radius_m=1.0,
        albedo=0.2,
    )

    magnitudes = np.asarray(out["ab_mag_observed"])
    assert magnitudes.shape == (3,)
    assert np.isfinite(magnitudes).any()
    assert np.asarray(out["angles_deg"]["phase_sun_obj_obs_deg"]).shape == (3,)
