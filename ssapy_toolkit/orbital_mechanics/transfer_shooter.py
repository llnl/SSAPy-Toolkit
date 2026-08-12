"""Standard fixed-time transfer wrapper.

The closest-approach finite-difference shooter has been replaced with the
validated Lambert plus optional finite-burn refinement path in
``transfer_ssapy``.
"""

from __future__ import annotations

from ssapy_toolkit.orbital_mechanics.transfer_lambertian import _with_tof
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy


def transfer_shooter(*args, orbit1=None, orbit2=None, initial=None, target=None, r1=None, v1=None, r2=None, v2=None, t1=0.0, t2=None, tof=None, **kwargs):
    """Solve a fixed-time transfer and return the canonical transfer dict."""
    departure, arrival = _with_tof(
        *args,
        orbit1=orbit1,
        orbit2=orbit2,
        initial=initial,
        target=target,
        r1=r1,
        v1=v1,
        r2=r2,
        v2=v2,
        t1=t1,
        t2=t2,
        tof=tof,
    )
    result = transfer_ssapy(departure, arrival, **kwargs)
    result["method"] = "transfer_shooter"
    result["assumptions"].append("standardized wrapper around transfer_ssapy; no closest-approach finite-difference search")
    return result
