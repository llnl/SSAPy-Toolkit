"""Standard Lambert transfer wrapper.

This function intentionally uses the validated ``transfer_ssapy`` Lambert
solver. The historical custom Lambertian implementation was removed because it
used a non-standard approximation and could produce unphysical delta-v values.
"""

from __future__ import annotations

from ssapy_toolkit.orbital_mechanics._transfer_result import transfer_boundary_states
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy


def _with_tof(*args, orbit1=None, orbit2=None, initial=None, target=None, r1=None, v1=None, r2=None, v2=None, t1=0.0, t2=None, tof=None):
    initial_state, target_state = transfer_boundary_states(
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
        name="fixed-time transfer",
    )
    if target_state["t"] <= initial_state["t"]:
        raise ValueError("target epoch must be after initial epoch, or supply tof/t2 in seconds")
    return (initial_state["r"], initial_state["v"], initial_state["t"]), (target_state["r"], target_state["v"], target_state["t"])


def transfer_lambertian(*args, orbit1=None, orbit2=None, initial=None, target=None, r1=None, v1=None, r2=None, v2=None, t1=0.0, t2=None, tof=None, **kwargs):
    """Solve a fixed-time Lambert transfer and return the canonical transfer dict.

    Boundary states may be ``(initial, target)`` or raw vectors
    ``(r1, v1, r2, v2)``.  Raw vector calls require ``tof`` or ``t2``.
    """
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
    result["method"] = "transfer_lambertian"
    result["assumptions"].append("compatibility name for transfer_ssapy Lambert solve")
    return result
