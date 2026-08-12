"""Coplanar fixed-time transfer wrapper using validated Lambert math."""

from __future__ import annotations

import numpy as np

from ssapy_toolkit.orbital_mechanics.transfer_lambertian import _with_tof
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy


def transfer_coplanar(*args, orbit1=None, orbit2=None, initial=None, target=None, r1=None, v1=None, r2=None, v2=None, t1=0.0, t2=None, tof=None, coplanar_tol=1e-6, **kwargs):
    """Solve a fixed-time transfer and report coplanarity diagnostics."""
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
    result["method"] = "transfer_coplanar"
    w_components = [burn["delta_v_ntw"][2] for burn in result["burns"] if burn.get("delta_v_ntw") is not None]
    max_w = max((abs(float(value)) for value in w_components), default=0.0)
    result["diagnostics"]["max_out_of_plane_delta_v"] = max_w
    result["diagnostics"]["coplanar_tol"] = coplanar_tol
    result["success"] = result["success"] and max_w <= coplanar_tol
    result["assumptions"].append("coplanar transfer diagnostic: W-axis delta-v should be near zero")
    return result
