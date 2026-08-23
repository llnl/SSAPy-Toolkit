"""High-accuracy 6-DoF propagation helpers."""

from ..dynamics import propagate_6dof as _propagate_6dof


def propagate_6dof_high_accuracy(**kwargs):
    """Call :func:`ssapy_toolkit.dynamics.propagate_6dof` with high-accuracy defaults."""

    kwargs.setdefault("method", "DOP853")
    kwargs.setdefault("rtol", 1e-10)
    kwargs.setdefault("atol", 1e-9)
    return _propagate_6dof(**kwargs)
