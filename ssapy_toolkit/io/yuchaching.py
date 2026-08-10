"""Compatibility wrapper for the legacy HDF5 cache helpers."""

from .h5cache import h5cache, h5load


def yucache(*args, **kwargs):
    """Deprecated alias for :func:`ssapy_toolkit.io.h5cache.h5cache`."""
    return h5cache(*args, **kwargs)


def yuload(*args, **kwargs):
    """Deprecated alias for :func:`ssapy_toolkit.io.h5cache.h5load`."""
    return h5load(*args, **kwargs)


__all__ = ["h5cache", "h5load", "yucache", "yuload"]
