"""SSATK HDF5 cache convenience helpers."""

from .h5cache import h5cache, h5load


def ssatk_cache(*args, **kwargs):
    """Save an HDF5 cache via :func:`ssapy_toolkit.io.h5cache.h5cache`."""
    return h5cache(*args, **kwargs)


def ssatk_load(*args, **kwargs):
    """Load an HDF5 cache via :func:`ssapy_toolkit.io.h5cache.h5load`."""
    return h5load(*args, **kwargs)


__all__ = ["h5cache", "h5load", "ssatk_cache", "ssatk_load"]
