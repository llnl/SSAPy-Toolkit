"""SSATK local data path helper."""

from .datapath import datapath, dpath


def ssatk_data(filename="data", dirs=None):
    """Return a local data path via :func:`ssapy_toolkit.io.datapath.datapath`."""
    return datapath(filename, dirs=dirs)


__all__ = ["datapath", "dpath", "ssatk_data"]
