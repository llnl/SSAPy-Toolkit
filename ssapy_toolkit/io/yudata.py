"""Compatibility wrapper for the legacy local data path helper."""

from .datapath import datapath, dpath


def yudata(filename="data", dirs=None):
    """Deprecated alias for :func:`ssapy_toolkit.io.datapath.datapath`."""
    return datapath(filename, dirs=dirs)


__all__ = ["datapath", "dpath", "yudata"]
