"""Shared helpers for ground-track dashboard plots."""

from __future__ import annotations

import numpy as np


def clean_lonlat(lon, lat):
    """Insert NaNs at longitude wrap breaks for continuous ground-track lines."""

    wraps = np.abs(np.diff(lon)) > 180
    lon_nan = np.insert(lon, np.where(wraps)[0] + 1, np.nan)
    lat_nan = np.insert(lat, np.where(wraps)[0] + 1, np.nan)
    return lon_nan, lat_nan


def force_title(ax, text, size, y=1.02):
    """Set a title consistently for 2D and 3D matplotlib axes."""

    ax.set_title("")
    if hasattr(ax, "text2D"):
        ax.text2D(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=size)
    else:
        ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=size)
