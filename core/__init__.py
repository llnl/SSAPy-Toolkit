"""
core/__init__.py — SSAPy-Toolkit core library

Each group below is imported inside its own try/except. This mirrors the
fix applied to ssapy_toolkit/plots/__init__.py: previously a single bad
import anywhere in core/ (e.g. a typo in layers.py, or a missing optional
dependency like ppigrf) raised out of this file and took the *entire*
`core` package down with it — which is why toolkit_gui.py's
`from core import OrbitalState, PlotlyScene` would fail completely and
disable every layer, not just the broken one.

Now a failure in one group is caught, warned about, and the names it would
have exported are simply left out of `core.__all__` — everything else in
core/ still imports and works normally.
"""

import warnings

__all__ = []


def _try_import(label, do_import):
    """Run `do_import()` (which should return a dict of name -> object),
    merge successes into globals()/__all__, or warn and continue on failure."""
    try:
        result = do_import()
    except Exception as ex:
        warnings.warn(f"core/__init__.py: failed to import {label}: {ex}")
        return
    globals().update(result)
    __all__.extend(result.keys())


# ── orbit / propagation ──────────────────────────────────────────────────
def _load_orbit_state():
    from .orbit_state import OrbitalState, PropagatorConfig
    return {"OrbitalState": OrbitalState, "PropagatorConfig": PropagatorConfig}


_try_import("orbit_state.py", _load_orbit_state)

# ── plotting (Plotly scene builder for the 3D Preview tab) ──────────────
def _load_base_plot():
    from .base_plot import BasePlot3D, PlotlyScene
    return {"BasePlot3D": BasePlot3D, "PlotlyScene": PlotlyScene}


_try_import("base_plot.py", _load_base_plot)

# ── frames ────────────────────────────────────────────────────────────────
def _load_frames():
    from .frames import Frame, FrameTransform
    return {"Frame": Frame, "FrameTransform": FrameTransform}


_try_import("frames.py", _load_frames)

# ── layers ────────────────────────────────────────────────────────────────
def _load_layers():
    from .layers import EarthLayer, StarfieldLayer, GroundTrackLayer
    return {
        "EarthLayer": EarthLayer,
        "StarfieldLayer": StarfieldLayer,
        "GroundTrackLayer": GroundTrackLayer,
    }


_try_import("layers.py", _load_layers)

# ── satellite ─────────────────────────────────────────────────────────────
def _load_satellite():
    from .satellite import Satellite3D, BurnEvent
    return {"Satellite3D": Satellite3D, "BurnEvent": BurnEvent}


_try_import("satellite.py", _load_satellite)

# ── sun ephemeris + diffuse texture shading (Moon/Earth) ─────────────────
def _load_sun_mpl():
    from .sun_mpl import (
        get_sun_position,
        sun_direction_in_frame,
        shade_texture,
        apply_shading,
    )
    return {
        "get_sun_position": get_sun_position,
        "sun_direction_in_frame": sun_direction_in_frame,
        "shade_texture": shade_texture,
        "apply_shading": apply_shading,
    }


_try_import("sun_mpl.py", _load_sun_mpl)

# ── dedicated Sun renderer (limb-darkened) + light projection ────────────
def _load_sun_render():
    from .sun_render import (
        render_sun,
        light_direction_from_positions,
        background_sun_position,
        background_sun_radius,
    )
    return {
        "render_sun": render_sun,
        "light_direction_from_positions": light_direction_from_positions,
        "background_sun_position": background_sun_position,
        "background_sun_radius": background_sun_radius,
    }


_try_import("sun_render.py", _load_sun_render)

# Clean up helper names so they don't leak into `from core import *`
del _try_import, _load_orbit_state, _load_base_plot, _load_frames
del _load_layers, _load_satellite, _load_sun_mpl, _load_sun_render