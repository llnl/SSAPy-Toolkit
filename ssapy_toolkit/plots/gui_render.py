"""
ssapy_toolkit/plots/gui_render.py
----------------------------------
In-process figure builders for toolkit_gui.py.

Why this exists
---------------
The GUI historically ran every plot as a *subprocess*:

    write _gui_cfg_<key>.py  ->  conda run -n <env> python -m <module>
                             ->  module reads $GUI_CONFIG, renders, saves to disk

That had four problems, all of which this module fixes:

1. **Silent no-ops.** Four of the registered plot modules (orbit_plot,
   globe_plot, cislunar_plot_3d, cislunar_plot) are pure *library* modules --
   they define a function and have no ``if __name__ == "__main__"`` block at
   all. Running ``python -m ssapy_toolkit.plots.globe_plot`` therefore
   imports the module, does nothing, and exits 0. The GUI read that exit code
   as success and logged "complete" while producing no figure whatsoever.
2. **Speed.** Every plot paid for a fresh ``conda run`` interpreter startup
   plus a full re-import of numpy/matplotlib/plotly/astropy/ssapy.
3. **Disk round-trip.** Figures could only be delivered as files, so the GUI
   could not show them inline -- it printed a log and left you to go open the
   output folder yourself.
4. **Generated-config fragility.** The ``_gui_cfg_<key>.py`` files are
   machine-written Python imported at runtime; a single bad literal broke the
   whole run.

Calling the plot functions directly removes all four at once: a failure is a
real exception with a real traceback, not a silent success, and the caller
gets a live figure object it can hand straight to ``st.pyplot`` /
``st.plotly_chart``.

Units
-----
``OrbitalState.propagate()`` returns ``Trajectory.r`` in **kilometres**, but
the plot functions (``orbit_plot``, ``globe_plot``, ``cislunar_plot*``) all
expect **metres** -- internally they divide by ``RGEO`` or ``EARTH_RADIUS``
from ``ssapy_toolkit.constants``, which are metre-valued. ``_traj_to_metres``
below does that conversion in exactly one place so the 1000x error can't be
reintroduced per-adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    """What an in-process build returns.

    fig    : the live figure object (matplotlib Figure or plotly Figure), or
             None if the build failed.
    engine : "matplotlib" | "plotly" -- tells the GUI which Streamlit call to
             use (st.pyplot vs st.plotly_chart).
    ok     : whether the build succeeded.
    msg    : human-readable status/error, always populated on failure.
    """
    fig: Any = None
    engine: str = "matplotlib"
    ok: bool = False
    msg: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _traj_to_metres(traj):
    """Convert a Trajectory's km positions to the metres the plot functions want.

    See the module docstring: this is the single place the km->m conversion
    happens, deliberately, so individual adapters can't disagree about units.
    """
    r_m = np.asarray(traj.r, dtype=float) * 1000.0
    t = np.asarray(traj.t, dtype=float)
    return r_m, t


def _propagate(state, cfg):
    """Propagate an OrbitalState using the GUI's n_orbits/dt_s settings.

    Raises RuntimeError with the propagator's own message if it failed, rather
    than returning an empty trajectory that would render as a blank plot.
    """
    if state is None:
        raise RuntimeError(
            "No OrbitalState available -- ssapy_toolkit.plots core modules "
            "failed to import (see the GUI's startup diagnostics)."
        )
    traj = state.propagate(
        n_orbits=float(cfg.get("n_orbits", 3.0)),
        dt_s=float(cfg.get("dt_s", 60.0)),
    )
    if not getattr(traj, "ok", True):
        raise RuntimeError(getattr(traj, "msg", None) or "Propagation failed")
    if np.asarray(traj.r).size == 0:
        raise RuntimeError("Propagation returned no points")
    return traj


def _theme(cfg):
    """Map the GUI's light/dark preference onto the plot functions' `c` arg.

    The GUI stores this as ``dark_bg`` inside each plot's ``ps_<key>``
    settings dict (see PLOT_MODULES defaults in toolkit_gui.py), which
    run_script() merges into cfg -- not as a top-level ``dark`` key. Reading
    the wrong name here silently ignored the toggle and always rendered dark,
    so both spellings are accepted with ``dark_bg`` taking precedence.

    Note ``ps_globe`` ships no ``dark_bg`` at all, so the default matters:
    it's True, matching the plot functions' own ``c="black"`` default.
    """
    dark = cfg.get("dark_bg", cfg.get("dark", True))
    return "black" if dark else "white"


# ---------------------------------------------------------------------------
# Adapters -- one per PLOT_MODULES key
#
# Each takes (state, cfg) and returns a RenderResult. They deliberately do NOT
# catch exceptions: build_figure() below does that once, so a failure surfaces
# as a real message instead of a silent success.
# ---------------------------------------------------------------------------

def _build_orbit_full(state, cfg) -> RenderResult:
    from ssapy_toolkit.plots.orbit_plot import orbit_plot

    traj = _propagate(state, cfg)
    r_m, t = _traj_to_metres(traj)
    fig, _axes = orbit_plot(
        [r_m], t=[t],
        title=cfg.get("title", ""),
        frame=cfg.get("frame", "gcrf"),
        c=_theme(cfg),
        show=False,
        save_path=False,
    )
    return RenderResult(fig=fig, engine="matplotlib", ok=True,
                        msg="Full orbit panel rendered")


def _build_globe(state, cfg) -> RenderResult:
    from ssapy_toolkit.plots.globe_plot import globe_plot

    traj = _propagate(state, cfg)
    r_m, t = _traj_to_metres(traj)
    # globe_plot orients the texture from a real time when given one, so pass
    # the trajectory's first epoch rather than leaving the globe at lon0=0.
    fig, _ax = globe_plot(
        [r_m], t=[t],
        title=cfg.get("title", ""),
        c=_theme(cfg),
        globe_time=float(t[0]) if len(t) else None,
        save_path=None,
    )
    return RenderResult(fig=fig, engine="matplotlib", ok=True,
                        msg="Globe / ground track rendered")


def _build_cislunar_3d(state, cfg) -> RenderResult:
    from ssapy_toolkit.plots.cislunar_plot_3d import cislunar_plot_3d

    traj = _propagate(state, cfg)
    r_m, t = _traj_to_metres(traj)
    fig, _ax = cislunar_plot_3d(
        [r_m], t=[t],
        title=cfg.get("title", ""),
        c=_theme(cfg),
        show=False,
        save_path=False,
    )
    return RenderResult(fig=fig, engine="matplotlib", ok=True,
                        msg="Cislunar 3D rendered")


def _build_cislunar_combo(state, cfg) -> RenderResult:
    from ssapy_toolkit.plots.cislunar_plot import cislunar_plot

    traj = _propagate(state, cfg)
    r_m, t = _traj_to_metres(traj)
    fig, _axes = cislunar_plot(
        [r_m], t=[t],
        title=cfg.get("title") or None,
        c=_theme(cfg),
        show=False,
        save_path=False,
    )
    return RenderResult(fig=fig, engine="matplotlib", ok=True,
                        msg="Cislunar combined rendered")


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------

# Keys present here render in-process. Keys absent fall back to the legacy
# subprocess path in toolkit_gui.run_script(), so converting the remaining
# modules is incremental -- add an adapter, add it here, done.
BUILDERS: dict[str, Callable[[Any, dict], RenderResult]] = {
    "orbit_full":     _build_orbit_full,
    "globe":          _build_globe,
    "cislunar_3d":    _build_cislunar_3d,
    "cislunar_combo": _build_cislunar_combo,
}


def supports(key: str) -> bool:
    """True if `key` can be rendered in-process (no subprocess needed)."""
    return key in BUILDERS


def build_figure(key: str, state, cfg: dict) -> RenderResult:
    """Build one plot in-process.

    Returns a RenderResult in all cases -- including failure, where ok=False
    and msg carries the error. This is the important behavioural difference
    from the subprocess path: a module that produces nothing reports a
    failure here, instead of exiting 0 and being logged as a success.
    """
    builder = BUILDERS.get(key)
    if builder is None:
        return RenderResult(
            ok=False,
            msg=f"No in-process builder for '{key}' -- use the subprocess path.",
        )
    try:
        result = builder(state, cfg)
    except Exception as ex:  # surfaced to the GUI log, not swallowed
        return RenderResult(ok=False, msg=f"{type(ex).__name__}: {ex}")

    if result.fig is None:
        # A builder that returns no figure is a bug in the builder, not a
        # success -- catch it here rather than letting the GUI show a blank.
        return RenderResult(ok=False, engine=result.engine,
                            msg=f"'{key}' builder returned no figure")
    return result
