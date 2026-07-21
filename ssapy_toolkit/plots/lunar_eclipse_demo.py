"""
lunar_eclipse_demo.py — backward-compatibility shim
====================================================
This file's actual content was merged into eclipse_space_view_plotly.py
(now the single "main" eclipse module: search + matplotlib panel +
interactive 3D space view, all in one file). This shim just re-exports
the same names from there, so any existing code doing
`from lunar_eclipse_demo import moon_color` (or similar) keeps working
unchanged.
"""
try:
    from .eclipse_space_view_plotly import (
        find_and_plot_eclipse, moon_color, moon_brightness, moon_red_bias,
        D_MOON_A_KM, D_MOON_E, D_MOON_INC_DEG, R_MOON_KM,
    )
except ImportError:
    from eclipse_space_view_plotly import (
        find_and_plot_eclipse, moon_color, moon_brightness, moon_red_bias,
        D_MOON_A_KM, D_MOON_E, D_MOON_INC_DEG, R_MOON_KM,
    )
