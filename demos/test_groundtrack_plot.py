"""
Demo script for yeager_utils.groundtrack_plot

Saves each demo figure using:
    from yeager_utils import yufig

Output paths:
    tests/demo_groundtrack_01_default
    tests/demo_groundtrack_02_pacific_centered_raw
    tests/demo_groundtrack_03_pacific_centered_relabel
    tests/demo_groundtrack_04_multi_custom_styles
    tests/demo_groundtrack_05_custom_center_150_relabel
    tests/demo_groundtrack_06_custom_center_150_raw

Usage:
    python demo_groundtrack_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt

from yeager_utils import groundtrack_plot, yufig


def make_circular_orbit_track(
    alt_km=500.0,
    inc_deg=51.6,
    npts=1200,
    n_orbits=3.0,
    t0=0.0,
):
    """
    Simple synthetic circular orbit in an inertial frame.
    This is only for plotting demos; it is not intended as a high-fidelity propagator.

    Returns
    -------
    r : (n, 3) ndarray
        Position vectors [m]
    t : (n,) ndarray
        Time array [s]
    """
    mu = 3.986004418e14
    re = 6378.137e3
    a = re + alt_km * 1e3
    inc = np.radians(inc_deg)

    n = np.sqrt(mu / a**3)
    period = 2 * np.pi / n

    t = np.linspace(t0, t0 + n_orbits * period, npts)
    u = n * (t - t0)

    x_pf = a * np.cos(u)
    y_pf = a * np.sin(u)
    z_pf = np.zeros_like(u)

    # Inclination rotation about x-axis
    x = x_pf
    y = y_pf * np.cos(inc)
    z = y_pf * np.sin(inc)

    r = np.column_stack((x, y, z))
    return r, t


def save_demo(fig, name):
    """Save figure to tests/<name> using yufig."""
    yufig(fig, f"tests/{name}")


def main():
    # Synthetic example tracks
    r1, t1 = make_circular_orbit_track(alt_km=500, inc_deg=28.5, n_orbits=2.5)
    r2, t2 = make_circular_orbit_track(alt_km=700, inc_deg=63.4, n_orbits=2.5)
    r3, t3 = make_circular_orbit_track(alt_km=1200, inc_deg=98.0, n_orbits=2.5)

    ground_stations = np.array([
        [34.743, -120.572],   # Vandenberg-ish
        [28.573,  -80.649],   # KSC-ish
        [64.837, -147.716],   # Alaska-ish
        [35.247,  139.617],   # Japan-ish
    ])

    # 1) Default configuration
    fig1 = groundtrack_plot(
        r1,
        t1,
        title="Demo 1: Default Ground Track",
        ground_stations=ground_stations,
    )
    save_demo(fig1, "demo_groundtrack_01_default")

    # 2) Pacific-centered display with raw shifted tick labels
    fig2 = groundtrack_plot(
        r1,
        t1,
        title="Demo 2: Pacific-Centered Map",
        ground_stations=ground_stations,
        central_longitude=150,
        relabel_xticks=False,
    )
    save_demo(fig2, "demo_groundtrack_02_pacific_centered_raw")

    # 3) Pacific-centered display with standard relabeled longitudes
    fig3 = groundtrack_plot(
        r1,
        t1,
        title="Demo 3: Shifted Map, Standard Longitude Labels",
        ground_stations=ground_stations,
        central_longitude=180,
        relabel_xticks=True,
    )
    save_demo(fig3, "demo_groundtrack_03_pacific_centered_relabel")

    # 4) Multiple tracks with custom labels/colors/linestyles
    fig4 = groundtrack_plot(
        [r1, r2, r3],
        [t1, t2, t3],
        title="Demo 4: Multiple Orbits with Custom Styles",
        ground_stations=ground_stations,
        labels=["LEO 28.5 deg", "LEO 63.4 deg", "SSO 98 deg"],
        orbit_colors=["tab:blue", "tab:orange", "tab:green"],
        linestyles=["-", "--", ":"],
        central_longitude=0,
        relabel_xticks=True,
    )
    save_demo(fig4, "demo_groundtrack_04_multi_custom_styles")

    # 5) Shift map to a custom center longitude with relabeled ticks
    fig5 = groundtrack_plot(
        [r2, r3],
        [t2, t3],
        title="Demo 5: Custom Shifted Map (center=150 deg)",
        ground_stations=ground_stations,
        labels=["Molniya-like Inclination", "Sun-Synchronous-like"],
        orbit_colors=["purple", "crimson"],
        linestyles=["-.", "--"],
        central_longitude=150,
        relabel_xticks=True,
    )
    save_demo(fig5, "demo_groundtrack_05_custom_center_150_relabel")

    # 6) Same custom shift, but show raw shifted tick labels
    fig6 = groundtrack_plot(
        [r2, r3],
        [t2, t3],
        title="Demo 6: Custom Shifted Map with Raw Display Tick Labels",
        ground_stations=ground_stations,
        labels=["Track A", "Track B"],
        orbit_colors=["black", "teal"],
        linestyles=[":", "-"],
        central_longitude=150,
        relabel_xticks=False,
    )
    save_demo(fig6, "demo_groundtrack_06_custom_center_150_raw")

    # Optional interactive display
    plt.show()


if __name__ == "__main__":
    main()