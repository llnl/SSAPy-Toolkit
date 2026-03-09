import numpy as np
from yeager_utils import build_dashboard


def panel_altitude(ax, fig, t_min, alt_km):
    ax.plot(t_min, alt_km, lw=2.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True)
    ax.set_title("Altitude vs Time")

def panel_xy(ax, fig, x, y):
    ax.scatter(x, y, s=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title("XY")

t = np.linspace(0, 90, 500)
alt = 400 + 20*np.sin(2*np.pi*t/90)

x = np.random.randn(1000)
y = np.random.randn(1000)

fig, axes, _ = build_dashboard(
    panels=[
        {"loc": (0, 0), "render": panel_altitude, "kwargs": {"t_min": t, "alt_km": alt}},
        {"loc": (0, 1), "render": panel_xy,       "kwargs": {"x": x, "y": y}},
    ],
    nrows=1, ncols=2, figsize=(12, 4), show=True
)