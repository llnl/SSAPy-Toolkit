"""Useful continuous-transfer gallery demo replacing small transfer smoke scripts."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from ssapy_toolkit.constants import RGEO, VGEO
from ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous import transfer_velocity_continuous
from ssapy_toolkit.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "demo_gallery/figures"


def _inclination_deg(r, v):
    h = np.cross(r, v)
    return np.degrees(np.arccos(np.clip(h[:, 2] / np.linalg.norm(h, axis=1), -1.0, 1.0)))


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    r0 = np.array([RGEO, 0.0, 0.0])
    v0 = np.array([0.0, VGEO, 0.0])
    dv_target = 20.0 if fast else 50.0
    thrust = 1.0

    r_vel, v_vel, t_vel = transfer_velocity_continuous(
        r0=r0,
        v0=v0,
        v_target=dv_target,
        a_thrust=thrust,
        max_time=120.0,
    )
    r_inc, v_inc, t_inc = transfer_inclination_continuous(
        r0=r0,
        v0=v0,
        delta_v=dv_target,
        a_thrust=thrust,
        max_time=120.0,
    )

    speed_vel = np.linalg.norm(v_vel, axis=1)
    speed_inc = np.linalg.norm(v_inc, axis=1)
    inc_vel = _inclination_deg(r_vel, v_vel)
    inc_inc = _inclination_deg(r_inc, v_inc)

    if make_figures:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].plot(r_vel[:, 0] / 1e3, r_vel[:, 1] / 1e3, label="velocity thrust")
        axes[0].plot(r_inc[:, 0] / 1e3, r_inc[:, 1] / 1e3, label="normal thrust")
        axes[0].set_xlabel("x [km]")
        axes[0].set_ylabel("y [km]")
        axes[0].set_title("Short continuous-burn arcs")
        axes[0].axis("equal")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(t_vel, speed_vel - speed_vel[0], label="velocity thrust")
        axes[1].plot(t_inc, speed_inc - speed_inc[0], label="normal thrust")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("speed change [m/s]")
        axes[1].set_title("Speed response")
        axes[1].grid(alpha=0.3)

        axes[2].plot(t_vel, inc_vel, label="velocity thrust")
        axes[2].plot(t_inc, inc_inc, label="normal thrust")
        axes[2].set_xlabel("time [s]")
        axes[2].set_ylabel("inclination [deg]")
        axes[2].set_title("Inclination response")
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        figsave(fig, f"{FIGDIR}/continuous_transfer_overview.jpg")

    return {
        "velocity_burn": (r_vel, v_vel, t_vel),
        "normal_burn": (r_inc, v_inc, t_inc),
        "speed_change_velocity": speed_vel - speed_vel[0],
        "inclination_normal": inc_inc,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
