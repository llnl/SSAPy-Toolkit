"""Compare SSATK two-body propagation with Orekit's KeplerianPropagator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.io.ssatk_data import ssatk_data
from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.propagators_orbit import propagate_orbit_state

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
OREKIT_VERSION = "10.3.1"
OREKIT_DIR = Path(__file__).with_name("orekit")


def _orekit_jar(*, allow_install: bool) -> Path | None:
    configured = os.environ.get("OREKIT_JAR")
    if configured and Path(configured).is_file():
        return Path(configured)
    jar = Path.home() / ".m2" / "repository" / "org" / "orekit" / "orekit" / OREKIT_VERSION / f"orekit-{OREKIT_VERSION}.jar"
    if jar.is_file() or not allow_install or shutil.which("mvn") is None:
        return jar if jar.is_file() else None
    subprocess.run(
        ["mvn", "-q", "dependency:get", f"-Dartifact=org.orekit:orekit:{OREKIT_VERSION}"],
        cwd=OREKIT_DIR,
        check=True,
    )
    return jar if jar.is_file() else None


def _run_orekit(*, radius: float, duration: float, step: float, allow_install: bool):
    jar = _orekit_jar(allow_install=allow_install)
    if (
        jar is None
        or shutil.which("mvn") is None
        or shutil.which("javac") is None
        or shutil.which("java") is None
    ):
        return None
    with tempfile.TemporaryDirectory(prefix="ssatk-orekit-") as temp:
        temp = Path(temp)
        classpath_file = temp / "classpath.txt"
        subprocess.run(
            ["mvn", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={classpath_file}"],
            cwd=OREKIT_DIR,
            check=True,
        )
        dependencies = classpath_file.read_text(encoding="utf-8").strip()
        classpath = os.pathsep.join((str(jar), dependencies))
        subprocess.run(
            ["javac", "-source", "8", "-target", "8", "-cp", classpath, "-d", str(temp), str(OREKIT_DIR / "OrekitTwoBody.java")],
            check=True,
        )
        completed = subprocess.run(
            ["java", "-cp", os.pathsep.join((str(temp), classpath)), "OrekitTwoBody", str(EARTH_MU), str(radius), str(duration), str(step)],
            check=True,
            capture_output=True,
            text=True,
        )
    rows = np.loadtxt(completed.stdout.splitlines()[1:], delimiter=",")
    return rows.reshape((-1, 7))


def main(make_figures=None, fast=None, verbose=None, allow_install=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST
    if allow_install is None:
        allow_install = not UNDER_PYTEST

    radius = 7_000_000.0
    duration = 3_600.0 if fast else 43_200.0
    step = 60.0
    orekit = _run_orekit(radius=radius, duration=duration, step=step, allow_install=allow_install)
    if orekit is None:
        return {"skipped": True, "reason": "Orekit, Java, or Maven unavailable"}

    times = orekit[:, 0]
    ssatk = propagate_orbit_state(
        r0=orekit[0, 1:4],
        v0=orekit[0, 4:7],
        times=times,
        mu=EARTH_MU,
    )
    dr = np.linalg.norm(ssatk.r - orekit[:, 1:4], axis=1)
    dv = np.linalg.norm(ssatk.v - orekit[:, 4:7], axis=1)
    result = {
        "skipped": False,
        "tool": f"Orekit {OREKIT_VERSION} KeplerianPropagator",
        "duration_s": duration,
        "sample_count": int(times.size),
        "rms_position_error_m": float(np.sqrt(np.mean(dr**2))),
        "max_position_error_m": float(np.max(dr)),
        "rms_velocity_error_m_s": float(np.sqrt(np.mean(dv**2))),
        "max_velocity_error_m_s": float(np.max(dv)),
    }
    if verbose:
        print(json.dumps(result, indent=2, sort_keys=True))
    if make_figures:
        out_dir = Path(figpath("benchmarks"))
        out_dir.mkdir(parents=True, exist_ok=True)
        hours = times / 3600.0
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, dr, label="|Δr|")
        ax.set(xlabel="Elapsed time [hr]", ylabel="Position residual [m]", title="SSATK vs Orekit two-body position")
        ax.grid(True, alpha=0.3)
        ax.legend()
        position_path = out_dir / "orekit_two_body_position_error.png"
        fig.savefig(position_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hours, dv, label="|Δv|")
        ax.set(xlabel="Elapsed time [hr]", ylabel="Velocity residual [m/s]", title="SSATK vs Orekit two-body velocity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        velocity_path = out_dir / "orekit_two_body_velocity_error.png"
        fig.savefig(velocity_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        result["position_error_plot"] = str(position_path)
        result["velocity_error_plot"] = str(velocity_path)
        result_path = Path(ssatk_data("benchmarks/orekit_two_body_results.json"))
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False, verbose=True, allow_install=True)
