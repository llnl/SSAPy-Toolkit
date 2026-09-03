"""Solar and lunar eclipse demonstration for the SSAPy-Toolkit gallery."""
from __future__ import annotations

from pathlib import Path

GALLERY_CATEGORY = "eclipse"


def main(make_figures: bool | None = None, fast: bool | None = None, backend: str = "reference"):
    from ssapy_toolkit.eclipse import EventRequest, ObserverConfig, build_event, validate_event

    fast = bool(fast)
    minimum_states = 25 if fast else 81
    observer = ObserverConfig(latitude_deg=30.410279649731205, longitude_east_deg=-97.96311785197467)
    solar = build_event(EventRequest(kind="solar", backend=backend, sample_count=minimum_states, solar_scope="observer", observer=observer))
    lunar = build_event(EventRequest(kind="lunar", backend=backend, sample_count=minimum_states))
    solar_validation = validate_event(solar)
    lunar_validation = validate_event(lunar)
    result = {
        "solar_event": solar,
        "lunar_event": lunar,
        "observer": observer,
        "solar_validation": solar_validation,
        "lunar_validation": lunar_validation,
        "files": [],
    }
    if make_figures:
        from ssapy_toolkit.plots.eclipse_scientific_suite import generate_event_scientific_summary
        from ssapy_toolkit.plots.figpath import figpath
        out = Path(figpath("demo_gallery/figures/eclipse"))
        out.mkdir(parents=True, exist_ok=True)
        solar_path = out / "solar_eclipse_summary.png"
        lunar_path = out / "lunar_eclipse_summary.png"
        generate_event_scientific_summary(solar, solar_path)
        generate_event_scientific_summary(lunar, lunar_path)
        result["files"].extend([solar_path, lunar_path])
    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False)
