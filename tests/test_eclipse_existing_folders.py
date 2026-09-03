from __future__ import annotations

import numpy as np


def test_core_overlap_limits():
    from ssapy_toolkit.compute.eclipse_core import circle_overlap_visible_fraction
    assert float(circle_overlap_visible_fraction(1.0, 1.0, 3.0)) == 1.0
    assert float(circle_overlap_visible_fraction(2.0, 1.0, 0.0)) == 0.0


def test_canonical_reference_events():
    from ssapy_toolkit.eclipse import EventRequest, build_event, validate_event
    for kind in ("solar", "lunar"):
        event = build_event(EventRequest(kind=kind, backend="reference", sample_count=25))
        assert len(event.jd) >= 25
        assert validate_event(event).passed
        assert np.all(np.isfinite(event.moon_km))
        assert np.all(np.isfinite(event.sun_km))


def test_demo_fast():
    from demos.eclipse.demo_eclipse_scientific import main
    result = main(make_figures=False, fast=True, backend="reference")
    assert result["solar_validation"].passed
    assert result["lunar_validation"].passed
