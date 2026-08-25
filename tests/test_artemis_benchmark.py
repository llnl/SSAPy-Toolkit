import numpy as np
from astropy.time import Time

from demos.benchmarks.demo_artemis_benchmark import _match_executed_maneuvers


def test_artemis_maneuver_matching_uses_executed_events_only():
    times = Time(["2026-04-01T22:35:12", "2026-04-03T00:12:12"], scale="utc")
    metadata = {
        "launch_utc": "2026-04-01T22:35:12Z",
        "events": [
            {"name": "executed", "met_s": 92220, "status": "executed"},
            {"name": "skipped", "met_s": 92220, "status": "skipped"},
        ],
    }
    indices, events = _match_executed_maneuvers(times, metadata)
    assert indices == [1]
    assert [event["name"] for event in events] == ["executed"]
    assert np.isclose(events[0]["sample_offset_s"], 0.0)
