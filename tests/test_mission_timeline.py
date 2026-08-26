import pytest

from ssapy_toolkit.operations import MissionEvent, MissionTimeline


def test_timeline_orders_same_epoch_events_by_priority_then_name():
    timeline = MissionTimeline([
        MissionEvent(10.0, "burn", priority=2),
        MissionEvent(10.0, "sensor_off", priority=1),
        MissionEvent(2.0, "startup"),
    ])
    assert [event.name for event in timeline.events] == ["startup", "sensor_off", "burn"]
    assert [event.name for event in timeline.between(2.0, 10.0)] == ["startup", "sensor_off", "burn"]


def test_timeline_window_boundaries_and_tolerance():
    timeline = MissionTimeline([MissionEvent(1.0, "a"), MissionEvent(2.0, "b")])
    assert [event.name for event in timeline.between(1.0, 2.0, include_start=False, include_stop=False)] == []
    assert [event.name for event in timeline.at(1.001, tolerance=0.01)] == ["a"]


def test_timeline_rejects_duplicate_events_and_bad_windows():
    with pytest.raises(ValueError, match="duplicate"):
        MissionTimeline([MissionEvent(1.0, "a"), MissionEvent(1.0, "a")])
    with pytest.raises(ValueError, match="greater than"):
        MissionTimeline().between(2.0, 1.0)
