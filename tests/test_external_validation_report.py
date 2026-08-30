import json

from scripts import external_validation_report as report


def _successful(case_id):
    values = {
        "orekit_two_body": {"skipped": False, "max_position_error_m": 1e-3, "max_velocity_error_m_s": 1e-6},
        "gmat_two_body": {"skipped": False, "max_position_error_m": 2e-3, "max_velocity_error_m_s": 2e-6},
        "basilisk_6dof": {
            "skipped": False,
            "max_position_error_m": 1e-6,
            "max_velocity_error_m_s": 1e-8,
            "max_quaternion_error": 1e-9,
            "max_body_rate_error_rad_s": 1e-10,
        },
    }
    return values[case_id]


def test_external_validation_report_has_common_acceptance_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_call", lambda case, **kwargs: _successful(case["id"]))
    path = tmp_path / "external.json"

    payload = report.run_external_validation(output_path=path)

    assert payload["deterministic"] is True
    assert payload["summary"] == {"passed": 3, "failed": 0, "skipped": 0, "errors": 0, "total": 3}
    assert all(case["status"] == "passed" for case in payload["cases"])
    assert json.loads(path.read_text()) == payload
    assert all({"name", "value", "unit", "tolerance", "pass", "reference"} <= set(metric) for case in payload["cases"] for metric in case["acceptance"])

    full = report.run_external_validation(fast=False)
    assert [case["settings"]["duration_s"] for case in full["cases"]] == [43_200.0, 14_400.0, 120.0]


def test_require_external_rejects_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_call", lambda case, **kwargs: {"skipped": True, "reason": "not installed"})

    assert report.main(["--output", str(tmp_path / "report.json"), "--require-external"]) == 1
    assert report.main(["--output", str(tmp_path / "report-optional.json")]) == 0
