from __future__ import annotations

import importlib


def test_auto_import_packages_do_not_export_helper_or_modules():
    packages = [
        "ssapy_toolkit.accelerations_6dof",
        "ssapy_toolkit.accelerations_orbit",
        "ssapy_toolkit.coordinates",
        "ssapy_toolkit.engines",
        "ssapy_toolkit.plots",
        "ssapy_toolkit.propagators_6dof",
        "ssapy_toolkit.propagators_orbit",
    ]

    for package_name in packages:
        package = importlib.import_module(package_name)
        public = set(getattr(package, "__all__", ()))

        assert "import_public_modules" not in public
        assert not hasattr(package, "import_public_modules")
        assert "spacecraft" not in public
        assert "sixdof" not in public
        assert public.isdisjoint(
            {
                "Any",
                "Callable",
                "Iterable",
                "Path",
                "Sequence",
                "Time",
                "annotations",
                "dataclass",
                "field",
                "np",
                "warnings",
            }
        )


def test_auto_import_packages_keep_expected_public_api():
    from ssapy_toolkit import accelerations_6dof, coordinates, engines, propagators_6dof

    assert "SpacecraftAccelJ2" in accelerations_6dof.__all__
    assert accelerations_6dof.SpacecraftAccelJ2
    assert "attitude_quaternion_from_frame" in coordinates.__all__
    assert coordinates.attitude_quaternion_from_frame
    assert "make_thruster_acceleration" in engines.__all__
    assert engines.make_thruster_acceleration
    assert "Spacecraft" in propagators_6dof.__all__
    assert propagators_6dof.Spacecraft
