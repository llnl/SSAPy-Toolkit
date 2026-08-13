import ssapy
import ssapy_toolkit as ssatk
from ssapy_toolkit import coordinates, orbital_mechanics, vectors
from ssapy_toolkit.time_functions.convert_dd_and_hms import dd_to_hms


def test_base_ssapy_public_objects_are_available_from_toolkit_entrypoint():
    assert ssatk.Orbit is ssapy.Orbit
    assert ssatk.rv is ssapy.rv
    assert ssatk.AccelKepler is ssapy.AccelKepler
    assert ssatk.groundTrack is ssapy.groundTrack


def test_only_curated_base_ssapy_core_is_aliased():
    assert not hasattr(ssatk, "Time")
    assert not hasattr(ssatk, "TimeDelta")
    assert not hasattr(ssatk, "plotUtils")


def test_from_import_uses_lazy_base_aliases():
    from ssapy_toolkit import Orbit, rv

    assert Orbit is ssapy.Orbit
    assert rv is ssapy.rv


def test_base_ssapy_module_is_available_for_direct_access():
    assert ssatk.ssapy is ssapy
    assert ssatk.ssapy.utils is ssapy.utils


def test_toolkit_submodules_take_precedence_over_base_submodules():
    import ssapy_toolkit.io as toolkit_io
    import ssapy_toolkit.utils as toolkit_utils

    assert ssatk.io is toolkit_io
    assert ssatk.utils is toolkit_utils
    assert ssatk.io is not ssapy.io
    assert ssatk.utils is not ssapy.utils


def test_duplicated_helpers_resolve_to_toolkit_implementations():
    assert ssatk.norm is vectors.norm
    assert ssatk.deg0to360 is coordinates.deg0to360
    assert ssatk.period is orbital_mechanics.period
    assert ssatk.dd_to_hms is dd_to_hms
    assert ssatk.rightascension2hourangle is coordinates.rightascension2hourangle
    assert "rightascension2hourangle" in dir(ssatk)


def test_split_hms_helper_accepts_dms_string_input():
    assert ssatk.dd_to_hms("15:0:0") == "1:0:0"


def test_constants_still_use_toolkit_deduplicated_module():
    import ssapy_toolkit.constants as toolkit_constants

    assert ssatk.constants is toolkit_constants
    assert ssatk.EARTH_MU == ssapy.constants.EARTH_MU
