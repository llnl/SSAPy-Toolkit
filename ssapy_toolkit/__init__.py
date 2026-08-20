"""SSAPy Toolkit.

Utilities for orbital mechanics, plotting, coordinate transforms,
6-DoF propagators, io helpers, and SSAPy-related workflows.
"""

from importlib import import_module

__version__ = "1.0.5"

_CONSTANT_NAMES = None
_SSAPY_ALIAS_NAMES = frozenset(
    {
        "Accel",
        "AccelConstNTW",
        "AccelDrag",
        "AccelEarthRad",
        "AccelHarmonic",
        "AccelKepler",
        "AccelSolRad",
        "AccelSum",
        "AccelThirdBody",
        "BinarySelectorParams",
        "Body",
        "DanchickTwoPosOrbitSolver",
        "DirectInitializer",
        "DistanceProjectionInitializer",
        "EarthObserver",
        "EarthOrientation",
        "Ellipsoid",
        "EmceeSampler",
        "GEOProjectionInitializer",
        "GaussianRVInitializer",
        "GaussTwoPosOrbitSolver",
        "HarmonicCoefficients",
        "KeplerianPropagator",
        "LMOptimizer",
        "Leapfrog4Propagator",
        "LeapfrogPropagator",
        "Linker",
        "MHSampler",
        "MVNormalProposal",
        "ModelSelectorParams",
        "MoonOrientation",
        "MoonPosition",
        "Orbit",
        "OrbitalObserver",
        "Particles",
        "RK4Propagator",
        "RK78Propagator",
        "RK8Propagator",
        "RVProbability",
        "RVSigmaProposal",
        "SGP4Propagator",
        "SciPyPropagator",
        "SeriesPropagator",
        "SheferTwoPosOrbitSolver",
        "ThreeAngleOrbitSolver",
        "TwoPosOrbitSolver",
        "altaz",
        "circular_guess",
        "datadir",
        "dircos",
        "get_body",
        "groundTrack",
        "quickAltAz",
        "radec",
        "radecRate",
        "rv",
    }
)
_TOOLKIT_SUBMODULE_NAMES = frozenset(
    {
        "accelerations_6dof",
        "asteroids",
        "compute",
        "constants",
        "coordinates",
        "data",
        "demo_gallery",
        "dynamics",
        "engines",
        "hpc",
        "io",
        "orbital_mechanics",
        "plots",
        "propagators_6dof",
        "rockets",
        "run_all_demos",
        "ssapy_wrappers",
        "time_functions",
        "utils",
        "vectors",
        "yastropy",
    }
)
_TOOLKIT_DUPLICATE_ALIASES = {
    "apoapsis": ".orbital_mechanics.keplerian",
    "cart2sph_deg": ".coordinates.cartesian",
    "cart_to_cyl": ".coordinates.cartesian",
    "dd_to_dms": ".time_functions.convert_dd_and_dms",
    "dd_to_hms": ".time_functions.convert_dd_and_hms",
    "deg0to360": ".coordinates.angle_units",
    "deg0to360array": ".coordinates.angle_units",
    "deg90to90": ".coordinates.angle_units",
    "deg90to90array": ".coordinates.angle_units",
    "dms_to_dd": ".time_functions.convert_dd_and_dms",
    "dms_to_deg": ".coordinates.angle_units",
    "dms_to_rad": ".coordinates.angle_units",
    "ecliptic_to_equatorial": ".coordinates.equatorial_ecliptic",
    "ecliptic_xyz_to_equatorial": ".coordinates.equatorial_ecliptic",
    "ecliptic_xyz_to_equatorial_xyz": ".coordinates.equatorial_ecliptic",
    "einsum_norm": ".vectors",
    "equatorial_to_ecliptic": ".coordinates.equatorial_ecliptic",
    "equatorial_to_horizontal": ".coordinates.local_equatorial",
    "equatorial_xyz_to_ecliptic_xyz": ".coordinates.equatorial_ecliptic",
    "hms_to_dd": ".time_functions.convert_dd_and_hms",
    "horizontal_to_equatorial": ".coordinates.local_equatorial",
    "inert2rot": ".coordinates.rotating_frames",
    "load_earth_file": ".plots.plotutils",
    "load_moon_file": ".plots.plotutils",
    "lonlat_distance": ".coordinates.geodetic",
    "norm": ".vectors",
    "normSq": ".vectors",
    "normed": ".vectors",
    "periapsis": ".orbital_mechanics.keplerian",
    "propagate_6dof": ".dynamics",
    "Spacecraft": ".dynamics",
    "period": ".orbital_mechanics.keplerian",
    "ra_dec": ".coordinates.sky",
    "rad0to2pi": ".coordinates.angle_units",
    "rightascension2hourangle": ".coordinates.local_equatorial",
    "sim_lonlatrad": ".coordinates.rotating_frames",
    "ssatk_load_cache": ".io.ssatk_cache",
    "ssatk_save_cache": ".io.ssatk_cache",
    "ssatk_load": ".io.ssatk_save",
    "ssatk_save": ".io.ssatk_save",
    "supported_save_formats": ".io.ssatk_save",
    "sun_ra_dec": ".coordinates.sky",
    "unit_vector": ".vectors",
    "xyz_to_ecliptic": ".coordinates.equatorial_ecliptic",
    "xyz_to_equatorial": ".coordinates.equatorial_ecliptic",
}


def _constant_names():
    global _CONSTANT_NAMES
    if _CONSTANT_NAMES is None:
        _CONSTANT_NAMES = set(import_module(".constants", __name__).__all__)
    return _CONSTANT_NAMES


def _ssapy_module():
    return import_module("ssapy")


def __getattr__(name):
    if name == "ssapy":
        module = _ssapy_module()
        globals()[name] = module
        return module
    if name in _TOOLKIT_SUBMODULE_NAMES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _constant_names():
        value = getattr(import_module(".constants", __name__), name)
        globals()[name] = value
        return value
    if name in _TOOLKIT_DUPLICATE_ALIASES:
        module = import_module(_TOOLKIT_DUPLICATE_ALIASES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SSAPY_ALIAS_NAMES:
        value = getattr(_ssapy_module(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | {"ssapy"}
        | _TOOLKIT_SUBMODULE_NAMES
        | _constant_names()
        | set(_TOOLKIT_DUPLICATE_ALIASES)
        | _SSAPY_ALIAS_NAMES
    )

try:
    from astropy.utils import iers

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 365
except ImportError:
    pass

# # Folders
# from .yastropy import *
# from .accelerations_6dof import *
# from .compute import *
# from .coordinates import *
# from .propagators_6dof import *
# from .io import *
# from .orbital_mechanics import *
# from .plots import *
# from .ssapy_wrappers import *
# from .rockets import *
# from .time_functions import *

# # Single Files
# from .asteroids import *
# from .constants import *
# from .hpc import *
# from .orbit_initializer import *
# from .utils import *
# from .vectors import *

# try:
#     import ssapy
# except ImportError:
#     pass  # ssapy simply won't be exported if not installed

# __all__ = [name for name in globals() if not name.startswith("_")]
