"""SSAPy Toolkit.

Utilities for orbital mechanics, plotting, coordinate transforms,
integrators, io helpers, and SSAPy-related workflows.
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
        "accelerations",
        "asteroids",
        "compute",
        "constants",
        "coordinates",
        "data",
        "demo_gallery",
        "engines",
        "hpc",
        "integrators",
        "io",
        "orbital_mechanics",
        "plots",
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
    "cart2sph_deg": ".coordinates.cartesian_to_spherical",
    "cart_to_cyl": ".coordinates.cartesian_to_cylindrical",
    "dd_to_dms": ".time_functions.convert_dd_and_dms",
    "dd_to_hms": ".time_functions.convert_dd_and_hms",
    "deg0to360": ".coordinates.unit_conversions",
    "deg0to360array": ".coordinates.unit_conversions",
    "deg90to90": ".coordinates.unit_conversions",
    "deg90to90array": ".coordinates.unit_conversions",
    "dms_to_dd": ".time_functions.convert_dd_and_dms",
    "dms_to_deg": ".coordinates.unit_conversions",
    "dms_to_rad": ".coordinates.unit_conversions",
    "ecliptic_to_equatorial": ".coordinates.equitorial_and_ecliptic",
    "ecliptic_xyz_to_equatorial": ".coordinates.equitorial_and_ecliptic",
    "ecliptic_xyz_to_equatorial_xyz": ".coordinates.equitorial_and_ecliptic",
    "einsum_norm": ".vectors",
    "equatorial_to_ecliptic": ".coordinates.equitorial_and_ecliptic",
    "equatorial_to_horizontal": ".coordinates.local_and_equitorial",
    "equatorial_xyz_to_ecliptic_xyz": ".coordinates.equitorial_and_ecliptic",
    "hms_to_dd": ".time_functions.convert_dd_and_hms",
    "horizontal_to_equatorial": ".coordinates.local_and_equitorial",
    "inert2rot": ".coordinates.earth_trojan_sim",
    "load_earth_file": ".plots.plotutils",
    "load_moon_file": ".plots.plotutils",
    "lonlat_distance": ".coordinates.on_sky_distance",
    "norm": ".vectors",
    "normSq": ".vectors",
    "normed": ".vectors",
    "periapsis": ".orbital_mechanics.keplerian",
    "period": ".orbital_mechanics.keplerian",
    "ra_dec": ".coordinates.sky_angles",
    "rad0to2pi": ".coordinates.unit_conversions",
    "sim_lonlatrad": ".coordinates.earth_trojan_sim",
    "sun_ra_dec": ".coordinates.sky_angles",
    "unit_vector": ".vectors",
    "xyz_to_ecliptic": ".coordinates.equitorial_and_ecliptic",
    "xyz_to_equatorial": ".coordinates.equitorial_and_ecliptic",
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
except Exception:
    pass

# # Folders
# from .yastropy import *
# from .accelerations import *
# from .compute import *
# from .coordinates import *
# from .integrators import *
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
