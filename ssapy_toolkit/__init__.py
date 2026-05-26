"""SSAPy Toolkit.

Utilities for orbital mechanics, plotting, coordinate transforms,
integrators, io helpers, and SSAPy-related workflows.
"""

__version__ = "1.0.2"

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
