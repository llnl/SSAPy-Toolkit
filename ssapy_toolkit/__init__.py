"""SSAPy Toolkit.

Utilities for orbital mechanics, plotting, coordinate transforms,
integrators, IO helpers, and SSAPy-related workflows.
"""

__version__ = "1.0.0"

try:
    from astropy.utils import iers

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 365
except Exception:
    pass

# # Folders
# from .Yastropy import *
# from .Accelerations import *
# from .Compute import *
# from .Coordinates import *
# from .Integrators import *
# from .IO import *
# from .Orbital_Mechanics import *
# from .Plots import *
# from .SSAPy_wrappers import *
# from .Rockets import *
# from .Time_Functions import *

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
