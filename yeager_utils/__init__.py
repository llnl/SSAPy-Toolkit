"""Top-level package exports for Yeager Utilities.

This package collects orbital mechanics, plotting, IO, and related helpers
into a single namespace for convenient importing.
"""

from astropy.utils import iers
iers.conf.auto_download = True       # download fresh IERS A as needed
iers.conf.auto_max_age = 365         # days; adjust to your policy

# Folders
from .Yastropy import *
from .Accelerations import *
from .Compute import *
from .Coordinates import *
from .Integrators import *
from .IO import *
from .Orbital_Mechanics import *
from .Plots import *
from .SSAPy_wrappers import *
from .Rockets import *
from .Time_Functions import *

# Single Files
from .asteroids import *
from .constants import *
from .hpc import *
from .orbit_initializer import *
from .utils import *
from .vectors import *

try:
    import ssapy
except ImportError:
    pass  # ssapy simply won't be exported if not installed

__all__ = [name for name in globals() if not name.startswith("_")]
