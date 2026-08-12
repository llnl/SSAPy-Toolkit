"""Compatibility access to launch-site metadata.

The canonical launch-site and landing/test-site dictionaries live in
``ssapy_toolkit.launch_pads``. This module keeps the historical
``ssapy_toolkit.orbital_mechanics.launch_pads`` import path working without
maintaining a second copy of the same launch-pad records.
"""

from ssapy_toolkit.launch_pads import landing_pads, launch_pads

__all__ = ["launch_pads", "landing_pads"]
