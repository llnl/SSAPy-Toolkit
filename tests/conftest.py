"""Shared fixtures for the test suite.

``gps_epoch`` exists because every propagator test in this repository used to
start at ``t0 = 0``, and several classes of defect are invisible there:

  - a relative tolerance on an epoch comparison, since ``rtol * 0 == 0``
  - SciPy's ``min_step = 10 * ulp(t)`` floor, which is ~5e-322 s at zero and
    2.4e-6 s at GPS 1.4e9
  - elapsed-versus-absolute confusion in a force model that reads its epoch

Anything a propagator is handed that carries an epoch -- ``times``, ``t0``,
``Spacecraft.t``, a burn window's ``start``/``stop``, an event threshold, a
dense-output query -- has to be offset by the same fixture value, or the test
is measuring the offset rather than the code.
"""

import pytest

# GPS seconds. 1.4e9 is 2024-05-13; 3.8e9 is 2100-06-20. Zero stays in the set
# so the parametrised tests still cover the historical behaviour.
GPS_EPOCHS = (0.0, 1.4e9, 3.8e9)


@pytest.fixture(params=GPS_EPOCHS, ids=["t0", "gps2024", "gps2100"])
def gps_epoch(request):
    """Absolute epoch, in GPS seconds, to offset every epoch-carrying input by."""
    return request.param
