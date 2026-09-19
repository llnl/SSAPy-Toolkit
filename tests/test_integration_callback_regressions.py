from functools import partial
import unittest
from unittest.mock import patch

import numpy as np

from ssapy_toolkit.accelerations_orbit.accel_radial import accel_radial
from ssapy_toolkit.accelerations_orbit.accel_velocity import accel_velocity
from ssapy_toolkit.accelerations_orbit.accel_inclination import accel_inclination
from ssapy_toolkit.propagators_orbit.high_accuracy import propagate_orbit_state
from ssapy_toolkit.propagators_orbit.int_utils import acceleration_adapter
from ssapy_toolkit.propagators_orbit.leap_frog import leapfrog


class CallbackTests(unittest.TestCase):
    def test_velocity_callback_integrates_known_linear_drag_solution(self):
        def drag(r, v):
            return -0.01 * v
        tr = propagate_orbit_state(r0=[7e6, 0, 0], v0=[0, 7500, 0],
                                   t0=1.4e9, times=1.4e9 + np.arange(3),
                                   mu=0, acceleration=drag)
        np.testing.assert_allclose(tr.v[-1], [0, 7500 * np.exp(-0.02), 0], rtol=1e-10, atol=1e-9)

    def test_time_first_two_argument_model(self):
        def model(epoch, r):
            return np.array([epoch - 1.4e9, r[0], 0])
        out = acceleration_adapter(model)(1.4e9 + 2, np.array([3, 0, 0]), np.zeros(3))
        np.testing.assert_array_equal(out, [2, 3, 0])

    def test_noncanonical_velocity_name_requires_declaration(self):
        def model(r, v_gcrf):
            return v_gcrf
        with self.assertRaisesRegex(TypeError, 'acceleration_signature'):
            acceleration_adapter(model)
        np.testing.assert_array_equal(acceleration_adapter(model, 'rv')(9, [1, 2, 3], [4, 5, 6]), [4, 5, 6])

    def test_unbound_magnitude_models_rejected(self):
        for model in (accel_radial, accel_velocity, accel_inclination):
            with self.subTest(model=model), self.assertRaisesRegex(TypeError, 'acceleration_signature'):
                acceleration_adapter(model)
        model = acceleration_adapter(partial(accel_radial, magnitude=2))
        np.testing.assert_array_equal(model(1.4e9, np.array([7e6, 0, 0]), np.zeros(3)), [2, 0, 0])

    def test_noninspectable_model_called_once_and_original_error_preserved(self):
        class Model:
            acceleration_signature = 'rvt'
            calls = 0
            @property
            def __signature__(self):
                raise ValueError('no signature')
            def __call__(self, *args):
                self.calls += 1
                raise TypeError('force implementation error')
        model = Model()
        with self.assertRaisesRegex(TypeError, 'force implementation error'):
            acceleration_adapter(model)(1.4e9, np.zeros(3), np.zeros(3))
        self.assertEqual(model.calls, 1)

    def test_leapfrog_does_not_mask_callback_error(self):
        def broken(r, v, t):
            raise TypeError('original force error')
        with self.assertRaisesRegex(TypeError, 'original force error'):
            leapfrog([7e6, 0, 0], [0, 7500, 0], [0, 1], accels=broken)

    def test_adapter_does_not_inspect_during_rhs_evaluation(self):
        def model(r, v):
            return np.zeros(3)
        adapted = acceleration_adapter(model)
        with patch('inspect.signature', side_effect=AssertionError('repeated inspection')):
            for _ in range(5):
                adapted(0, np.zeros(3), np.zeros(3))
