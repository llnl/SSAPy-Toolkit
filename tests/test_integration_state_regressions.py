import unittest

import numpy as np

from ssapy_toolkit.environment import SpaceEnvironment
from ssapy_toolkit.accelerations_6dof.spacecraft import _call_density, _call_optional
from ssapy_toolkit.propagators_6dof.high_accuracy import propagate_spacecraft_segments, _propagate_spacecraft_segment
from ssapy_toolkit.propagators_6dof.sixdof import Spacecraft, propagate_6dof
from ssapy_toolkit.satellites import SpacecraftBody, Tank


class StateContinuityTests(unittest.TestCase):
    def craft(self, **kwargs):
        return Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], inertia=np.eye(3), **kwargs)

    def test_direct_burn_after_coast_preserves_mass_in_every_segment(self):
        def acceleration(t, r, v, q, omega):
            return np.zeros(3)
        acceleration.mass_flow_rate = lambda **kwargs: 1.
        tr = propagate_spacecraft_segments(self.craft(mass=10.),
            [dict(times=[0, 1]), dict(times=[1, 2], acceleration=acceleration),
             dict(times=[2, 3])], mu=0, dense_output=True)
        np.testing.assert_allclose(tr.mass, [10., 10., 9., 9.], atol=1e-10)
        np.testing.assert_allclose(tr.solution([0.5, 1.5, 2.5])[13], [10., 9.5, 9.], atol=1e-10)

    def test_ulp_close_segment_start_is_snapped_without_mutating_input(self):
        epoch = 1.4e9
        times = np.array([np.nextafter(epoch, -np.inf), epoch + 1])
        tr = propagate_spacecraft_segments(self.craft(t=epoch), [dict(times=times)], mu=0)
        self.assertEqual(tr.t[0], epoch)
        self.assertLess(times[0], epoch)

    def test_explicit_wheel_initial_state_overrides_orbit0(self):
        craft = self.craft(wheel_momentum=[0.2])
        tr = propagate_6dof(orbit0=craft, times=[0, 1], inertia=np.eye(3), mu=0,
                            wheel_axes_body=[0, 0, 1], wheel_momentum0=[0.1])
        np.testing.assert_allclose(tr.wheel_momentum[:, 0], [0.1, 0.1])

    def test_bodyless_wheel_state_survives_spacecraft_propagation(self):
        tr = self.craft(wheel_momentum=[0.2]).propagate(
            times=[0, 1], mu=0, wheel_axes_body=[0, 0, 1])
        np.testing.assert_allclose(tr.wheel_momentum[:, 0], [0.2, 0.2])

    def test_segment_handoff_preserves_selected_tank_and_updated_inertia(self):
        body = SpacecraftBody.box(name='bus', mass=10., size=(1., 1., 1.)).with_tanks(
            Tank(propellant_mass=2., dry_mass=1., name='main', position_body=[2., 0, 0]),
            Tank(propellant_mass=3., dry_mass=1., name='aux', position_body=[0, 2., 0]))
        craft = Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], body=body)
        def mass_flow(t, r, v, q, omega):
            return 1.
        mass_flow.tank_name = 'aux'
        _, final, _ = _propagate_spacecraft_segment(craft,
            dict(times=[0, 1], mass_flow_rate=mass_flow, mu=0), tracks_mass=True)
        expected = body.with_tank_propellant_mass('aux', 2.)
        np.testing.assert_allclose([tank.propellant_mass for tank in final.body.tanks], [2., 2.], atol=1e-10)
        np.testing.assert_allclose(final.inertia, expected.current_inertia, atol=1e-10)
        np.testing.assert_allclose(final.inertia, final.body.current_inertia, atol=1e-10)

    def test_segment_inertia_override_is_retained_at_handoff(self):
        inertia = np.diag([2., 3., 4.])
        _, final, _ = _propagate_spacecraft_segment(self.craft(),
            dict(times=[0, 1], inertia=inertia, mu=0))
        np.testing.assert_array_equal(final.inertia, inertia)


class ModelErrorTests(unittest.TestCase):
    def test_environment_density_original_typeerror_is_not_retried(self):
        calls = []
        def density(altitude, *args):
            calls.append(args)
            raise TypeError('broken density formula')
        env = SpaceEnvironment(atmosphere_density_model=density)
        with self.assertRaisesRegex(TypeError, 'broken density formula'):
            env.density(200e3, 1.4e9, [7e6, 0, 0])
        self.assertEqual(len(calls), 1)

    def test_environment_time_only_and_full_vector_callbacks(self):
        env = SpaceEnvironment(sun_position_model=lambda t: [t, 0, 0])
        np.testing.assert_array_equal(env.sun_position(1), [1, 0, 0])
        calls = []
        def broken(*args):
            calls.append(args)
            raise TypeError('broken ephemeris')
        env = SpaceEnvironment(sun_position_model=broken)
        with self.assertRaisesRegex(TypeError, 'broken ephemeris'):
            env.sun_position(1)
        self.assertEqual(len(calls), 1)

    def test_force_optional_and_density_callbacks_are_called_once(self):
        for evaluator in (_call_optional, _call_density):
            calls = []
            def broken(*args):
                calls.append(args)
                raise TypeError('force input failure')
            args = (0, np.zeros(3), np.zeros(3), [1, 0, 0, 0], np.zeros(3), None)
            if evaluator is _call_density:
                args = (200e3, *args)
            with self.assertRaisesRegex(TypeError, 'force input failure'):
                evaluator(broken, *args)
            self.assertEqual(len(calls), 1)

    def test_inertia_mass_callback_error_is_not_retried_without_mass(self):
        calls = []
        def inertia(t, r, v, q, omega, *, mass=None):
            calls.append(mass)
            raise TypeError('broken inertia formula')
        with self.assertRaisesRegex(TypeError, 'broken inertia formula'):
            propagate_6dof(r0=[7e6, 0, 0], v0=[0, 7500, 0], times=[0, 1],
                            inertia=inertia, mu=0, mass0=10.)
        self.assertEqual(calls, [10.])

    def test_inertia_without_mass_keyword_remains_supported(self):
        def inertia(t, r, v, q, omega):
            return np.eye(3)
        tr = propagate_6dof(r0=[7e6, 0, 0], v0=[0, 7500, 0], times=[0, 1],
                            inertia=inertia, mu=0, mass0=10.)
        np.testing.assert_array_equal(tr.mass, [10., 10.])
