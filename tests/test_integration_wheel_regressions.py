"""Bounded reaction wheels with consistent body, event and dense states."""

import unittest

import numpy as np

from ssapy_toolkit.coordinates.attitude import rotate_vector
from ssapy_toolkit.propagators_6dof.high_accuracy import propagate_spacecraft_segments
from ssapy_toolkit.propagators_6dof.sixdof import Spacecraft, propagate_6dof
from ssapy_toolkit.propagators_6dof.variational import propagate_6dof_variational


class WheelConstraintTests(unittest.TestCase):
    def options(self, **changes):
        options = dict(r0=[7e6, 0, 0], v0=[0, 7500, 0], inertia=np.eye(3), mu=0,
                       wheel_axes_body=[0, 0, 1], wheel_momentum_capacity=[0.01],
                       wheel_torque=lambda *args: [0.02])
        options.update(changes)
        return options

    def test_capacity_and_restart_for_all_scipy_methods(self):
        for method in ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"):
            with self.subTest(method=method):
                tr = propagate_6dof(times=[0, 1], method=method, **self.options())
                self.assertEqual(tr.wheel_momentum[-1, 0], -0.01)
                np.testing.assert_allclose(tr.omega[-1], [0, 0, 0.01], rtol=0, atol=2e-10)
                restarted = propagate_6dof(
                    times=[1, 2], t0=1, omega0=tr.omega[-1], q0=tr.q[-1],
                    wheel_momentum0=tr.wheel_momentum[-1], method=method, **self.options(),
                )
                np.testing.assert_array_equal(restarted.wheel_momentum[:, 0], [-0.01, -0.01])
                np.testing.assert_allclose(restarted.omega[-1], tr.omega[-1], rtol=0, atol=1e-12)

    def test_reversed_command_desaturates_without_lost_momentum(self):
        def command(t, *args):
            return [0.02 if t < 1.0 else -0.02]

        times = np.linspace(0, 2, 101)
        tr = propagate_6dof(times=times, dense_output=True,
                            **self.options(wheel_torque=command, max_step=0.05))
        expected = np.where(times < 1, np.maximum(-0.02 * times, -0.01),
                            np.minimum(-0.01 + 0.02 * (times - 1), 0.01))
        self.assertTrue(np.all(np.abs(tr.wheel_momentum) <= 0.01))
        np.testing.assert_allclose(tr.wheel_momentum[:, 0], expected, rtol=0, atol=3e-9)
        np.testing.assert_allclose(tr.omega[:, 2] + tr.wheel_momentum[:, 0], 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tr.q[-1], [np.cos(0.0075/2), 0, 0, np.sin(0.0075/2)],
                                   rtol=0, atol=3e-9)

    def test_dense_event_and_callback_states_share_the_constraint(self):
        seen = []

        def command(t, r, v, q, omega, *, mass=None, wheel_momentum=None):
            seen.append(wheel_momentum.copy())
            return [0.02]

        command.accepts_wheel_momentum = True

        def stop(t, y):
            self.assertLessEqual(abs(y[13]), 0.01)
            self.assertAlmostEqual(y[12] + y[13], 0.0, places=11)
            return t - 0.8

        stop.terminal = True
        tr = propagate_6dof(times=[0, 1], dense_output=True, events=stop,
                            **self.options(wheel_torque=command))
        self.assertTrue(all(abs(h[0]) <= 0.01 for h in seen))
        self.assertAlmostEqual(tr.t[-1], 0.8, places=12)
        self.assertEqual(tr.y_events[0][-1, 13], -0.01)
        queries = np.r_[np.linspace(0, 0.8, 501)[::-1], 0.5, 0.0, 0.8]
        dense = tr.solution(queries)
        self.assertTrue(np.all(np.abs(dense[13]) <= 0.01))
        np.testing.assert_allclose(dense[12] + dense[13], 0, rtol=0, atol=2e-12)
        np.testing.assert_allclose(tr.solution(tr.t[-1]), tr.y_events[0][-1], rtol=0, atol=2e-12)
        self.assertEqual(tr.solution(np.array([])).shape, (14, 0))

    def test_multiple_wheels_preserve_inertial_momentum_with_nonaxial_rotation(self):
        axes = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
        axes /= np.linalg.norm(axes, axis=0)
        inertia = np.diag([2., 3., 4.])
        capacities = np.array([0.01, 0.02, 0.03])
        tr = propagate_6dof(
            times=np.linspace(0, 2, 101), dense_output=True,
            **self.options(inertia=inertia, omega0=[0.1, 0.2, 0.3],
                           wheel_axes_body=axes, wheel_momentum_capacity=capacities,
                           wheel_torque=lambda *args: [0.02, -0.04, 0.06], rtol=1e-11),
        )
        self.assertTrue(np.all(np.abs(tr.wheel_momentum) <= capacities))
        momentum = [rotate_vector(q, inertia @ w + axes @ h)
                    for q, w, h in zip(tr.q, tr.omega, tr.wheel_momentum)]
        np.testing.assert_allclose(momentum, np.tile(momentum[0], (len(tr.t), 1)),
                                   rtol=0, atol=3e-10)

    def test_segment_handoff_can_continue_and_desaturate(self):
        craft = Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], inertia=np.eye(3))
        tr = propagate_spacecraft_segments(
            craft, [dict(times=[0, 1], wheel_torque=lambda *args: [0.02]),
                    dict(times=[1, 2], wheel_torque=lambda *args: [-0.01])],
            mu=0, dense_output=True, wheel_axes_body=[0, 0, 1], wheel_momentum_capacity=[0.01],
        )
        np.testing.assert_allclose(tr.wheel_momentum[:, 0], [0, -0.01, 0], rtol=0, atol=1e-11)
        np.testing.assert_allclose(tr.solution([0.75, 1.5])[13], [-0.01, -0.005], rtol=0, atol=1e-11)

    def test_invalid_initial_momentum_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            propagate_6dof(times=[0, 1], wheel_momentum0=[-0.010001], **self.options())

    def test_variational_saturation_includes_change_in_hit_time(self):
        result = propagate_6dof_variational(
            times=[0, 1], jacobian_step=1e-5,
            **self.options(omega0=[0, 0, 0.03], wheel_momentum0=[0.001],
                           rtol=1e-11, atol=1e-14),
        )
        tr = result.trajectory
        self.assertEqual(tr.wheel_momentum[-1, 0], -0.01)
        np.testing.assert_allclose(result.stm[0], np.eye(14), rtol=0, atol=1e-10)
        hit = 0.55
        angle = 0.03 + 0.02 * (hit - 0.5 * hit**2)
        expected = np.zeros(14)
        expected[6] = -0.5 * np.sin(angle/2) * (1-hit)
        expected[9] = 0.5 * np.cos(angle/2) * (1-hit)
        expected[12] = 1.0
        np.testing.assert_allclose(result.stm[-1, :, 13], expected, rtol=0, atol=2e-7)

    def test_variational_initial_capacity_uses_inward_perturbation(self):
        result = propagate_6dof_variational(
            times=[0, 0.1], jacobian_step=1e-5,
            **self.options(wheel_momentum0=[-0.01], wheel_torque=lambda *args: [-0.02],
                           rtol=1e-11, atol=1e-14),
        )
        self.assertTrue(np.all(np.abs(result.trajectory.wheel_momentum) <= 0.01))
        self.assertAlmostEqual(result.stm[-1, 13, 13], 1.0, places=7)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            propagate_6dof_variational(times=[0, 1], wheel_momentum0=[0.011], **self.options())
