"""Angular momentum balances for changing inertia and propellant outflow."""

import unittest

import numpy as np
from scipy.optimize import brentq

from ssapy_toolkit.coordinates.attitude import rotate_vector
from ssapy_toolkit.propagators_6dof.sixdof import (
    Spacecraft, _body_inertia_model, propagate_6dof,
)
from ssapy_toolkit.propagators_6dof.variational import propagate_6dof_variational
from ssapy_toolkit.satellites import SpacecraftBody, Tank


class VariableInertiaTests(unittest.TestCase):
    def test_time_dependent_inertia_conserves_momentum_at_absolute_epochs(self):
        for epoch in (0.0, 1.4e9, 3.8e9):
            for analytic in (False, True):
                with self.subTest(epoch=epoch, analytic=analytic):
                    times = epoch + np.linspace(0.0, 1.0, 11)

                    def inertia(t, r, v, q, omega):
                        return (1.0 + t - epoch) * np.eye(3)

                    options = {"inertia_rate": np.eye(3)} if analytic else {}
                    tr = propagate_6dof(
                        r0=[7e6, 0, 0], v0=[0, 7500, 0], omega0=[0, 0, 0.1],
                        times=times, t0=epoch, inertia=inertia, mu=0, dense_output=True, **options,
                    )
                    np.testing.assert_allclose(
                        (1 + tr.t - epoch) * tr.omega[:, 2], 0.1, rtol=0, atol=2e-8,
                    )
                    query = times[[5, 2, 9, 0]]
                    np.testing.assert_allclose(
                        (1 + query - epoch) * tr.solution(query)[12], 0.1, rtol=0, atol=2e-8,
                    )

    def test_nonaxial_closed_system_conserves_inertial_momentum(self):
        def inertia(t, r, v, q, omega):
            return np.diag([2 + 0.1 * t, 3 + 0.2 * t, 4 + 0.05 * t])

        tr = propagate_6dof(
            r0=[1, 0, 0], v0=[0, 0, 0], times=np.linspace(0, 3, 41),
            omega0=[0.1, 0.2, 0.3], inertia=inertia, mu=0, rtol=1e-11,
        )
        momentum = [rotate_vector(q, inertia(t, r, v, q, w) @ w)
                    for t, r, v, q, w in zip(tr.t, tr.r, tr.v, tr.q, tr.omega)]
        np.testing.assert_allclose(momentum, np.tile(momentum[0], (len(tr.t), 1)),
                                   rtol=0, atol=2e-9)

    def test_state_dependent_inertia_includes_translation_attitude_and_mass_rates(self):
        def inertia(t, r, v, q, omega, *, mass=None):
            return np.eye(3) * (2 + 0.1 * r[0] + 0.2 * v[0] + q[3] + 0.1 * mass)

        tr = propagate_6dof(
            r0=[1, 0, 0], v0=[0.2, 0, 0], times=np.linspace(0, 1, 21),
            acceleration=lambda *args: [0.3, 0, 0], omega0=[0, 0, 0.2],
            mass0=10, mass_flow_rate=lambda *args: 1.0, inertia=inertia, mu=0,
        )
        axial = [(inertia(t, r, v, q, w, mass=m) @ w)[2]
                 for t, r, v, q, w, m in zip(tr.t, tr.r, tr.v, tr.q, tr.omega, tr.mass)]
        np.testing.assert_allclose(axial, axial[0], rtol=0, atol=2e-9)

    def test_omega_dependent_inertia_uses_implicit_angular_mass_matrix(self):
        def inertia(t, r, v, q, omega):
            return (2 + np.dot(omega, omega) + 0.25 * t) * np.eye(3)

        times = np.linspace(0, 2, 21)
        tr = propagate_6dof(r0=[1, 0, 0], v0=[0, 0, 0], times=times,
                            inertia=inertia, omega0=[0, 0, 0.2], mu=0)
        expected = [brentq(lambda w: (2 + w * w + 0.25 * t) * w - 2.04 * 0.2,
                           0.0, 0.3) for t in times]
        np.testing.assert_allclose(tr.omega[:, 2], expected, rtol=0, atol=1e-9)

    def test_explicit_flux_sign_and_analytic_rate(self):
        seen_rates = []

        def inertia(t, r, v, q, omega, *, mass=None):
            return mass * np.eye(3)

        def rate(t, r, v, q, omega, *, mass=None, mass_rate=None):
            seen_rates.append(mass_rate)
            return mass_rate * np.eye(3)

        tr = propagate_6dof(
            r0=[1, 0, 0], v0=[0, 0, 0], times=np.linspace(0, 1, 11),
            omega0=[0, 0, 0.1], mass0=10, mass_flow_rate=lambda *args: 1.0,
            inertia=inertia, inertia_rate=rate, angular_momentum_flux=[0, 0, -0.03], mu=0,
        )
        np.testing.assert_allclose(tr.mass * tr.omega[:, 2], 1 - 0.03 * tr.t,
                                   rtol=0, atol=2e-10)
        self.assertTrue(all(rate == -1.0 for rate in seen_rates))

    def test_tank_outflow_default_and_zero_flux_override_through_dry_mass(self):
        body = SpacecraftBody(name="bus", mass=10, inertia=np.eye(3)).with_tanks(
            Tank(name="fuel", propellant_mass=2, position_body=[1, 0, 0]))
        craft = Spacecraft(r=[1, 0, 0], v=[0, 0, 0], body=body, omega=[0, 0, 0.1])
        options = dict(times=np.linspace(0, 3, 31), mass_flow_rate=lambda *args: 1.0,
                       mu=0, dense_output=True, max_step=0.1)
        corotating = craft.propagate(**options)
        closed = craft.propagate(**options, angular_momentum_flux=np.zeros(3))
        np.testing.assert_allclose(corotating.omega[:, 2], 0.1, rtol=0, atol=1e-10)
        expected_h = body.current_inertia[2, 2] * 0.1
        tensors = [body.with_current_mass(m).current_inertia[2, 2] for m in closed.mass]
        np.testing.assert_allclose(tensors * closed.omega[:, 2], expected_h, rtol=0, atol=2e-8)
        self.assertAlmostEqual(closed.mass[-1], 10.0, places=9)
        self.assertGreater(closed.omega[-1, 2], 0.1)

    def test_automatic_selected_and_proportional_tank_rate_matches_mass_properties(self):
        body = SpacecraftBody(name="bus", mass=10, inertia=np.diag([2., 3., 4.])).with_tanks(
            Tank(name="a", propellant_mass=2, position_body=[1, 0, 0]),
            Tank(name="b", propellant_mass=3, position_body=[0, 2, 0]))
        craft = Spacecraft(r=[1, 0, 0], v=[0, 0, 0], body=body)
        for selected in (None, "b"):
            with self.subTest(tank=selected):
                model = _body_inertia_model(craft, tank_name=selected)
                args = (0.0, craft.r, craft.v, craft.q, craft.omega)
                mass = body.current_mass - 0.5
                step = 1e-4
                finite_difference = (model(*args, mass=mass-step) - model(*args, mass=mass+step)) / (2*step)
                np.testing.assert_allclose(model.inertia_rate(*args, mass=mass, mass_rate=-1),
                                           finite_difference, rtol=0, atol=1e-9)

    def test_selected_tank_exhaustion_keeps_other_fuel_and_stops_direct_thrust(self):
        body = SpacecraftBody(name="bus", mass=10, inertia=np.eye(3)).with_tanks(
            Tank(name="reserve", dry_mass=1, propellant_mass=3, position_body=[1, 0, 0]),
            Tank(name="feed", dry_mass=1, propellant_mass=1, position_body=[0, 2, 0]))

        def acceleration(t, r, v, q, omega):
            return [1.0, 0, 0]

        acceleration.mass_flow_rate = lambda **kwargs: 1.0
        acceleration.tank_name = "feed"
        craft = Spacecraft(r=[1, 0, 0], v=[0, 0, 0], body=body, omega=[0, 0, 0.1])
        for stop in (False, True):
            with self.subTest(stop=stop):
                tr = craft.propagate(times=[0, 0.5, 2], mu=0, acceleration=acceleration,
                                     stop_at_dry_mass=stop, dense_output=True)
                self.assertAlmostEqual(tr.mass[-1], 15.0, places=9)
                self.assertAlmostEqual(tr.v[-1, 0], 1.0, places=8)
                self.assertAlmostEqual(tr.t[-1], 1.0 if stop else 2.0, places=9)
                np.testing.assert_allclose(tr.omega[:, 2], 0.1, rtol=0, atol=1e-10)
                final = tr.spacecraft(body=body, tank_name="feed")
                np.testing.assert_allclose([tank.propellant_mass for tank in final.body.tanks],
                                           [3.0, 0.0], rtol=0, atol=1e-9)
                self.assertAlmostEqual(final.body.current_mass, tr.mass[-1], places=9)
                if not stop:
                    np.testing.assert_allclose(tr.solution([1.25, 1.75])[3], [1, 1], rtol=0, atol=1e-8)

    def test_rate_callback_errors_propagate_once(self):
        calls = []

        def broken(t, r, v, q, omega, *, mass=None, mass_rate=None):
            calls.append(t)
            raise TypeError("inertia-rate model failed")

        with self.assertRaisesRegex(TypeError, "inertia-rate model failed"):
            propagate_6dof(r0=[1, 0, 0], v0=[0, 0, 0], times=[0, 1],
                            inertia=np.eye(3), inertia_rate=broken, mu=0)
        self.assertEqual(len(calls), 1)
        for value in (np.ones(3), np.full((3, 3), np.nan), np.triu(np.ones((3, 3)))):
            with self.assertRaisesRegex(ValueError, "inertia_rate"):
                propagate_6dof(r0=[1, 0, 0], v0=[0, 0, 0], times=[0, 1],
                                inertia=np.eye(3), inertia_rate=value, mu=0)

    def test_direct_torque_and_frame_accelerations_stop_at_depletion(self):
        body = SpacecraftBody(name="bus", mass=10, inertia=np.diag([1., 1., 2.])).with_tanks(
            Tank(name="fuel", dry_mass=1, propellant_mass=1))
        craft = Spacecraft(r=[0, 1, 0], v=[1, 0, 0], body=body)
        for channel in ("torque", "body_acceleration", "ntw_acceleration"):
            with self.subTest(channel=channel):
                vector = [0, 0, 0.2] if channel == "torque" else (
                    [0, 1, 0] if channel == "ntw_acceleration" else [1, 0, 0])

                def model(t, r, v, q, omega):
                    return vector

                model.mass_flow_rate = lambda **kwargs: 1.0
                model.tank_name = "fuel"
                tr = craft.propagate(times=[0, 0.5, 2], mu=0, **{channel: model})
                self.assertAlmostEqual(tr.mass[-1], 11.0, places=9)
                if channel == "torque":
                    self.assertAlmostEqual(tr.omega[-1, 2], 0.1, places=8)
                else:
                    self.assertAlmostEqual(tr.v[-1, 0], 2.0, places=8)

    def test_variational_rate_block_matches_closed_analytic_solution(self):
        result = propagate_6dof_variational(
            r0=[1, 0, 0], v0=[0, 0, 0], times=[0, 1], mu=0,
            omega0=[0, 0, 0.1], inertia=lambda t, *args: (1+t) * np.eye(3),
            inertia_rate=np.eye(3),
        )
        np.testing.assert_allclose(result.trajectory.omega[-1], [0, 0, 0.05], rtol=0, atol=1e-9)
        np.testing.assert_allclose(result.stm[-1, 10:13, 10:13], 0.5 * np.eye(3), rtol=0, atol=1e-8)
