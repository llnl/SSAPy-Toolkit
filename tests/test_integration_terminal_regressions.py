"""Actual solve_ivp integration regressions; no external ephemerides required."""

import unittest

import numpy as np

from ssapy_toolkit.propagators_6dof.sixdof import propagate_6dof


class TerminalEventTests(unittest.TestCase):
    def propagate(self, epoch, offsets, events):
        return propagate_6dof(
            r0=[7e6, 0, 0], v0=[0, 7500, 0], t0=epoch,
            times=epoch + np.array(offsets), inertia=np.eye(3), mu=0,
            events=events, max_step=0.1, dense_output=True,
        )

    @staticmethod
    def event(epoch, offset, terminal=False):
        def crossing(t, y):
            return t - (epoch + offset)
        crossing.terminal = terminal
        return crossing

    def test_monitor_before_stop_preserves_final_state_and_time_order(self):
        for epoch in (0.0, 1.4e9, 3.8e9):
            with self.subTest(epoch=epoch):
                events = [self.event(epoch, 0.25), self.event(epoch, 1.25, True)]
                tr = self.propagate(epoch, [0, 0.5, 1, 1.5, 2], events)
                self.assertEqual(tr.status, 1)
                self.assertTrue(np.all(np.diff(tr.t) > 0))
                self.assertEqual(tr.t[-1], tr.t_events[1][-1])
                np.testing.assert_allclose(tr.r[-1], [7e6, 7500 * 1.25, 0], rtol=0, atol=0.01)
                np.testing.assert_allclose(tr.r[-1], tr.y_events[1][-1, :3], rtol=0, atol=0)

    def test_stop_before_first_requested_sample(self):
        tr = self.propagate(0, [1, 2], self.event(0, 0.25, True))
        np.testing.assert_allclose(tr.t, [0.25], rtol=0, atol=1e-12)
        self.assertEqual(tr.r.shape, (1, 3))

    def test_occurrence_count_uses_last_root(self):
        def event(t, y):
            return np.cos(np.pi * t)
        event.terminal = 2
        tr = self.propagate(0, [0, 1, 2, 3], event)
        np.testing.assert_allclose(tr.t, [0, 1, 1.5], rtol=0, atol=1e-12)

    def test_nearby_sample_does_not_replace_terminal_state(self):
        tr = self.propagate(0, [0, 1.25 - 5e-7, 2], self.event(0, 1.25, True))
        self.assertEqual(len(tr.t), 3)
        self.assertEqual(tr.t[-1], tr.t_events[0][-1])

    def test_stop_on_sample_is_not_duplicated(self):
        tr = self.propagate(0, [0, 1.25, 2], self.event(0, 1.25, True))
        np.testing.assert_array_equal(tr.t, [0, 1.25])
