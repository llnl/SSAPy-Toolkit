import unittest

import numpy as np

from ssapy_toolkit.propagators_6dof.high_accuracy import propagate_spacecraft_segments
from ssapy_toolkit.propagators_6dof.sixdof import Spacecraft, _piecewise_solution, _piecewise_solution_sequence


def crossing(epoch):
    def event(t, y):
        return t - epoch
    return event


class SegmentEventTests(unittest.TestCase):
    def run_segments(self, first, second):
        return propagate_spacecraft_segments(
            Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], inertia=np.eye(3)),
            [dict(times=[0, 1], events=first), dict(times=[1, 2], events=second)],
            mu=0,
        )

    def test_shared_functions_merge_by_identity(self):
        a, b = crossing(0.25), crossing(1.25)
        tr = self.run_segments([a, b], [b, a])
        self.assertEqual(tr.event_functions, (a, b))
        np.testing.assert_allclose(tr.t_events, [[0.25], [1.25]], rtol=0, atol=1e-12)

    def test_different_functions_in_same_slot_stay_separate(self):
        a, b = crossing(0.25), crossing(1.25)
        tr = self.run_segments([a], [b])
        self.assertEqual(tr.event_functions, (a, b))
        self.assertEqual(len(tr.t_events), 2)

    def test_segment_with_no_events_is_supported(self):
        event = crossing(1.25)
        tr = self.run_segments(None, [event])
        self.assertEqual(tr.event_functions, (event,))
        np.testing.assert_allclose(tr.t_events[0], [1.25], rtol=0, atol=1e-12)

    def test_continuous_boundary_event_not_duplicated(self):
        event = crossing(1.0)
        tr = self.run_segments([event], [event])
        self.assertEqual(len(tr.t_events[0]), 1)

    def test_pairwise_dense_output_accepts_empty_queries(self):
        def value(t):
            return np.ones((13, np.asarray(t).size)) if np.ndim(t) else np.ones(13)
        self.assertEqual(_piecewise_solution(value, value, 1)([]).shape, (13, 0))

    def test_flat_dense_grouping_matches_independent_segment_evaluation(self):
        rng = np.random.default_rng(218)
        breaks = np.arange(1., 50.)
        coefficients = rng.normal(size=(50, 13, 2))
        def factory(coefficient):
            def solution(t):
                t = np.asarray(t)
                return coefficient[:, 0, None] + coefficient[:, 1, None] * t if t.ndim else coefficient[:, 0] + coefficient[:, 1] * t
            return solution
        solutions = [factory(coefficient) for coefficient in coefficients]
        combined = _piecewise_solution_sequence(solutions, breaks)
        queries = np.concatenate((rng.uniform(0., 50., 500), breaks, breaks))
        rng.shuffle(queries)
        # Scalar interval selection is an independent oracle, including the
        # pre-impulse convention at duplicate exact segment boundaries.
        expected = np.column_stack([solutions[sum(t > breaks)](t) for t in queries])
        np.testing.assert_array_equal(combined(queries), expected)
