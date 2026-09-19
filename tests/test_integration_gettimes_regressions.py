"""Sample-count regressions for :func:`ssapy_toolkit.time_functions.get_times`.

A caller that derives its step from a sample count asks for an exact integer
number of intervals. ``duration / freq`` is computed in binary floating point,
so that quotient can land one unit in the last place below the integer it
should equal. Truncating it silently returned a grid one sample short, which
left position and time arrays misaligned downstream.
"""

import unittest

import numpy as np

from ssapy_toolkit.time_functions.get_times import get_times


class GetTimesSampleCountTests(unittest.TestCase):
    def test_demo_case_returns_every_requested_sample(self):
        # Exact values taken from demos/orbital_mechanics/
        # demo_ellipse_fit_against_ssapy.py in fast mode, where
        # duration / freq evaluates to 118.99999999999999 rather than 119.
        duration = 23288.287991979163
        freq = 195.6998990922619
        self.assertLess(duration / freq, 119.0)
        self.assertEqual(len(get_times(duration=(duration, "s"), freq=(freq, "s"))), 120)

    def test_sample_count_matches_request_across_grid_sizes(self):
        duration = 23315.0
        for count in (3, 5, 120, 170, 180, 182, 256, 339, 356, 400, 1000):
            with self.subTest(count=count):
                freq = duration / (count - 1)
                times = get_times(duration=(duration, "s"), freq=(freq, "s"))
                self.assertEqual(len(times), count)
                np.testing.assert_allclose(
                    np.diff(times.gps), freq, rtol=0.0, atol=1e-6
                )

    def test_endpoints_are_preserved(self):
        duration = 23288.287991979163
        freq = 195.6998990922619
        times = get_times(duration=(duration, "s"), freq=(freq, "s"), t0=1.0e9)
        self.assertAlmostEqual(times.gps[0], 1.0e9, places=6)
        self.assertAlmostEqual(times.gps[-1], 1.0e9 + duration, places=6)

    def test_partial_final_interval_keeps_existing_behaviour(self):
        # A non-integer ratio must be untouched by the snap. 100 / 30 spans
        # three whole steps, so the grid holds four samples, and get_times
        # then stretches them across the full duration: the spacing becomes
        # 100/3 s and the last epoch is the requested end, not 90 s.
        times = get_times(duration=(100.0, "s"), freq=(30.0, "s"), t0=0.0)
        self.assertEqual(len(times), 4)
        self.assertAlmostEqual(times.gps[-1], 100.0, places=6)
        np.testing.assert_allclose(
            np.diff(times.gps), 100.0 / 3.0, rtol=0.0, atol=1e-6
        )

    def test_tf_anchored_grid_matches_request(self):
        duration = 23288.287991979163
        freq = 195.6998990922619
        times = get_times(duration=(duration, "s"), freq=(freq, "s"), tf=2.0e9)
        self.assertEqual(len(times), 120)
        self.assertAlmostEqual(times.gps[-1], 2.0e9, places=6)


if __name__ == "__main__":
    unittest.main()
