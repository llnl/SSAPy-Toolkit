import unittest

import numpy as np

from ssapy_toolkit.propagators_orbit.int_utils import build_profile


class ProfileTests(unittest.TestCase):
    def test_dictionary_list_with_grid_length_is_not_samples(self):
        out = build_profile([{'start': 0, 'end': 1, 'thrust': 2},
                             {'start': 1, 'end': 2, 'thrust': 3}], [0, 10])
        np.testing.assert_array_equal(out, [2, 3])

    def test_explicit_samples_and_segments_accept_same_values(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            build_profile((1, 2), [0, 10])
        np.testing.assert_array_equal(build_profile({'samples': (1, 2)}, [0, 10]), [1, 2])
        np.testing.assert_array_equal(build_profile({'segments': [(1, 2)]}, [0, 10]), [0, 2])

    def test_integer_times_have_same_meaning_on_different_grids(self):
        profile = {'start_time': 1800, 'end_time': 2400, 'thrust': 2}
        for times in (np.arange(3601), np.arange(0, 3601, 600)):
            expected = 2 * ((times >= 1800) & (times < 2400))
            np.testing.assert_array_equal(build_profile(profile, times), expected)

    def test_invalid_indices_and_booleans_rejected(self):
        for start in (-1, True, 1800):
            with self.subTest(start=start), self.assertRaises(ValueError):
                build_profile({'start': start, 'end': 2, 'thrust': 1}, [0, 10, 20])

    def test_mixed_bounds_rejected(self):
        with self.assertRaises(ValueError):
            build_profile({'start_time': 1, 'end_index': 2}, [0, 10, 20])

    def test_existing_unambiguous_segments_and_samples(self):
        np.testing.assert_array_equal(build_profile([(1, 3, 2), (3, 1)], np.arange(5)), [0, 2, 2, 1, 1])
        np.testing.assert_array_equal(build_profile([1, 2, 3, 4, 5], np.arange(5)), [1, 2, 3, 4, 5])

    def test_malformed_sample_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            build_profile(np.ones((2, 3)), [0, 10])
