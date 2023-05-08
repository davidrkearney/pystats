import unittest
from statsky_notes import t_tests

class TestPairwiseTTests(unittest.TestCase):
    def test_pairwise_t_tests(self):
        data = [
            [1.2, 1.3, 1.5, 1.6, 1.7],
            [2.1, 2.2, 2.3, 2.5, 2.6],
            [3.0, 3.2, 3.4, 3.6, 3.8],
        ]
        results = perform_pairwise_t_tests(data)
        self.assertEqual(len(results), 3)  # Three pairwise comparisons
        for i, j, t_stat, p_value in results:
            self.assertLess(i, j)  # Groups should be sorted in ascending order

    def test_empty_data(self):
        data = [[]]
        with self.assertRaises(ValueError):
            perform_pairwise_t_tests(data)

if __name__ == "__main__":
    unittest.main()
