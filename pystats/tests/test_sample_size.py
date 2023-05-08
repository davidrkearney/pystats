import unittest
from pystats.sample_size import calculate_sample_size

class TestCalculateSampleSize(unittest.TestCase):
    def test_sample_size_calculation(self):
        effect_size = 0.5
        alpha = 0.05
        power = 0.8
        ratios = [0.25, 0.25, 0.25, 0.25]
        sample_sizes, mde = calculate_sample_size(effect_size, alpha, power, ratios)
        self.assertEqual(len(sample_sizes), 4)
        self.assertEqual(mde, effect_size)
        self.assertAlmostEqual(sum(ratios), 1, places=2)

    def test_invalid_input(self):
        effect_size = 0.5
        alpha = 0.05
        power = 0.8
        ratios = [0.3, 0.3, 0.3]
        with self.assertRaises(ValueError):
            calculate_sample_size(effect_size, alpha, power, ratios)

if __name__ == "__main__":
    unittest.main()
