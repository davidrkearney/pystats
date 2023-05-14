from typing import List, Tuple
from statsmodels.stats.power import TTestIndPower

def calculate_sample_size(
    effect_size: float,
    alpha: float,
    power: float,
    ratios: List[float],
) -> Tuple[List[int], float]:
    """
    Calculate the required sample size for multiple-group independent t-tests with specified ratios,
    and return the minimum detectable effect.

    :param effect_size: The standardized effect size (Cohen's d) to detect.
    :param alpha: The desired significance level (Type I error rate).
    :param power: The desired statistical power (1 - Type II error rate).
    :param ratios: A list of ratios for each group (must sum to 1).
    :return: A tuple containing a list of the calculated sample size for each group and the MDE.
    """
    if len(ratios) < 2:
        raise ValueError("The number of groups must be at least 2.")

    if not (0.99 <= sum(ratios) <= 1.01):
        raise ValueError("The ratios must sum to 1.")

    power_analysis = TTestIndPower()
    sample_size_total = power_analysis.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=sum(ratios) / len(ratios)
    )

    sample_sizes = [int(round(sample_size_total * ratio)) for ratio in ratios]
    return sample_sizes, effect_size
