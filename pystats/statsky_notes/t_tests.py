from typing import List, Tuple
from scipy.stats import ttest_ind
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def pairwise_t_tests(data: List[List[float]]) -> List[Tuple[int, int, float, float]]:
    """
    Perform pairwise t-tests on experimental data.

    :param data: A list of lists, where each inner list represents a group of experimental data.
    :return: A list of tuples, each tuple containing four values: the indices of the two groups being compared,
             the t-statistic, and the p-value.
    """
    results = []
    num_comparisons = len(data) * (len(data) - 1) // 2
    with tqdm(total=num_comparisons) as pbar:
        # Iterate over all pairs of groups
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                group1 = data[i]
                group2 = data[j]

                # Perform t-test and store the result
                t_stat, p_value = ttest_ind(group1, group2)
                results.append((i, j, t_stat, p_value))

                # Create a bar plot of the two groups with colored confidence interval bars
                fig, ax = plt.subplots()
                means = [np.mean(group1), np.mean(group2)]
                sems = [np.std(group1) / np.sqrt(len(group1)), np.std(group2) / np.sqrt(len(group2))]
                cis = [1.96 * sem for sem in sems]
                if p_value < 0.05:
                    colors = ['blue', 'green']
                else:
                    colors = ['grey', 'grey']
                ax.bar([f'Group {i+1}', f'Group {j+1}'], means, yerr=cis, capsize=10, color=colors)
                ax.set_ylabel('Mean')
                ax.set_title(f'Comparison of Group {i+1} and Group {j+1}')

                pbar.update(1)

    return results
