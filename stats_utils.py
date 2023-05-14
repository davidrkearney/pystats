import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def basic_eda(df: pd.DataFrame) -> None:
    """
    Perform basic exploratory data analysis on a DataFrame's columns.

    :param df: A pandas DataFrame.
    """
    print("Summary statistics:")
    print(df.describe())
    
    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nUnique values:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nCorrelations:")
    corr = df.corr()
    print(corr)
    
    print("\nCorrelations with background gradient:")
    corr_styled = corr.style.background_gradient().set_precision(2)
    display(corr_styled)

    print("\nBoxplots:")
    num_cols = df.select_dtypes(include=["number"]).columns
    non_num_cols = df.select_dtypes(exclude=["number"]).columns

    for non_num_col in non_num_cols:
        for num_col in num_cols:
            plt.figure()
            sns.boxplot(x=non_num_col, y=num_col, data=df)
            plt.title(f"Boxplot of {num_col} by {non_num_col}")
            plt.show()

    print("\nDistribution plots:")
    for num_col in num_cols:
        plt.figure()
        sns.displot(df[num_col], kde=True)
        plt.title(f"Distribution plot of {num_col}")
        plt.show()

    print("\nCountplots:")
    for non_num_col in non_num_cols:
        plt.figure()
        sns.countplot(x=non_num_col, data=df)
        plt.title(f"Countplot of {non_num_col}")
        plt.show()

    print("\nCorrelation Heatmap:")
    plt.figure()
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap')
    plt.show()

def linear_regression(X, y):
    """
    Performs linear regression on the data set X, y and plots the line of best fit
    """
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Calculate the slope and y-intercept of the line of best fit
    m, b = np.polyfit(X.ravel(), y, 1)
    
    # Plot the data points and the line of best fit
    plt.scatter(X, y)
    plt.plot(X, m*X + b, color='red')
    plt.show()
    
    # Return the slope and y-intercept of the line of best fit
    return m, b


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



from typing import List, Tuple

def detect_outliers(
    data: List[float], factor: float = 1.5, remove: bool = False
) -> Tuple[List[float], List[int], List[float]]:
    """
    Detect outliers in the data using the IQR method.

    :param data: A list of numeric data.
    :param factor: The IQR factor to use for detecting outliers (default: 1.5).
    :param remove: Whether to remove the outliers from the data (default: False).
    :return: A tuple containing the processed data, the indices of the detected outliers, and the outlier values.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outlier_indices = [
        index for index, value in enumerate(data) if value < lower_bound or value > upper_bound
    ]

    outlier_values = [data[index] for index in outlier_indices]

    if remove:
        data = [value for value in data if lower_bound <= value <= upper_bound]

    return data, outlier_indices, outlier_values
