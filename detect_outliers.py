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
