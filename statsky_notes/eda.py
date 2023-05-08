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