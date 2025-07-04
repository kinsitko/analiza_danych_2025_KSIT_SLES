import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr


# 2. Cramér’s V dla zmiennych kategorycznych
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def cramers_v_matrix(df, cat_cols):
    results = pd.DataFrame(index=cat_cols, columns=cat_cols)
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                results.loc[col1, col2] = 1.0
            else:
                results.loc[col1, col2] = cramers_v(df[col1], df[col2])
    results = results.astype(float)
    print("Macierz Cramér’s V dla zmiennych kategorycznych:")
    print(results)
    plt.figure(figsize=(8,6))
    sns.heatmap(results, annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.title("Cramér’s V między zmiennymi kategorycznymi")
    plt.show()
    return results

# 3. Mutual Information między zmiennymi numerycznymi a kategoryczną zmienną celu
def mutual_info_numeric_cat(df, target_cat_col):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[num_cols]
    y = df[target_cat_col].astype('category').cat.codes
    mi = mutual_info_classif(X, y, discrete_features=False)
    mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=False)
    print(f"Informacja wzajemna między numerycznymi a '{target_cat_col}':")
    print(mi_series)
    return mi_series

def spearman_corr_numeric(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr, pvals = spearmanr(df[num_cols])
    corr_df = pd.DataFrame(corr, index=num_cols, columns=num_cols)
    print("Korelacje Spearmana (numeryczne):")
    print(corr_df)
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelacje Spearmana między zmiennymi numerycznymi")
    plt.show()
    return corr_df