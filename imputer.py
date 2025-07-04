from statsmodels.imputation.mice import MICEData
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.impute import KNNImputer

def impute_with_mice(df: pd.DataFrame, iterations: int = 5) -> pd.DataFrame:
    """
    Imputuje brakujące dane w DataFrame za pomocą metody MICE.
    
    Parametry:
    - df: pd.DataFrame – wejściowy DataFrame z brakującymi danymi
    - iterations: int – liczba iteracji imputacji (domyślnie 5)

    Zwraca:
    - df_imputed: pd.DataFrame – DataFrame z imputowanymi wartościami
    """
    
    df_mice = df.copy()
    
    # Normalizacja nazw kolumn
    df_mice.columns = (
        df_mice.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('/', '_per_', regex=False)
    )
    
    # Kodowanie zmiennych kategorycznych
    categorical_cols = df_mice.select_dtypes(include='object').columns
    encoder = OrdinalEncoder()
    df_mice[categorical_cols] = encoder.fit_transform(df_mice[categorical_cols])
    
    # Imputacja MICE
    mice_data = MICEData(df_mice)
    for _ in range(iterations):
        mice_data.update_all()
    
    df_imputed = mice_data.data.copy()
    
    # Dekodowanie zmiennych kategorycznych z powrotem
    df_imputed[categorical_cols] = encoder.inverse_transform(df_imputed[categorical_cols])
    
    return df_imputed


def impute_with_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputuje brakujące dane w DataFrame za pomocą KNNImputer.
    Dane kategoryczne są kodowane one-hot przed imputacją.

    Parametry:
    - df: pd.DataFrame – dane wejściowe z brakującymi wartościami
    - n_neighbors: int – liczba sąsiadów KNN (domyślnie 5)

    Zwraca:
    - df_imputed: pd.DataFrame – DataFrame z imputowanymi wartościami
    """
    df_transformed = df.copy()

    # One-hot encoding dla zmiennych kategorycznych (dtype 'object')
    df_transformed = pd.get_dummies(df_transformed, drop_first=False)

    # Imputacja KNN
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_transformed)
    df_imputed = pd.DataFrame(imputed_array, columns=df_transformed.columns)

    # Debug: sprawdzenie czy pozostały jakieś NaN
    remaining_nulls = df_imputed.isnull().sum()
    if remaining_nulls.any():
        print("Kolumny z nadal brakującymi wartościami:", remaining_nulls[remaining_nulls != 0].index.tolist())

    return df_imputed

