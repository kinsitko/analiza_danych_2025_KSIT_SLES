# Wczytywanie danych
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile, os

def load_kaggle_dataset(dataset_name: str, zip_file_name: str, csv_file_name: str, data_path: str = 'data') -> pd.DataFrame:
    api = KaggleApi()
    api.authenticate()
    # if .... plikn istnieje
    os.makedirs(data_path, exist_ok=True) 
    api.dataset_download_files(dataset_name, path=data_path, unzip=False)
    
    zip_path = os.path.join(data_path, zip_file_name)

    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Pliki w ZIP:", z.namelist())
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_file_name) as f:
            df = pd.read_csv(f)
    
    return df

