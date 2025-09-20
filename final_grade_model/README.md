# Analiza danych i model predykcyjny wyników uczniów w matematyce

Projekt końcowy w ramach studiów *AI i automatyzacja procesów biznesowych, edycja 2024*.  
Celem jest eksploracja danych dotyczących uczniów oraz budowa modelu regresyjnego przewidującego wyniki końcowe na podstawie cech demograficznych, społecznych i edukacyjnych.

##  Struktura repozytorium

projekt_koncowy_AIB2024_KSIT_SLES/
│
├── data/ # dane
│
├── notebooks/
│ └── EDA.ipynb # analiza eksploracyjna danych
│
├── src/
│ ├── config/ # pliki konfiguracyjne
│ ├── models/ # trenowanie, tuning i ewaluacja modeli
│ ├── pipelines/ # pipeline'y ML (trening/predykcja)
│ └── utils/ # funkcje pomocnicze
│
├── tests/ # testy jednostkowe
│
├── requirements.txt # lista wymaganych bibliotek
├── README.md # opis projektu
└── .gitignore # ignorowane pliki


## Dane

- Dataset: [Math Students – Kaggle](https://www.kaggle.com/datasets/adilshamim8/math-students)  
- Format: CSV (`Math-Students.csv`)  
- Dane zawierają cechy dotyczące uczniów (m.in. czas nauki, absencje, alkohol, edukacja rodziców, płeć, itp.) oraz oceny cząstkowe i końcowe.  

Dane **nie są przechowywane w repozytorium** (`data/` jest w `.gitignore`).  
Aby pobrać dane:
```bash
kaggle datasets download -d adilshamim8/math-students -p data/raw --unzip

```

## Wymagania

Do uruchomienia projektu potrzebujesz Python 3.10+ oraz pakiety z pliku requirements.txt.

## Uruchamianie

1. Analiza danych (EDA)
Uruchom notebook:
jupyter notebook notebooks/EDA.ipynb


2. Trenowanie modelu
Pipeline treningowy:
python src/pipelines/train_pipeline.py

3. Predykcja
Użycie wytrenowanego modelu na nowych danych:
python src/pipelines/predict_pipeline.py --input data/external/new_data.csv


## Wyniki

- Modele testowane: XGBoost, RandomForest, Linear Regression
- Optymalizacja hiperparametrów: Optuna
- Metryki ewaluacyjne: RMSE, MAE, R²
- Wizualizacje: rozkłady cech, korelacje, SHAP feature importance

***Tu wkleić nasze wyniki*** 

## Technologie

Python
scikit-learn
XGBoost
Optuna
Pandas, NumPy
Matplotlib, Seaborn
SHAP
Jupyter Notebook

## Autorzy

- Sylwia Lesner (GitHub @kondratowicz95)
- Kinga Sitkowska (GitHub @kinsitko)