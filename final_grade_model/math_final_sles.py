import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import joblib
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from import_data import load_kaggle_dataset

df = load_kaggle_dataset(
    dataset_name='adilshamim8/math-students',
    zip_file_name='math-students.zip',
    csv_file_name='Math-Students.csv',
    data_path='data'
)

print(df.head())

# Zmienna docelowa
target = 'G3'
df = df.drop(columns=['G1', 'G2'])

# Podział zmiennych
X = df.drop(columns=[target])
y = df[target]

# Podział na typy danych
categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# Model bazowy do selekcji cech
def select_features(X, y):
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    pipe.fit(X, y)
    model = pipe.named_steps['model']
    X_preprocessed = pipe.named_steps['preprocessor'].transform(X)
    selector = SelectFromModel(model, prefit=True)
    return selector, X_preprocessed, pipe

selector, X_transformed, full_pipe = select_features(X, y)
X_selected = selector.transform(X_transformed)

# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Optuna tuning
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    model = XGBRegressor(**params, random_state=42)
    return -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=3).mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# Finalny model
final_model = XGBRegressor(**study.best_params, random_state=42)
final_model.fit(X_train, y_train)

# Ewaluacja
y_pred = final_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Residual plot
sns.scatterplot(x=y_test, y=y_test - y_pred)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Rzeczywiste G3")
plt.ylabel("Błąd predykcji")
plt.title("Residual Plot")
plt.show()

# SHAP interpretacja
explainer = shap.Explainer(final_model, X_train.astype(np.float64))
shap_values = explainer(X_train.astype(np.float64), check_additivity=False)
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)

# Zapis modelu
joblib.dump(final_model, 'xgb_g3_model.pkl')
