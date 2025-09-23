# --- Internal imports ---
from pathlib import Path
import joblib

# --- External imports ---
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import optuna
import shap

# --- Local imports ---
from final_grade_model.src.utils.utils import load_kaggle_dataset   #w nowym repo usunąć "final_grade_model"
from final_grade_model.src.config.config_d import config

