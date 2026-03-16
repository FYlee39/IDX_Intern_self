"""
models.py

This module contains machine learning model classes used for prediction.
 Each model follows a consistent interface for training, prediction, and evaluation.

Each model class implements:
    - fit()
    - predict()
    - evaluate()

Models included:
    - ElasticNetModel
    - HistGBModel
    - SegmentModel
"""

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import StackingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.ticker as ticker
import pickle
import joblib
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform, loguniform