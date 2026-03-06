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