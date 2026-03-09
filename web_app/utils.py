import os
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Define required features expected by the trained pipeline
REQUIRED_FEATURES = [
    "LivingArea",
    "Beds",
    "Baths",
    "LotSize"
]


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():
    """
    Load the trained model pipeline from disk.

    Returns
    -------
    model : sklearn pipeline
        Trained pipeline used for prediction
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# --------------------------------------------------
# Feature helpers
# --------------------------------------------------

def get_required_features():
    """
    Return the list of required feature columns.
    """

    return REQUIRED_FEATURES


# --------------------------------------------------
# Input validation
# --------------------------------------------------

def validate_uploaded_data(df):
    """
    Validate the uploaded dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    bool
        Whether the file is valid
    str
        Message describing validation result
    """

    if df.empty:
        return False, "Uploaded file is empty."

    required_cols = get_required_features()

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    return True, "Validation successful."


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def prepare_features(df):
    """
    Select and reorder the feature columns used by the model.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Model-ready dataframe
    """

    features = get_required_features()

    X = df[features].copy()

    return X


# --------------------------------------------------
# Prediction logic
# --------------------------------------------------

def generate_predictions(model, X):
    """
    Run model inference.

    Parameters
    ----------
    model : sklearn pipeline
    X : pandas.DataFrame

    Returns
    -------
    numpy.ndarray
        Predicted prices
    """

    y_pred = model.predict(X)

    return y_pred

# --------------------------------------------------
# Post-processing
# --------------------------------------------------

def attach_predictions(df, predictions):
    """
    Add prediction column to original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
    predictions : array-like

    Returns
    -------
    pandas.DataFrame
    """

    result = df.copy()

    result["PredictedPrice"] = predictions

    return result


# --------------------------------------------------
# Full prediction pipeline
# --------------------------------------------------

def predict_from_dataframe(model, df):
    """
    Complete prediction workflow.

    Parameters
    ----------
    model : sklearn pipeline
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Original data with prediction column
    """

    # Prepare features
    X = prepare_features(df)

    # Run prediction
    preds = generate_predictions(model, X)

    # using log-price model
    preds = np.expm1(preds)

    # Attach predictions
    result_df = attach_predictions(df, preds)

    return result_df