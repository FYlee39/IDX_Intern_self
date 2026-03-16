import numpy as np
from pathlib import Path
import joblib
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Define required features expected by the trained pipeline
# Need modified
REQUIRED_FEATURES = [
    'BuyerAgentAOR', 'ListAgentAOR', 'ViewYN', 'PoolPrivateYN', 'CloseDate',
       'ListAgentFirstName', 'ListAgentLastName', 'Latitude', 'Longitude',
       'UnparsedAddress', 'LivingArea', 'DaysOnMarket', 'ListOfficeName',
       'BuyerOfficeName', 'CoListOfficeName', 'ListAgentFullName',
       'CoListAgentFirstName', 'CoListAgentLastName', 'BuyerAgentMlsId',
       'BuyerAgentFirstName', 'BuyerAgentLastName', 'AssociationFeeFrequency',
       'MLSAreaMajor', 'CountyOrParish', 'AttachedGarageYN', 'ParkingTotal',
       'LotSizeAcres', 'SubdivisionName', 'BuyerOfficeAOR', 'YearBuilt',
       'StreetNumberNumeric', 'BathroomsTotalInteger', 'City', 'BedroomsTotal',
       'ContractStatusChangeDate', 'PurchaseContractDate',
       'ListingContractDate', 'StateOrProvince', 'FireplaceYN', 'Stories',
       'LotSizeArea', 'MainLevelBedrooms', 'NewConstructionYN', 'GarageSpaces',
       'HighSchoolDistrict', 'AssociationFee', 'LotSizeSquareFeet',
       'EmailDomain', 'ZIP_prefix', 'BuyerAgentAOR_missing',
       'ListAgentAOR_missing', 'Flooring_missing', 'ViewYN_missing',
       'PoolPrivateYN_missing', 'ListAgentFirstName_missing',
       'ListAgentLastName_missing', 'UnparsedAddress_missing',
       'BuyerOfficeName_missing', 'CoListOfficeName_missing',
       'ListAgentFullName_missing', 'CoListAgentFirstName_missing',
       'CoListAgentLastName_missing', 'BuyerAgentMlsId_missing',
       'BuyerAgentFirstName_missing', 'BuyerAgentLastName_missing',
       'AssociationFeeFrequency_missing', 'MLSAreaMajor_missing',
       'AttachedGarageYN_missing', 'SubdivisionName_missing',
       'BuyerOfficeAOR_missing', 'YearBuilt_missing',
       'StreetNumberNumeric_missing', 'City_missing',
       'PurchaseContractDate_missing', 'FireplaceYN_missing',
       'Stories_missing', 'Levels_missing', 'MainLevelBedrooms_missing',
       'NewConstructionYN_missing', 'HighSchoolDistrict_missing',
       'AssociationFee_missing', 'EmailDomain_missing', 'Levels_final',
       'CarpetYN', 'LaminateYN', 'VinylYN', 'WoodYN', 'TileYN', 'ConcreteYN',
       'StoneYN', 'SeeRemarksYN', 'BambooYN', 'BrickYN',
]


# --------------------------------------------------
# Model loading
# cache the model so it loads only once
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the whole pipline"""
    if not MODEL_PATH.exists():
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
    :param df: uploaded dataframe
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
    :param df: uploaded dataframe
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
    :param model: trained model
    :param X: input data
    """

    y_pred = model.predict(X)

    return y_pred

# --------------------------------------------------
# Post-processing
# --------------------------------------------------

def attach_predictions(df, predictions):
    """
    Add prediction column to original dataframe.
    :param df: uploaded dataframe
    :param predictions: prediction from model
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
    :param model: trained model
    :param df: uploaded dataframe
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