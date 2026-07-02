from pathlib import Path
import joblib
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Feature columns expected by the trained pipeline, in model input order.
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

SAMPLE_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "Latitude",
    "Longitude",
    "YearBuilt",
    "City",
    "CountyOrParish",
]


# --------------------------------------------------
# Model loading
# cache the model so it loads only once
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
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


def get_sample_features():
    """
    Return a short subset of feature columns for documentation and UI examples.
    """

    return SAMPLE_FEATURES


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

    missing_cols = get_missing_features(df)

    if missing_cols:
        preview = ", ".join(missing_cols[:10])
        remaining = len(missing_cols) - 10
        suffix = f" and {remaining} more" if remaining > 0 else ""
        return False, f"Missing {len(missing_cols)} required column(s): {preview}{suffix}."

    return True, "Validation successful."


def get_missing_features(df):
    """
    Return required model features that are not present in the dataframe.
    """

    return [c for c in get_required_features() if c not in df.columns]


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def prepare_features(df):
    """
    Select and reorder the feature columns used by the model.
    :param df: uploaded dataframe
    """

    missing_cols = get_missing_features(df)
    if missing_cols:
        raise ValueError(
            "Cannot prepare features because required columns are missing: "
            + ", ".join(missing_cols)
        )

    X = df[get_required_features()].copy()

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
    preds_log = generate_predictions(model, X)

    return preds_log
