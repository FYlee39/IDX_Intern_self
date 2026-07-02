import streamlit as st
from utils import get_required_features, get_sample_features
from state_helpers import initialize_session_state
initialize_session_state()

# Title
st.title("🏠 House Price Prediction App")

st.write(
    """
    This application predicts housing prices using a trained machine learning model.
    
    Users can upload a CSV file containing housing features, and the model will generate
    predicted property prices for each row in the dataset.
    """
)

st.markdown("---")

# --------------------------------------------------
# How to use the app
# --------------------------------------------------

st.header("How to Use")

st.markdown(
    """
    1. Navigate to **Upload and Predict** in the sidebar.
    2. Upload a CSV file containing the required housing features.
    3. Click **Run Prediction**.
    4. Download the results with predicted prices.
    """
)

st.markdown("---")

# --------------------------------------------------
# Required columns
# --------------------------------------------------

st.header("Required Input Columns")

features = get_required_features()

st.write(f"Your CSV file must include all {len(features)} model feature columns.")

st.table(features)

st.markdown("---")

# --------------------------------------------------
# Example CSV format
# --------------------------------------------------

st.header("Example Input Format")
sample_features = get_sample_features()
st.code(
    ",".join(sample_features)
    + "\n1800,3,2,34.05,-118.24,1998,Los Angeles,Los Angeles",
    language="csv"
)

st.caption("The example shows a subset of columns; uploaded files still need the full schema listed above.")

st.markdown("---")

# --------------------------------------------------
# Model information
# --------------------------------------------------

st.header("About the Model")

st.write(
    """
    The prediction model was trained on historical housing market data from (Jan to Nov 2025), and uses
    a machine learning pipeline that includes data preprocessing and regression modeling.
    
    Predictions represent estimated market values and should be interpreted as approximate values.
    """
)

st.markdown("---")

st.caption("Navigate using the sidebar to upload data and generate predictions.")
