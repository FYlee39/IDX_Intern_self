import streamlit as st
from state_helpers import initialize_session_state
initialize_session_state()

# Title
st.title("🏠 House Price Prediction App")

st.markdown(
    """
    This Streamlit app demonstrates a batch inference workflow for residential
    housing price prediction. It is designed to make the trained model easier to
    review: upload a CSV, validate the required schema, run predictions, and
    inspect the serialized model pipeline.

    The project includes exploratory notebooks, feature engineering work, model
    comparison experiments, and a deployable web app around the final model.
    """
)

st.header("Technical Scope")

st.markdown(
    """
    - Data cleaning and feature engineering for California residential sales data
    - Regression modeling with scikit-learn-compatible pipelines
    - Streamlit interface for CSV validation and batch prediction
    - Session-state handling so uploaded data and predictions persist across pages
    """
)

st.header("Limitations")

st.markdown(
    """
    Predictions are estimates from historical data and should not be treated as
    official appraisals. Model quality depends on the uploaded data matching the
    training schema and preprocessing assumptions.
    """
)
