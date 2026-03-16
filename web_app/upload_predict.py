import streamlit as st
import pandas as pd
from utils import load_model, validate_uploaded_data, predict_from_dataframe
from state_helpers import *

initialize_session_state()

# Title
st.title("Upload and Predict")

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

st.write(
    """
    Upload a CSV file containing housing features.

    The app will validate the file and generate predicted house prices.
    """
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    save_uploaded_data(df, uploaded_file.name)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())
    st.write(f"Shape of uploaded data: {df.shape[0]} rows × {df.shape[1]} columns")

    is_valid, message = validate_uploaded_data(df)
    save_validation_result(is_valid, message)

    if is_valid:
        st.success(message)

        # Load model
        model = load_model()

        if st.button("Run Prediction"):
            result_df = predict_from_dataframe(model, df)
            save_prediction_data(result_df)
            st.dataframe(result_df.head())

            st.subheader("Prediction Result")
            st.dataframe(result_df.head())

            # Convert dataframe to CSV for download
            csv_data = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="house_price_predictions.csv",
                mime="text/csv"
            )

    else:
        st.error(message)

if st.button("Clear Current Data"):
    clear_uploaded_workflow()
    st.success("Session data cleared.")


# Footer
st.markdown("---")
st.caption(
    "Please upload a CSV file with the required feature columns. "
    "Predictions are estimates based on the trained model."
)