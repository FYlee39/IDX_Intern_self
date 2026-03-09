import streamlit as st
import pandas as pd
from utils import load_model, validate_uploaded_data, predict_from_dataframe

if __name__ == '__main__':
    # --------------------------------------------------
    # Page config
    # --------------------------------------------------
    st.set_page_config(
        page_title="House Price Prediction App",
        page_icon="🏠",
        layout="wide"
    )

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    st.title("🏠 House Price Prediction App")
    st.write(
        """
        Upload a CSV file containing housing features.
        The app will validate the file and generate predicted house prices.
        """
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = load_model()

    # --------------------------------------------------
    # File upload
    # --------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())

            st.write(f"Shape of uploaded data: {df.shape[0]} rows × {df.shape[1]} columns")

            # ------------------------------------------
            # Validate input data
            # ------------------------------------------
            is_valid, message = validate_uploaded_data(df)

            if not is_valid:
                st.error(message)
            else:
                st.success("File validation passed.")

                # --------------------------------------
                # Prediction button
                # --------------------------------------
                if st.button("Run Prediction"):
                    result_df = predict_from_dataframe(model, df)

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

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    st.markdown("---")
    st.caption(
        "Please upload a CSV file with the required feature columns. "
        "Predictions are estimates based on the trained model."
    )