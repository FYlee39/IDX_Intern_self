import streamlit as st


if __name__ == '__main__':

    st.set_page_config(
        page_title="House Price Prediction App",
        page_icon="🏠",
        layout="wide"
    )

    pg = st.navigation(
        {
            "Main": [
                st.Page("home.py", title="Home", icon="🏠"),
                st.Page("upload_predict.py", title="Upload and Predict", icon="📤"),
            ],
            "Information": [
                st.Page("model_info.py", title="Model Info", icon="📊"),
                st.Page("about.py", title="About", icon="ℹ️"),
            ],
        }
    )

    pg.run()