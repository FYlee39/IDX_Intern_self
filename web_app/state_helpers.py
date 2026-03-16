"""
This is the state helper for the web app
"""

import streamlit as st


DEFAULT_STATE = {
    "uploaded_df": None,
    "prediction_df": None,
    "upload_filename": None,
    "validation_message": None,
    "validation_passed": False,
}


def initialize_session_state():
    """
    Initialize all required session_state keys once.
    Call this at the top of every page.
    """
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def save_uploaded_data(df, filename=None):
    """
    Save uploaded raw dataframe and filename.
    """
    st.session_state["uploaded_df"] = df
    st.session_state["upload_filename"] = filename


def save_prediction_data(prediction_df):
    """
    Save prediction result dataframe.
    """
    st.session_state["prediction_df"] = prediction_df


def save_validation_result(passed, message):
    """
    Save validation result.
    """
    st.session_state["validation_passed"] = passed
    st.session_state["validation_message"] = message


def get_uploaded_data():
    """
    Return uploaded dataframe, or None if not available.
    """
    return st.session_state.get("uploaded_df")


def get_prediction_data():
    """
    Return prediction dataframe, or None if not available.
    """
    return st.session_state.get("prediction_df")


def clear_uploaded_workflow():
    """
    Clear uploaded file, validation, and prediction results.
    """
    st.session_state["uploaded_df"] = None
    st.session_state["prediction_df"] = None
    st.session_state["upload_filename"] = None
    st.session_state["validation_message"] = None
    st.session_state["validation_passed"] = False


def has_uploaded_data():
    """
    Whether uploaded dataframe exists.
    """
    return st.session_state.get("uploaded_df") is not None


def has_prediction_data():
    """
    Whether prediction dataframe exists.
    """
    return st.session_state.get("prediction_df") is not None