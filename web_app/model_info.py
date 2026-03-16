import streamlit as st
from state_helpers import initialize_session_state
from utils import load_model
initialize_session_state()

# Title
st.title("Model Info")

model = load_model()

st.subheader("Model Structure")

if hasattr(model, "_repr_html_"):
    st.components.v1.html(model._repr_html_(), height=400, scrolling=True)
else:
    st.code(str(model))