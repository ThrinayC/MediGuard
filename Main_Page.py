import streamlit as st

st.set_page_config(
    page_title="Health Risk Simulator",
    layout="centered"
)

st.title("MediGuard AI")

st.markdown(
    """
    Welcome! 
    Use the **sidebar** to navigate between different health risk assessments.

    - ❤️ Heart Disease Risk
    - 🩸 Diabetes Risk

    ⚠️ All predictions are **educational only** and not medical advice.
    """
)

st.divider()

st.info(
    """
    **Disclaimer**  
    This application demonstrates machine learning models trained on
    public datasets for educational and research purposes.
    It does not replace professional medical consultation.
    """
)
