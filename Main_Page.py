import streamlit as st

st.set_page_config(
    page_title="Health Risk Simulator",
    layout="centered"
)
st.markdown("""
<style>
/* Force dark theme everywhere */
html, body, [class*="css"] {
    background-color: #0b132b !important;
    color: #eaeaea !important;
}

/* Text */
label, span, p, div {
    color: #eaeaea !important;
}

/* Inputs */
input, textarea, select {
    background-color: #1c2541 !important;
    color: #ffffff !important;
}

/* Slider text */
.stSlider > div {
    color: #ffffff !important;
}

/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: #9be7ff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("MediGuard AI")

st.markdown(
    """
    Welcome! 
    Use the **sidebar** to navigate between different health risk assessments.

    - ❤️ Heart Disease Risk
    - 🩸 Diabetes Risk
    - 👱 Thyroid Risk

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
