import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from diabetes_prediction import diabetes_risk_predict


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.title("🩸 Diabetes Risk Prediction")
st.caption("Educational screening tool — not a medical diagnosis")

st.divider()

# -------------------------------------------------
# Gradient background (same style)
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0b4f6c,
            #0a2a4f,
            #2b0f3f
        );
        color: white;
    }
    h1, h2, h3 { color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
st.header("Enter clinical details")

pregnancies = st.slider("Number of Pregnancies", 0, 20, 2)

glucose = st.slider(
    "Glucose Level (mg/dL)", 50, 300, 120,
    help="Plasma glucose concentration after fasting"
)

blood_pressure = st.slider(
    "Blood Pressure (mmHg)", 40, 200, 70,
    help="Diastolic blood pressure"
)

skin_thickness = st.slider(
    "Skin Thickness (mm)", 0, 100, 20,
    help="Triceps skin fold thickness"
)

insulin = st.slider(
    "Insulin Level", 0, 900, 80,
    help="2-hour serum insulin"
)

bmi = st.slider(
    "BMI", 10.0, 70.0, 28.0,
    help="Body Mass Index"
)

st.subheader("Family History (Genetic Risk)")

st.caption("Answer a few questions to estimate genetic diabetes risk")

parent_diabetes = st.radio(
    "Do either of your parents have diabetes?",
    ["No", "Yes"]
)

sibling_diabetes = st.radio(
    "Do any of your siblings have diabetes?",
    ["No", "Yes"]
)

grandparent_diabetes = st.radio(
    "Do any grandparents have diabetes?",
    ["No", "Yes"]
)

early_onset = st.radio(
    "Did any family member develop diabetes before age 50?",
    ["No", "Yes"]
)

# -------------------------------
# DPF CALCULATION LOGIC
# -------------------------------
dpf = 0.1  # base population risk

if parent_diabetes == "Yes":
    dpf += 0.4

if sibling_diabetes == "Yes":
    dpf += 0.3

if grandparent_diabetes == "Yes":
    dpf += 0.2

if early_onset == "Yes":
    dpf += 0.3

# Clamp to reasonable bounds
dpf = min(round(dpf, 2), 2.5)

st.info(f"Estimated genetic risk score (DPF): **{dpf}**")
st.progress(min(dpf / 2.5, 1.0))


age = st.slider("Age", 18, 90, 35)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
st.divider()

if st.button("Predict Diabetes Risk"):
    risk = diabetes_risk_predict(
        pregnancies2_x=pregnancies,
        glucose2_x=glucose,
        bloodpressure2_x=blood_pressure,
        skinthickness2_x=skin_thickness,
        insulin2_x=insulin,
        bmi2_x=bmi,
        dpf2_x=dpf,
        age2_x=age
    )

    st.subheader("📊 Estimated Diabetes Risk")
    st.metric(
        label="Probability of Diabetes",
        value=f"{risk:.1f}%"
    )

# -------------------------------------------------
# FINE PRINT (bottom right)
# -------------------------------------------------
st.markdown(
    """
    <style>
    .fine-print {
        position: fixed;
        bottom: 12px;
        right: 16px;
        font-size: 11px;
        color: #cfd8dc;
        opacity: 0.7;
        max-width: 320px;
        text-align: right;
        line-height: 1.4;
    }
    </style>

    <div class="fine-print">
    Model trained on the Pima Indians Diabetes Dataset<br>
    (National Institute of Diabetes and Digestive and Kidney Diseases).<br>
    Educational use only . not medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
