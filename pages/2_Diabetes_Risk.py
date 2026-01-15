import sys
import os
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import joblib
import pandas as pd
import numpy as np

diabetes_bundle = joblib.load("models/diabetes_logreg_model.pkl")

diabetes_model = diabetes_bundle["model"]
diabetes_scaler = diabetes_bundle["scaler"]
diabetes_feature_cols = diabetes_bundle["feature_cols"]

#update- button mapping 

def diabetes_risk_predict(
    pregnancies2_x,
    glucose2_x,
    bloodpressure2_x,
    skinthickness2_x,
    insulin2_x,
    bmi2_x,
    dpf2_x,
    age2_x
):
    input_dict = {
        "Pregnancies": pregnancies2_x,
        "Glucose": glucose2_x,
        "BloodPressure": bloodpressure2_x,
        "SkinThickness": skinthickness2_x,
        "Insulin": insulin2_x,
        "BMI": bmi2_x,
        "DiabetesPedigreeFunction": dpf2_x,
        "Age": age2_x
    }

    df = pd.DataFrame([input_dict], columns=diabetes_feature_cols)

    df_scaled = diabetes_scaler.transform(df)
    prob = diabetes_model.predict_proba(df_scaled)[0][1]

    return round(prob * 100, 2)






# Page setup

st.title("🩸 Diabetes Risk Prediction")
st.caption("Educational screening tool — not a medical diagnosis")

st.divider()

# Gradient 
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


# inputs

st.header("Enter clinical details")

st.subheader("Sex")

gender = st.radio(
    "Select biological sex",
    ["Female", "Male"],
    horizontal=True
)


if gender == "Female":
    pregnancies = st.slider(
        "Number of Pregnancies",
        0, 20, 2,
        help="Only applicable for females"
    )
else:
    pregnancies = 0
   


glucose = st.slider(
    "Glucose Level (mg/dL)", 50, 300, 120,
    help="glucose concentration after fasting . enter your last known value or \n choose any value between 70-100(normal blood glucose level)"
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

st.subheader("Body Measurements")

col1, col2 = st.columns(2)

with col1:
    height_feet = st.number_input(
        "Height (feet)",
        min_value=3,
        max_value=8,
        value=5
    )

with col2:
    height_inches = st.number_input(
        "Additional inches",
        min_value=0,
        max_value=11,
        value=6
    )

weight_kg = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=200.0,
    value=70.0
)

height_m = (height_feet * 0.3048) + (height_inches * 0.0254)

# bmi calc
bmi = weight_kg / (height_m ** 2)
    
if bmi < 18.5:
        bmi_label = "Underweight"
        bmi_color = "#ff4d4d"   # red
elif bmi < 25:
        bmi_label = "Normal"
        bmi_color = "#2ecc71"   # green
elif bmi < 30:
        bmi_label = "Overweight"
        bmi_color = "#f1c40f"   # yellow
else:
        bmi_label = "Obese"
        bmi_color = "#e74c3c"   # dark red

st.markdown(
        f"""
        <div style="
            background-color: {bmi_color};
            padding: 12px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            text-align: center;
        ">
            Calculated BMI: {bmi} <br>
            Category: {bmi_label}
        </div>
        """,
        unsafe_allow_html=True
    )

bmi2_x=bmi


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

# dpf calc
dpf = 0.1 
if parent_diabetes == "Yes":
    dpf += 0.4

if sibling_diabetes == "Yes":
    dpf += 0.3

if grandparent_diabetes == "Yes":
    dpf += 0.2

if early_onset == "Yes":
    dpf += 0.3


dpf = min(round(dpf, 2), 2.5)

st.info(f"Estimated genetic risk score (DPF): **{dpf}**")
st.progress(min(dpf / 2.5, 1.0))


age = st.slider("Age", 18, 90, 35)


# predict

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

    st.subheader("📊 Risk Result")

    fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                number={"suffix": "%"},
                title={"text": "Probability of Diabetes"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, 35], "color": "#4CAF50"},
                        {"range": [35, 60], "color": "#FFC107"},
                        {"range": [60, 100], "color": "#F44336"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": risk
                    }
                }
            )
        )

    fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )

    st.plotly_chart(fig, use_container_width=True)


# fine print

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
