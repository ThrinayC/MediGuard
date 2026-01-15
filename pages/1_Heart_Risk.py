import sys
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


uci_bundle = joblib.load("models/uci_heart_xgb_model.pkl")
uci_model = uci_bundle["model"]
uci_feature_cols = uci_bundle["feature_cols"]
uci_threshold = uci_bundle["threshold"]



# update - for button mapping 

def predict_heart_disease_probability(
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
):
    # Explicit order 
    row = [
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]

    df = pd.DataFrame([row], columns=uci_feature_cols)
    prob = uci_model.predict_proba(df)[0][1]
    return float(round(prob, 3))

    return float(round(prob, 3))

def assess_baseline_risk(
    age_x,
    gender_x,
    blood_pressure_x,
    bmi_x,
    smoking_x,
    exercise_x,
    sleep_hours_x,
    stress_x,
    sugar_x,
    diabetes_x,
    family_history_x,
    high_bp_x
):
    user_dict = {
        "Age": age_x,
        "Gender": gender_x,
        "Blood Pressure": blood_pressure_x,
        "BMI": bmi_x,
        "Smoking": smoking_x,
        "Exercise Habits": exercise_x,
        "Sleep Hours": sleep_hours_x,
        "Stress Level": stress_x,
        "Sugar Consumption": sugar_x,
        "Diabetes": diabetes_x,
        "Family Heart Disease": family_history_x,
        "High Blood Pressure": high_bp_x
    }

    df = pd.DataFrame([user_dict], columns=feature_cols)
    risk = model.predict_proba(df)[0][1]

    return round(float(risk), 3), user_dict


def assess_updated_risk(baseline_user_dict, updated_fields_dict):
    updated_user = baseline_user_dict.copy()
    updated_user.update(updated_fields_dict)

    df = pd.DataFrame([updated_user], columns=feature_cols)
    risk = model.predict_proba(df)[0][1]

    return round(float(risk), 3)

bundle = joblib.load("models/heart_disease_model.pkl")
model = bundle["model"]
feature_cols = bundle["feature_cols"]


st.title("❤️ Heart Disease Risk Simulator")

 #mode selec


st.subheader("Select assessment mode")

mode = st.radio(
    "Choose risk assessment type",
    [
        "📝 Questionnaire mode (Educational)",
        "🩺 Medical mode  (UCI clinical based)"
    ]
)


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

    h1, h2, h3 {
        color: white;
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# questionnare mode

if "Questionnaire" in mode:

    st.warning(
        "⚠️ This mode is **educational only**.\n\n"
        "It estimates relative risk based on lifestyle patterns and population data.\n"
        "It is **not a medical diagnosis **.\n"
        "*the accuracy of the model is lower than of clinical mode*" 
    )

    st.caption("Educational screening tool . not a medical diagnosis")
    st.divider()

    # ---- Lifestyle visual impact weights (UI-only) ----
    LIFESTYLE_IMPACT = {
        "Smoking": 0.06,
        "Exercise Habits": 0.04,
        "Stress Level": 0.03,
        "Sleep Hours": 0.02,
        "Sugar Consumption": 0.03
    }

    
    # taking inputs
    
    st.header("Step 1: Your current details")

    age = st.slider("Age", 18, 90, 45)

    gender = st.radio("Gender", ["Male", "Female"])
    gender_x = 1 if gender == "Male" else 0

    blood_pressure = st.slider("Blood Pressure (mmHg)", 90, 180, 140)
    bmi = st.slider("BMI", 15.0, 40.0, 28.0)

    smoking = st.selectbox("Smoking", ["Yes", "No"])
    smoking_x = 1 if smoking == "Yes" else 0

    exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
    exercise_x = {"Low": 0, "Medium": 1, "High": 2}[exercise]

    sleep_hours = st.slider("Sleep Hours", 4, 10, 6)

    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    stress_x = {"Low": 0, "Medium": 1, "High": 2}[stress]

    sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    sugar_x = {"Low": 0, "Medium": 1, "High": 2}[sugar]

    diabetes = st.radio("Diabetes", ["No", "Yes"])
    diabetes_x = 1 if diabetes == "Yes" else 0

    family_history = st.radio("Family Heart Disease", ["No", "Yes"])
    family_history_x = 1 if family_history == "Yes" else 0

    high_bp = st.radio("Diagnosed High BP", ["No", "Yes"])
    high_bp_x = 1 if high_bp == "Yes" else 0

    
    # risk pred.
    baseline_risk, baseline_user = assess_baseline_risk(
        age,
        gender_x,
        blood_pressure,
        bmi,
        smoking_x,
        exercise_x,
        sleep_hours,
        stress_x,
        sugar_x,
        diabetes_x,
        family_history_x,
        high_bp_x
    )

    st.divider()
    st.subheader("📊 Baseline Risk")

    st.metric(
        label="Estimated Heart Disease Risk",
        value=f"{baseline_risk * 100:.1f}%"
    )

    # changes 

    st.header("Prediction based on lifestyle changes*")

    updated_fields = {}

    # Conditional toggles
    quit_smoking = False
    better_exercise = False
    lower_stress = False
    better_sleep = False
    reduce_sugar = False

    if smoking_x == 1:
        quit_smoking = st.checkbox("Quit smoking")
        if quit_smoking:
            updated_fields["Smoking"] = 0

    if exercise_x < 2:
        better_exercise = st.checkbox("Increase exercise")
        if better_exercise:
            updated_fields["Exercise Habits"] = 2

    if stress_x > 0:
        lower_stress = st.checkbox("Reduce stress")
        if lower_stress:
            updated_fields["Stress Level"] = 0

    if sleep_hours < 5:
        better_sleep = st.checkbox("Improve sleep")
        if better_sleep:
            updated_fields["Sleep Hours"] = 8

    if sugar_x > 0:
        reduce_sugar = st.checkbox("Reduce sugar intake")
        if reduce_sugar:
            updated_fields["Sugar Consumption"] = 0

    if updated_fields:
        updated_risk = assess_updated_risk(baseline_user, updated_fields)

        lifestyle_bonus = 0.0
        if quit_smoking:
            lifestyle_bonus += LIFESTYLE_IMPACT["Smoking"]
        if better_exercise:
            lifestyle_bonus += LIFESTYLE_IMPACT["Exercise Habits"]
        if lower_stress:
            lifestyle_bonus += LIFESTYLE_IMPACT["Stress Level"]
        if better_sleep:
            lifestyle_bonus += LIFESTYLE_IMPACT["Sleep Hours"]
        if reduce_sugar:
            lifestyle_bonus += LIFESTYLE_IMPACT["Sugar Consumption"]

        displayed_risk = max(updated_risk - lifestyle_bonus, 0)
        delta = (baseline_risk - displayed_risk) * 100

        st.divider()
        st.subheader("🔁 Updated Risk")

        st.metric(
            label="New Estimated Risk",
            value=f"{displayed_risk * 100:.1f}%",
            delta=f"-{delta:.1f}%"
        )

    st.divider()
    st.caption("Demo model • Not for clinical use")

    st.markdown(
        """
        <style>
        .fine-print {
            position: fixed;
            bottom: 12px;
            right: 16px;
            font-size: 11px;
            text-align:right;
            color: #cfd8dc;
            opacity: 0.7;
            max-width: 320px;
        }
        </style>

        <div class="fine-print">
        Initial risk is predicted using a machine learning model trained on population data.<br>
        *Lifestyle impact simulations are <b>educational estimates</b> and not medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )


# Medi mode

else:
    st.success(
        "Medical mode uses a clinical model trained on diagnostic data.\n"
        "This still does NOT replace professional medical advice."
    )

    st.divider()
    st.header("Clinical Inputs")

    # age
    age = st.slider(
        "Age",
        18, 90, 55,
        help="Age in years. Cardiovascular risk generally increases with age."
    )

    # sex
    sex = st.radio(
        "Sex",
        ["Male", "Female"],
        help="Biological sex. Males generally have a higher baseline risk."
    )
    sex_x = 1 if sex == "Male" else 0

    # Chest pain
    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical angina",
            1: "Atypical angina",
            2: "Non-anginal pain",
            3: "No symptoms"
        }[x],
        help=(
            "Typical angina: classic heart-related pain. "
            "Atypical angina: unusual chest discomfort. "
            "Non-anginal pain: likely not heart-related. "
            "No symptoms: silent but still risky."
        )
    )

    # bp
    trestbps = st.slider(
        "Resting Blood Pressure (mmHg)",
        90, 200, 130,
        help="Blood pressure measured at rest. Higher values increase risk."
    )

    # CHOL
    chol = st.slider(
        "Cholesterol (mg/dL)",
        100, 400, 230,
        help="Total serum cholesterol. Elevated levels are linked to heart disease."
    )

    #  fastin blood sugar
    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dL",
        ["No", "Yes"],
        help="High fasting blood sugar may indicate diabetes or impaired glucose control."
    )
    fbs_x = 1 if fbs == "Yes" else 0

    # resting ecg 
    restecg = st.selectbox(
        "Resting ECG Result",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T abnormality",
            2: "Left ventricular hypertrophy"
        }[x],
        help=(
            "Normal: no ECG abnormalities. "
            "ST-T abnormality: possible ischemia. "
            "Left ventricular hypertrophy: thickened heart muscle."
        )
    )

    # max heart rate
    thalach = st.slider(
        "Maximum Heart Rate Achieved",
        70, 210, 150,
        help="Maximum heart rate achieved during exercise testing."
    )

    # ex angina
    exang = st.radio(
        "Exercise Induced Angina",
        ["No", "Yes"],
        help="Chest pain during exercise may indicate reduced blood flow to the heart."
    )
    exang_x = 1 if exang == "Yes" else 0

    # old peak 
    oldpeak = st.slider(
        "ST Depression (oldpeak)",
        0.0, 6.0, 1.0, step=0.1,
        help=(
            "ST depression induced by exercise. "
            "Higher values suggest more severe ischemia."
        )
    )

    # st slpe
    slope = st.selectbox(
        "ST Segment Slope",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",     # LOW risk
            1: "Flat",          # MEDIUM risk
            2: "Downsloping"    # HIGH risk
        }[x],
        help=(
            "Upsloping: lower risk. "
            "Flat: moderate risk. "
            "Downsloping: higher risk."
        )
    )


    # vessels
    ca = st.slider(
        "Number of Major Vessels (ca)",
        0, 3, 0,
        help="Number of major coronary vessels with detectable blockage."
    )

    # thallessmia
    thal = st.selectbox(
        "Thalassemia",
        [3, 6, 7],
        format_func=lambda x: {
            3: "Normal",
            6: "Fixed defect",
            7: "Reversible defect"
        }[x],
        help=(
            "Normal: healthy blood flow. "
            "Fixed defect: old heart damage. "
            "Reversible defect: active ischemia."
        )
    )

    st.divider()


    prob = predict_heart_disease_probability(
        age=age,
        sex=sex_x,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs_x,
        restecg=restecg,
        thalach=thalach,
        exang=exang_x,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal
    )

    risk_pct = prob * 100

    st.subheader("📊 Risk Result")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%"},
            title={"text": "Estimated Heart Disease Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 30], "color": "#4CAF50"},
                    {"range": [30, 48], "color": "#FFC107"},
                    {"range": [48, 100], "color": "#F44336"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": risk_pct
                }
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"uci_gauge_{risk_pct:.4f}"
)


    st.divider()
    st.caption("Clinical model • Educational use only")
    st.markdown(
    """
    <style>
    .uci-fine-print {
        position: fixed;
        bottom: 12px;
        right: 16px;
        font-size: 11px;
        color: #cfd8dc;
        opacity: 0.7;
        text-align:right;
        max-width: 340px;
        line-height: 1.4;
    }
    </style>

    <div class="uci-fine-print">
    Medical risk estimation is based on a machine learning model trained on the 
    </b>UCI Heart Disease dataset (Cleveland Clinic Foundation), originally
    published for academic research and benchmarking.</b>
    This tool is intended for </b>educational and demonstrative purposes only</b>
    and is not a substitute for professional medical evaluation.
    </div>
    """,
    unsafe_allow_html=True
)
