import streamlit as st
from uci_heart_prediction import predict_heart_disease_probability


# import ONLY the functions from your ML file
from heart_disease_prediction import (
    assess_baseline_risk,
    assess_updated_risk
)

st.set_page_config(
    page_title="Heart Disease Risk Demo",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Simulator")

# ======================
# MODE SELECTION
# ======================

st.subheader("Select assessment mode")

mode = st.radio(
    "Choose risk assessment type",
    [
        "📝 Questionnaire mode (Educational)",
        "🩺 Medical mode  (UCI clinical based)"
    ]
)

# -------------------------------------------------
# Gradient background
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

# =================================================
# QUESTIONNAIRE MODE
# =================================================
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

    # ======================
    # BASELINE INPUTS
    # ======================
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

    # ======================
    # BASELINE RISK
    # ======================
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

    # ======================
    # LIFESTYLE CHANGES
    # ======================
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
            text-align: right;
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

# =================================================
# MEDICAL MODE (placeholder for UCI)
# =================================================
else:
    st.success(
        "Medical mode uses a clinical model trained on diagnostic data.\n"
        "This still does NOT replace professional medical advice."
    )

    st.divider()
    st.header("Clinical Inputs")

    # ---------- AGE ----------
    age = st.slider("Age", 18, 90, 55)

    # ---------- SEX ----------
    sex = st.radio("Sex", ["Male", "Female"])
    sex_x = 1 if sex == "Male" else 0

    # ---------- CHEST PAIN ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical angina",
                1: "Atypical angina",
                2: "Non-anginal pain",
                3: "No symptoms"
            }[x]
        )
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                Chest pain categories:
                - Typical angina: classic heart-related pain
                - Atypical angina: unusual chest discomfort
                - Non-anginal pain: likely not heart-related
                - No symptoms: silent but still risky
                """
            )

    # ---------- RESTING BP ----------
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)

    # ---------- CHOLESTEROL ----------
    chol = st.slider("Cholesterol (mg/dL)", 100, 400, 230)

    # ---------- FASTING BLOOD SUGAR ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        fbs_x = 1 if fbs == "Yes" else 0
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                High fasting blood sugar may indicate diabetes
                or impaired glucose control.
                """
            )

    # ---------- RESTING ECG ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        restecg = st.selectbox(
            "Resting ECG Result",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T abnormality",
                2: "Left ventricular hypertrophy"
            }[x]
        )
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                ECG measures electrical activity of the heart.
                Abnormal results can signal heart strain or damage.
                """
            )

    # ---------- MAX HEART RATE ----------
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)

    # ---------- EXERCISE ANGINA ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        exang_x = 1 if exang == "Yes" else 0
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                Chest pain during exercise may indicate reduced
                blood flow to the heart.
                """
            )

    # ---------- ST DEPRESSION ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                ST depression measures stress-related changes
                in heart electrical activity.
                """
            )

    # ---------- ST SLOPE ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        slope = st.selectbox(
            "ST Segment Slope",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Downsloping",
                1: "Flat",
                2: "Upsloping"
            }[x]
        )
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                Shape of ST segment during exercise:
                - Downsloping: higher risk
                - Flat: moderate risk
                - Upsloping: lower risk
                """
            )

    # ---------- MAJOR VESSELS ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        ca = st.slider("Number of Major Vessels (ca)", 0, 3, 0)
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                Number of large coronary vessels with blockage.
                More vessels = higher risk.
                """
            )

    # ---------- THAL ----------
    col1, col2 = st.columns([6, 1])
    with col1:
        thal = st.selectbox(
            "Thalassemia",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Fixed defect",
                3: "Reversible defect"
            }[x]
        )
    with col2:
        with st.popover("i"):
            st.markdown(
                """
                Blood flow scan results:
                - Normal: healthy flow
                - Fixed defect: old damage
                - Reversible defect: active blockage
                """
            )

    st.divider()

    if st.button("Predict Medical Risk"):
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

        st.metric(
            label="Estimated Heart Disease Probability",
            value=f"{prob * 100:.1f}%"
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
        text-align: right;
        max-width: 340px;
        line-height: 1.4;
    }
    </style>

    <div class="uci-fine-print">
    Medical risk estimation is based on a machine learning model trained on the 
    <b>UCI Heart Disease dataset</b> (Cleveland Clinic Foundation), originally
    published for academic research and benchmarking.<br>
    This tool is intended for <b>educational and demonstrative purposes only</b>
    and is not a substitute for professional medical evaluation.
    </div>
    """,
    unsafe_allow_html=True
)
