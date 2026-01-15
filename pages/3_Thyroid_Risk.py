import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go


st.set_page_config(page_title="Thyroid Risk", layout="centered")

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


st.title("Thyroid Cancer Risk Assessment")
st.caption("Educational screening tool.not a medical diagnosis")

st.divider()

#loading model 
@st.cache_resource
def load_thyroid_model():
    return joblib.load("models/thyroid_questionnaire_xgb.pkl")

model = load_thyroid_model()


#questionnare mode 

st.header("Basic Risk Screening (No Lab Tests)")

with st.form("thyroid_questionnaire"):
    age = st.slider("Age", 18, 90, 45)

    gender = st.radio("Gender", ["Male", "Female"])
    gender_x = 1 if gender == "Male" else 0
    radiation = st.selectbox(
        "Past radiation exposure to head or neck?",
        ["No", "Yes"],
        help=(
            "This includes radiation therapy to the head or neck, especially in childhood. "
            "If you are unsure, select 'Yes' for a conservative estimate."
        )
    )
    radiation_x = 1 if radiation == "Yes" else 0

    family_history = st.selectbox(
        "Family history of thyroid disease?", ["No", "Yes"]
    )

    iodine = st.selectbox(
        "Iodine deficiency?",
        ["No", "Yes"],
        help=(
            "Iodine deficiency may occur if you do not consume iodized salt "
            "or have limited seafood/dairy intake. "
            "usually no if you consume iodised salt"
            "If you are unsure, select 'Yes' for a conservative (worst-case) estimate."
        )
    )
    iodine_x = 1 if iodine == "Yes" else 0

    smoking = st.selectbox("Smoking", ["No", "Yes"])
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

    # BMI calculation
    height_m = (height_feet * 0.3048) + (height_inches * 0.0254)
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        bmi_label = "Underweight"
        bmi_color = "#e74c3c"
    elif bmi < 25:
        bmi_label = "Normal"
        bmi_color = "#2ecc71"
    elif bmi < 30:
        bmi_label = "Overweight"
        bmi_color = "#f1c40f"
    else:
        bmi_label = "Obese"
        bmi_color = "#c0392b"

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
            Calculated BMI: {bmi}<br>
            Category: {bmi_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    obesity_x = 1 if bmi >= 30 else 0

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])

    nodule_size = st.number_input(
        "Thyroid nodule size (cm)",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        help=(
            "A thyroid nodule is a lump in the thyroid gland, usually detected by "
            "ultrasound or physical examination. "
            "If you do not know the size, leave this as 0."
        )
    )


    ethnicity = st.selectbox(
        "Ethnicity",
        ["African", "Asian", "Caucasian", "Hispanic", "Middle Eastern"]
    )

    submit = st.form_submit_button("Assess Risk")

if submit:
    ethnicity_map = {
        "Asian": [1, 0, 0, 0],
        "Caucasian": [0, 1, 0, 0],
        "Hispanic": [0, 0, 1, 0],
        "Middle Eastern": [0, 0, 0, 1],
        "African": [0, 0, 0, 0]  # baseline
    }

    eth_vals = ethnicity_map[ethnicity]

    x = np.array([[
        age,
        gender_x,
        1 if family_history == "Yes" else 0,
        radiation_x,
        iodine_x,
        1 if smoking == "Yes" else 0,
        obesity_x,
        1 if diabetes == "Yes" else 0,
        nodule_size,
        *eth_vals
    ]])


    risk = model.predict_proba(x)[0][1] * 100

    st.subheader("📊 Risk Result")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%"},
            title={"text": "Estimated Malignancy Risk"},
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


    if risk >= 58:
        st.error(
            "⚠️ High risk detected.\n\n"
            "We recommend proceeding to clinical assessment "
            "with basic thyroid lab tests (TSH, T3, T4)."
        )
        st.info(
            "Clinical assessment significantly improves prediction accuracy."
        )

    elif risk >= 35:
        st.warning(
            "Moderate risk detected.\n\n"
            "Regular monitoring and lifestyle management are advised.\n\n"
            "We recommend clinical test to be sure."
        )
    else:
        st.success(
            "Low risk detected.\n\n"
            "Maintain healthy lifestyle and routine check-ups."
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
        max-width: 360px;
        text-align: right;
        line-height: 1.4;
    }
    </style>

    <div class="fine-print">
    <b>Model trained on a publicly available thyroid cancer risk dataset</b><br>
    derived from longitudinal clinical records of patients with<br>
    <b>well-differentiated thyroid cancer</b>, followed for up to <b>15 years</b>.<br>
    Dataset released under the <b>MIT License</b> via the Kaggle platform.<br>
    Educational use only · not a medical diagnosis or clinical advice.
    </div>
    """,
    unsafe_allow_html=True
)
