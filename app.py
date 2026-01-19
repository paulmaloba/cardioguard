import joblib
import streamlit as st
import pandas as pd
import numpy as np

# load model and scaler
model = joblib.load("heart_disease_prediction_model (2).pk1")
scaler = joblib.load("scaler_8_feats.pk1")

st.set_page_config(
    page_title="CardioGuard",
    page_icon="❤️",
    layout="centered"
)



st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="
            text-align:center;
            color:#d62828;
            font-size:clamp(24px, 5vw, 36px);
        ">
            ❤️ CardioGuard 🩺
        </h1>
        <p style="font-size:16px; text-align:center; color:#555555;">
                AI-powered heart disease risk assessment
            </p>
        <p style="
            text-align:center;
            font-size:clamp(14px, 3vw, 18px);
        ">
            Enter patient details below to estimate heart disease risk
        </p>
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)



col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)

with col2:
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    sex = st.radio("Sex", ["Female", "Male"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)
    chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "Other"]
    )

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)

# Encode binary inputs
fasting_bs_val = 1 if fasting_bs == "Yes" else 0
sex_m = 1 if sex == "Male" else 0

# Encode chest pain
cp_ata = 1 if chest_pain == "ATA" else 0
cp_nap = 1 if chest_pain == "NAP" else 0

# Build input DataFrame (order matters!)
input_df = pd.DataFrame([{
    'Age': age,
    'FastingBS': fasting_bs_val,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'Oldpeak': oldpeak,
    'Sex_M': sex_m,
    'ChestPainType_ATA': cp_ata,
    'ChestPainType_NAP': cp_nap
}])


# scale and predict
input_scaled = scaler.transform(input_df)
############################## Prediction ####################
if predict_btn:
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.markdown(
            """
            <div style="
                background-color:#ffe5e5;
                padding:20px;
                border-radius:10px;
                border-left:6px solid #d62828;
            ">
                <h3 style="color:#d62828;">⚠️ High Risk Detected</h3>
                <p style="color:#52241B;">This patient shows indicators associated with heart disease.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background-color:#e6f4ea;
                padding:20px;
                border-radius:10px;
                border-left:6px solid #2a9d8f;
            ">
                <h3 style="color:#2a9d8f;">✅ Low Risk Detected</h3>
                <p style="color:#1B5240;">No strong indicators of heart disease were detected.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


##############################     ######################


