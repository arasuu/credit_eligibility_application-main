import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open("random_forest_credit.pickle", "rb") as f:
    rf_model = pickle.load(f)

# Load trained feature list
with open("trained_features.pickle", "rb") as f:
    trained_features = pickle.load(f)

# Set page config
st.set_page_config(page_title="Loan Eligibility Checker", layout="centered")

# App header
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        h1 { color: #1f77b4; text-align: center; }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Eligibility Predictor")

# Form UI
with st.form("loan_form"):
    st.subheader("📋 Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
        applicant_income = st.number_input("Applicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_term = st.selectbox("Loan Term (days)", [360.0, 120.0, 180.0, 300.0, 240.0, 60.0, 84.0, 36.0])
        credit_history = st.selectbox("Credit History", ["1.0 (Good)", "0.0 (Bad)"])
        married = st.selectbox("Married", ["Yes", "No"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Check Eligibility")

# Prediction
if submitted:
    input_dict = {
        "Dependents": float(dependents.replace("+", "")),
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history[0]),
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Education_Not_Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0
    }

    # Fill any missing features with 0
    for col in trained_features:
        if col not in input_dict:
            input_dict[col] = 0

    input_data = pd.DataFrame([input_dict])[trained_features]

    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]  # probability of class 1

    if prediction == 1:
        st.success(f"✅ **Loan Approved!**\n\nProbability of Approval: `{probability:.2%}`")
    else:
        st.error(f"❌ **Loan Not Approved.**\n\nProbability of Approval: `{probability:.2%}`")
