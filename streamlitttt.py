import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open("random_forest_credit.pickle", "rb") as f:
    rf_model = pickle.load(f)

# Load the list of features the model was trained on
with open("trained_features.pickle", "rb") as f:
    trained_features = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Loan Eligibility Checker", layout="centered")
st.title("🏦 Loan Eligibility Predictor")

# Input fields
st.subheader("Enter applicant details:")

dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"], index=0)
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Term (in days)", [360.0, 120.0, 180.0, 300.0, 240.0, 60.0, 84.0, 36.0])
credit_history = st.selectbox("Credit History", ["1.0 (Good)", "0.0 (Bad)"])

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Submit button
if st.button("Check Eligibility"):
    
    # Build input dict
    input_dict = {
        "Dependents": float(dependents.replace("+", "")),
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history[0]),  # extract 1 or 0 from string

        # Dummies
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Education_Not_Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }

    # Ensure all features are included (even if 0)
    for col in trained_features:
        if col not in input_dict:
            input_dict[col] = 0

    input_data = pd.DataFrame([input_dict])[trained_features]

    prediction = rf_model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved! The applicant is **eligible**.")
    else:
        st.error("❌ Loan Not Approved. The applicant is **not eligible**.")

