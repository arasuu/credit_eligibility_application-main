import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('random_forest_credit.pickle', 'rb') as f:
    rf_model = pickle.load(f)

st.set_page_config(page_title="Loan Eligibility Predictor")

st.title("🏦 Loan Eligibility Prediction App")
st.write("Fill the form below to check if you're eligible for a loan:")

# Form for user input
with st.form("loan_form"):
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Married = st.selectbox("Married", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Amount Term", [360.0, 180.0, 120.0, 240.0, 300.0])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

    submitted = st.form_submit_button("Check Eligibility")

    if submitted:
        input_dict = {
            'Dependents': [Dependents],
            'ApplicantIncome': [ApplicantIncome],
            'CoapplicantIncome': [CoapplicantIncome],
            'LoanAmount': [LoanAmount],
            'Loan_Amount_Term': [Loan_Amount_Term],
            'Credit_History': [Credit_History],
            'Gender_Male': [1 if Gender == 'Male' else 0],
            'Married_Yes': [1 if Married == 'Yes' else 0],
            'Education_Not_Graduate': [1 if Education == 'Not Graduate' else 0],
            'Self_Employed_Yes': [1 if Self_Employed == 'Yes' else 0],
            'Property_Area_Semiurban': [1 if Property_Area == 'Semiurban' else 0],
            'Property_Area_Urban': [1 if Property_Area == 'Urban' else 0]
        }

        input_data = pd.DataFrame(input_dict)

        # Align features with training data
        trained_features = list(rf_model.feature_names_in_)

        # Ensure all required columns are present
        for col in trained_features:
            if col not in input_data.columns:
                input_data[col] = 0

        # Ensure exact order
        input_data = input_data[trained_features]

        # Make prediction
        prediction = rf_model.predict(input_data)[0]

        # Show result
        if prediction == 1:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Not Approved.")
