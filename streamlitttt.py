import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("random_forest_credit.pickle", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Set Streamlit page configuration
st.set_page_config(page_title="Credit Eligibility Prediction", layout="centered")

# App Title
st.title("🏦 Credit Eligibility Prediction App")

# Form for user input
with st.form("credit_form"):
    st.subheader("Enter Applicant Details")

    Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Amount Term (months)", [360, 120, 180, 300, 240, 60, 84, 12])
    Credit_History = st.selectbox("Credit History", [1, 0])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("Check Eligibility")

    if submitted:
        # One-hot encode features
        Dependents_1 = 1 if Dependents == "1" else 0
        Dependents_2 = 1 if Dependents == "2" else 0
        Dependents_3 = 1 if Dependents == "3+" else 0

        Gender_Male = 1 if Gender == "Male" else 0
        Married_Yes = 1 if Married == "Yes" else 0
        Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
        Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0

        # Construct input DataFrame
        input_data = pd.DataFrame([[
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History,
            Gender_Male, Married_Yes, Education_Not_Graduate, Self_Employed_Yes,
            Property_Area_Semiurban, Property_Area_Urban,
            Dependents_1, Dependents_2, Dependents_3
        ]], columns=[
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History',
            'Gender_Male', 'Married_Yes', 'Education_Not_Graduate', 'Self_Employed_Yes',
            'Property_Area_Semiurban', 'Property_Area_Urban',
            'Dependents_1', 'Dependents_2', 'Dependents_3'
        ])

        trained_features = rf_model.feature_names_in_
        input_data = input_data[trained_features]

        # Reorder to match training features if necessary
        trained_features = rf_model.feature_names_in_
        input_data = input_data[trained_features]

        # Make prediction
        prediction = rf_model.predict(input_data)

        # Show result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("✅ You are eligible for a loan!")
        else:
            st.error("❌ Sorry, you are not eligible for a loan.")
