import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page title and description (this must be the first Streamlit command)
st.set_page_config(page_title="Credit Loan Eligibility Predictor", layout="centered")

# Title and description
st.title("🏦 Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is **eligible** for a loan 
based on their personal and financial characteristics using a Machine Learning model.
""")

# Load the pre-trained model
with open("random_forest_credit.pickle", "rb") as rf_pickle:
    rf_model = pickle.load(rf_pickle)

# --- FORM INPUTS ---
with st.form("user_inputs"):
    st.subheader("📋 Applicant Details")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Marital Status", ["Yes", "No"])
    Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Monthly Income", min_value=0, step=1000)
    CoapplicantIncome = st.number_input("Coapplicant Monthly Income", min_value=0, step=1000)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0, step=1000)
    Loan_Amount_Term = st.selectbox("Loan Amount Term (in months)", ["360", "180", "240", "120", "60"])
    Credit_History = st.selectbox("Credit History", ["1", "0"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Loan Eligibility")


if submitted:
    # --- FEATURE ENCODING ---
    Gender_Male = 1 if Gender == "Male" else 0
    Married_Yes = 1 if Married == "Yes" else 0
    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0

    if submitted:
    # Properly encode categorical variables
    Dependents_1, Dependents_2, Dependents_3 = 0, 0, 0
    if Dependents == "1":
        Dependents_1 = 1
    elif Dependents == "2":
        Dependents_2 = 1
    elif Dependents == "3+":
        Dependents_3 = 1

    Gender_Male = 1 if Gender == "Male" else 0
    Married_Yes = 1 if Married == "Yes" else 0
    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0

    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0

    # Construct the final input
    input_data = [[
        Dependents_1, Dependents_2, Dependents_3,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        int(Loan_Amount_Term), int(Credit_History),
        Gender_Male, Married_Yes,
        Education_Not_Graduate, Self_Employed_Yes,
        Property_Area_Semiurban, Property_Area_Urban
    ]]

    # You must match this input to the training order
    input_df = pd.DataFrame(input_data, columns=[
        'Dependents_1', 'Dependents_2', 'Dependents_3',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History',
        'Gender_Male', 'Married_Yes',
        'Education_Not_Graduate', 'Self_Employed_Yes',
        'Property_Area_Semiurban', 'Property_Area_Urban'
    ])

    # Predict
    prediction = rf_model.predict(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("✅ You are eligible for a loan!")
    else:
        st.error("❌ Sorry, you are not eligible for a loan.")

    # --- PREDICTION INPUT ---
    input_data = [[
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        int(Loan_Amount_Term),
        int(Credit_History),
        Gender_Male,
        Married_Yes,
        Education_Not_Graduate,
        Self_Employed_Yes,
        Property_Area_Semiurban,
        Property_Area_Urban,
        Dependents_1,
        Dependents_2,
        Dependents_3_plus
    ]]

    # --- PREDICT ---
    prediction = rf_model.predict(input_data)

    st.subheader("🔍 Prediction Result:")
    if prediction[0] == 1:
        st.success("🎉 You are **eligible** for the loan!")
    else:
        st.error("❌ Sorry, you are **not eligible** for the loan.")

    # --- Feature Importance Chart ---
    st.markdown("---")
    st.write("🔍 **Feature Importance**")
    st.image("feature_importance.png")
