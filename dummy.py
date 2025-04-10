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

# Page configuration
st.set_page_config(page_title="Loan Eligibility Checker", page_icon="🏦", layout="centered")

# App title and description
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>🏦 Loan Eligibility Predictor</h1>
    <p style='text-align: center; font-size: 18px;'>Enter the applicant's details to check if they qualify for a loan.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Layout: Form inside a container
with st.container():
    st.subheader("📋 Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=1000)
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=1000)
        loan_amount = st.number_input("Loan


