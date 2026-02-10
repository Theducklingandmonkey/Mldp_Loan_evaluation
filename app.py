import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load model and features
# -------------------------
model = joblib.load("Best_gradient_boosting_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Loan Interest Rate Predictor")

st.write("Enter your financial details to estimate your loan interest rate.")

# -------------------------
# User inputs (raw)
# -------------------------
annual_inc = st.number_input("Annual Income", min_value=0.0, value=50000.0)
loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, value=15.0)
revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=2000.0)
revol_util = st.number_input("Revolving Utilisation (%)", min_value=0.0, max_value=100.0, value=30.0)
credit_history_years = st.number_input("Credit History Length (years)", min_value=0.0, value=5.0)
emp_length = st.number_input("Employment Length (years)", min_value=0, value=3)
term = st.selectbox("Loan Term (months)", [36, 60])
issue_year = st.number_input("Issue Year", min_value=2000, value=2024)
issue_month = st.number_input("Issue Month", min_value=1, max_value=12, value=6)
has_ever_delinquent = st.selectbox("Ever Delinquent?", [0, 1])

# -------------------------
# Build input dataframe
# -------------------------
input_df = pd.DataFrame([{
    "annual_inc": annual_inc,
    "loan_amnt": loan_amnt,
    "dti": dti,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "credit_history_years": credit_history_years,
    "emp_length": emp_length,
    "term": term,
    "issue_year": issue_year,
    "issue_month": issue_month,
    "has_ever_delinquent": has_ever_delinquent
}])

# -------------------------
# Feature engineering (MUST match training)
# -------------------------
input_df["log_annual_inc"] = np.log1p(input_df["annual_inc"])
input_df["log_revol_bal"] = np.log1p(input_df["revol_bal"])
input_df["dti_sq"] = input_df["dti"] ** 2
input_df["revol_util_sq"] = input_df["revol_util"] ** 2
input_df["dti_x_revol_util"] = input_df["dti"] * input_df["revol_util"]
input_df["income_x_credit_history"] = (
    input_df["annual_inc"] * input_df["credit_history_years"]
)
input_df["issue_year_sq"] = input_df["issue_year"] ** 2

# -------------------------
# One-hot encoding
# -------------------------
input_encoded = pd.get_dummies(input_df)

# Align columns to training data
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Interest Rate"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Interest Rate: {prediction:.2f}%")
