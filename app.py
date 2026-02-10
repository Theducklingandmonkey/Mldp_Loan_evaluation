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
st.write(
    "Estimate the interest rate you may receive before applying for a loan."
)

# =========================
# FINANCIAL DETAILS
# =========================
st.subheader("Financial Information")

col1, col2 = st.columns(2)

with col1:
    annual_inc = st.number_input(
        "Annual Income", min_value=0.0, value=50000.0
    )
    revol_bal = st.number_input(
        "Revolving Credit Balance", min_value=0.0, value=2000.0
    )
    credit_history_years = st.number_input(
        "Credit History Length (years)", min_value=0.0, value=5.0
    )

with col2:
    loan_amnt = st.number_input(
        "Loan Amount", min_value=0.0, value=10000.0
    )
    revol_util = st.number_input(
        "Average Revolving Credit Utilisation (%)",
        min_value=0.0, max_value=110.0, value=30.0
    )
    emp_length = st.number_input(
        "Employment Length (years)", min_value=0, value=3
    )

# =========================
# LOAN DETAILS
# =========================
st.subheader("Loan Details")

col3, col4 = st.columns(2)

with col3:
    term = st.selectbox("Loan Term (months)", [36, 60])
    issue_year = st.number_input(
        "Issue Year", min_value=1930, value=2024
    )

with col4:
    issue_month = st.selectbox(
        "Issue Month", list(range(1, 13))
    )
    monthly_debt = st.number_input(
        "Current Monthly Debt",
        min_value=0.0,
        value=500.0,
        help="Total monthly debt payments (e.g. loans, credit cards)"
    )

# -------------------------
# Calculate DTI automatically
# -------------------------
monthly_income = annual_inc / 12 if annual_inc > 0 else 0.0

if monthly_income > 0:
    dti = (monthly_debt / monthly_income) * 100
else:
    dti = 0.0

st.caption(f"Calculated Debt-to-Income Ratio (DTI): {dti:.2f}%")

# =========================
# CREDIT HISTORY
# =========================
st.subheader("Credit History")

col5, col6 = st.columns(2)

with col5:
    has_ever_delinquent = st.selectbox(
        "Ever Delinquent?",
        ["No", "Yes"]
    )

with col6:
    if has_ever_delinquent == "Yes":
        mths_since_last_delinq = st.number_input(
            "Months Since Last Delinquency",
            min_value=0,
            max_value=36,
            value=12,
            step=1
        )
    else:
        mths_since_last_delinq = 999

# =========================
# CATEGORICAL DETAILS
# =========================
st.subheader("Verification & Housing")

col7, col8 = st.columns(2)

with col7:
    verification_status = st.selectbox(
        "Verification Status",
        ["Not Verified", "Verified", "Source Verified"]
    )

with col8:
    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "OTHER"]
    )

# =========================
# BUILD BASE INPUT
# =========================
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
    "has_ever_delinquent": int(has_ever_delinquent == "Yes"),
    "mths_since_last_delinq": mths_since_last_delinq
}])

# =========================
# FEATURE ENGINEERING
# =========================
input_df["log_annual_inc"] = np.log1p(input_df["annual_inc"])
input_df["log_revol_bal"] = np.log1p(input_df["revol_bal"])
input_df["dti_sq"] = input_df["dti"] ** 2
input_df["revol_util_sq"] = input_df["revol_util"] ** 2
input_df["dti_x_revol_util"] = (
    input_df["dti"] * input_df["revol_util"]
)
input_df["income_x_credit_history"] = (
    input_df["annual_inc"] * input_df["credit_history_years"]
)
input_df["issue_year_sq"] = input_df["issue_year"] ** 2

# =========================
# MANUAL ONE-HOT ENCODING
# =========================
input_df["verification_status_Verified"] = int(
    verification_status == "Verified"
)
input_df["verification_status_Source Verified"] = int(
    verification_status == "Source Verified"
)

input_df["home_ownership_RENT"] = int(home_ownership == "RENT")
input_df["home_ownership_OWN"] = int(home_ownership == "OWN")
input_df["home_ownership_OTHER"] = int(home_ownership == "OTHER")

# =========================
# ALIGN WITH TRAINING DATA
# =========================
input_encoded = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)

# Safety: drop target if present
input_encoded = input_encoded.drop(
    columns=["int_rate"],
    errors="ignore"
)

# =========================
# PREDICTION (INLINE)
# =========================
col_btn, col_result = st.columns([1, 2])

with col_btn:
    predict_clicked = st.button("Predict Interest Rate")

with col_result:
    if predict_clicked:
        prediction = model.predict(input_encoded)[0]
        st.markdown(
            f"""
            <div style="
                padding: 0.5rem 0.8rem;
                border-radius: 0.4rem;
                background: rgba(0, 255, 0, 0.08);
                color: #32cd32;
                border: 1px solid rgba(0, 255, 0, 0.25);
                font-size: 1.1rem;
                font-weight: 600;
            ">
                Estimated Interest Rate: {prediction:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

