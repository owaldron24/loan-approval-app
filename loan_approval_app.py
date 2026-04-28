import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open(r'C:\Users\owald\OneDrive\Desktop\mldeploy\my_model.pkl', 'rb'))
preprocessor = pickle.load(open(r'C:\Users\owald\OneDrive\Desktop\mldeploy\scaler.pkl', 'rb'))

THRESHOLD = 0.68


def main():
    st.title("Loan Approval Prediction")

    reason_label = st.selectbox(
        "Reason for Loan",
        [
            "Cover an unexpected cost",
            "Credit card refinancing",
            "Debt consolidation",
            "Home improvement",
            "Major purchase",
            "Other"
        ]
    )
    reason_map = {
        "Cover an unexpected cost": "cover_an_unexpected_cost",
        "Credit card refinancing": "credit_card_refinancing",
        "Debt consolidation": "debt_conslidation",
        "Home improvement": "home_improvement",
        "Major purchase": "major_purchase",
        "Other": "other"
    }
    reason = reason_map[reason_label]

    requested_loan_amount = st.slider(
        "Requested Loan Amount ($)",
        min_value=5000,
        max_value=125000,
        value=20000,
        step=1000
    )

    fico_score = st.number_input(
        "FICO Score",
        min_value=385,
        max_value=850,
        step=1,
        value=650
    )

    employment_status_label = st.selectbox(
        "Employment Status",
        ["Full Time", "Part Time", "Unemployed"]
    )
    employment_status_map = {
        "Full Time": "full_time",
        "Part Time": "part_time",
        "Unemployed": "unemployed"
    }
    employment_status = employment_status_map[employment_status_label]

    employment_sector_label = st.selectbox(
        "Employment Sector",
        [
            "Unknown",
            "Communication Services",
            "Consumer Discretionary",
            "Consumer Staples",
            "Energy",
            "Financials",
            "Health Care",
            "Industrials",
            "Information Technology",
            "Materials",
            "Real Estate",
            "Utilities"
        ]
    )
    employment_sector_map = {
        "Unknown": "Unknown",
        "Communication Services": "communication_services",
        "Consumer Discretionary": "consumer_discretionary",
        "Consumer Staples": "consumer_staples",
        "Energy": "energy",
        "Financials": "financials",
        "Health Care": "health_care",
        "Industrials": "industrials",
        "Information Technology": "information_technology",
        "Materials": "materials",
        "Real Estate": "real_estate",
        "Utilities": "utilities"
    }
    employment_sector = employment_sector_map[employment_sector_label]

    monthly_gross_income = st.number_input(
        "Monthly Gross Income ($)",
        min_value=-2559,
        max_value=14005,
        step=100,
        value=5000
    )

    monthly_housing_payment = st.number_input(
        "Monthly Housing Payment ($)",
        min_value=300,
        max_value=3300,
        step=50,
        value=1500
    )

    ever_bankrupt_or_foreclose = st.checkbox("Ever declared bankruptcy or foreclosure?")
    ever_bankrupt = int(ever_bankrupt_or_foreclose)

    lender = st.selectbox("Lender", ["A", "B", "C"])

    # Derive Income Level automatically from Monthly Gross Income
    if monthly_gross_income <= 3542:
        income_level = "Low"
    elif monthly_gross_income <= 5008:
        income_level = "Mid-Low"
    elif monthly_gross_income <= 7347:
        income_level = "Mid-High"
    else:
        income_level = "High"

    user_input = {
        "Reason": reason,
        "Requested_Loan_Amount": requested_loan_amount,
        "FICO_score": fico_score,
        "Employment_Status": employment_status,
        "Employment_Sector": employment_sector,
        "Monthly_Gross_Income": monthly_gross_income,
        "Monthly_Housing_Payment": monthly_housing_payment,
        "Ever_Bankrupt_or_Foreclose": ever_bankrupt,
        "Lender": lender,
        "Income_Level": income_level
    }

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        processed_input = preprocessor.transform(input_df)
        probability = model.predict_proba(processed_input)[:, 1][0]
        prediction = int(probability >= THRESHOLD)

        st.write(f"Approval Probability: {probability:.2%}")

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Denied")


if __name__ == "__main__":
    main()