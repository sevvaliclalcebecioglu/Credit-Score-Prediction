import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Başlık
st.title("💳 Credit Score Prediction App")

# Model ve kolonları yükle
model = joblib.load("../data/random_forest_model.pkl")
columns = joblib.load("../data/columns.pkl")  # eğitim sırasında kaydettiğimiz feature isimleri

st.header("Enter your financial details:")

# Kullanıcı inputları
Annual_Income = st.number_input("Annual Income:", min_value=0.0, value=20000.0)
Monthly_Inhand_Salary = st.number_input("Monthly Inhand Salary:", min_value=0.0, value=2000.0)
Num_Bank_Accounts = st.number_input("Number of Bank Accounts:", min_value=0, value=2)
Num_Credit_Card = st.number_input("Number of Credit Cards:", min_value=0, value=2)
Interest_Rate = st.number_input("Interest Rate (%):", min_value=0.0, value=3.0)
Num_of_Loan = st.number_input("Number of Loans:", min_value=0, value=1)
Delay_from_due_date = st.number_input("Average number of days delayed:", min_value=0, value=0)
Num_of_Delayed_Payment = st.number_input("Number of delayed payments:", min_value=0, value=0)
Credit_Mix = st.selectbox("Credit Mix:", options=["Bad", "Standard", "Good"])
Outstanding_Debt = st.number_input("Outstanding Debt:", min_value=0.0, value=5000.0)
Credit_History_Age = st.number_input("Credit History Age (in months):", min_value=0, value=24)
Monthly_Balance = st.number_input("Monthly Balance:", min_value=0.0, value=1000.0)

# Inputu DataFrame olarak hazırla
user_input = {
    "Annual_Income": Annual_Income,
    "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
    "Num_Bank_Accounts": Num_Bank_Accounts,
    "Num_Credit_Card": Num_Credit_Card,
    "Interest_Rate": Interest_Rate,
    "Num_of_Loan": Num_of_Loan,
    "Delay_from_due_date": Delay_from_due_date,
    "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
    "Credit_Mix_Bad": 0,
    "Credit_Mix_Standard": 0,
    "Credit_Mix_Good": 0,
    "Outstanding_Debt": Outstanding_Debt,
    "Credit_History_Age": Credit_History_Age,
    "Monthly_Balance": Monthly_Balance
}

# Kategorik Credit_Mix değerini one-hot yap
if Credit_Mix == "Bad":
    user_input["Credit_Mix_Bad"] = 1
elif Credit_Mix == "Standard":
    user_input["Credit_Mix_Standard"] = 1
elif Credit_Mix == "Good":
    user_input["Credit_Mix_Good"] = 1

input_df = pd.DataFrame([user_input])

# Eksik kolonları ekle
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Kolon sırasını model ile eşle
input_df = input_df[columns]

# Tahmin butonu
if st.button("Predict Credit Score"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Credit Score: {prediction[0]}")
