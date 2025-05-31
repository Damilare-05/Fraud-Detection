
import streamlit as st
import numpy as np
import joblib
import math

# Load model and scaler
model = joblib.load('logistic_fraud_model1.pkl')
scaler = joblib.load('scaler (2).pkl')


st.title("💳📱💸 Mobile Money Transfer Fraud Detection")

from PIL import Image

image = Image.open("fraudimage.jpg")
st.image(image, use_container_width=True)


# Define transaction types
transaction_types = ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT"]

st.sidebar.header("Input Transaction Details")

# User Inputs
amount = st.sidebar.number_input("💰 Transaction Amount", min_value=0.0, value=10000.0)
txn_type = st.sidebar.selectbox("🔁 Transaction Type", options=transaction_types)
step = st.sidebar.number_input("⏱ Time Step (hour)", min_value=0, max_value=744, value=100)
oldbalanceOrg = st.sidebar.number_input("💼 Origin Account Balance Before", min_value=0.0, value=15000.0)
newbalanceOrig = st.sidebar.number_input("💼 Origin Account Balance After", min_value=0.0, value=5000.0)
oldbalanceDest = st.sidebar.number_input("🏦 Destination Balance Before", min_value=0.0, value=0.0)
newbalanceDest = st.sidebar.number_input("🏦 Destination Balance After", min_value=0.0, value=0.0)

if st.button("Predict Fraud"):

    # Derived features
    log_amount = math.log1p(amount)
    isHighRiskType = 1 if txn_type in ["TRANSFER", "CASH_OUT"] else 0
    largeTransaction = 1 if amount > 200000 else 0
    errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest

    # One-hot encoded transaction type
    type_TRANSFER = 1 if txn_type == "TRANSFER" else 0
    type_PAYMENT = 1 if txn_type == "PAYMENT" else 0
    type_CASH_OUT = 1 if txn_type == "CASH_OUT" else 0
    type_DEBIT = 1 if txn_type == "DEBIT" else 0

    # Final feature array (must match training feature order)
    features = np.array([[ 
        isHighRiskType,
        log_amount,
        type_TRANSFER,
        type_PAYMENT,
        largeTransaction,
        step,
        errorBalanceDest,
        errorBalanceOrig,
        type_CASH_OUT,
        newbalanceOrig,
        oldbalanceOrg,
        oldbalanceDest,
        type_DEBIT
    ]])

    # Scale and predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0][1]

    st.markdown(f"### 🔍 Prediction: **{'FRAUD' if prediction else 'NOT FRAUD'}**")
    st.markdown(f"📈 Fraud Probability: `{proba:.2%}`")
