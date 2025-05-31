
import streamlit as st
import numpy as np
import joblib
import math

# Load model and scaler
model = joblib.load('logistic_fraud_model.pkl')
scaler = joblib.load('scaler (1).pkl')

# Map types
type_mapping = {"CASH_OUT": 0, "TRANSFER": 1}

st.title("💳 Mobile Money Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraudulent.")

# User Inputs
amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=10000.0)
txn_type = st.selectbox("🔁 Transaction Type", options=list(type_mapping.keys()))
step = st.number_input("⏱ Time Step (hour)", min_value=0, max_value=744, value=100)
oldbalanceOrg = st.number_input("💼 Origin Account Balance Before", min_value=0.0, value=15000.0)
newbalanceOrig = st.number_input("💼 Origin Account Balance After", min_value=0.0, value=5000.0)
oldbalanceDest = st.number_input("🏦 Destination Balance Before", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("🏦 Destination Balance After", min_value=0.0, value=0.0)
flagged = st.selectbox("🛑 Was it Flagged by Rules?", options=[0, 1])

if st.button("Predict Fraud"):
    # Derived features
    log_amount = math.log1p(amount)
    type_encoded = type_mapping[txn_type]
    isHighRiskType = 1 if txn_type in ["TRANSFER", "CASH_OUT"] else 0
    largeTransaction = 1 if amount > 200000 else 0
    errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest

    # Final feature order (must match training)
    features = np.array([[
        isHighRiskType,
        log_amount,
        type_encoded,
        largeTransaction,
        step,
        errorBalanceDest,
        errorBalanceOrig,
        newbalanceOrig,
        oldbalanceOrg,
        oldbalanceDest,
        flagged
    ]])

    # Scale and predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0][1]

    st.markdown(f"### 🔍 Prediction: **{'FRAUD' if prediction else 'NOT FRAUD'}**")
    st.markdown(f"📈 Fraud Probability: `{proba:.2%}`")
