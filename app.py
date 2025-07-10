import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import xgboost  # Ensure xgboost is installed via pip
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("WATSON_API_KEY")

# Load local model and preprocessing tools
@st.cache_resource
def load_local_model():
    model = joblib.load("salary_model_clean.pkl")
    scaler = joblib.load("scaler_clean.pkl")
    label_encoders = joblib.load("label_encoders_clean.pkl")
    return model, scaler, label_encoders

# Local prediction function
def predict_local(data):
    model, scaler, label_encoders = load_local_model()

    # Normalize positions and departments
    position_mapping = {
        "Software Engineer": "Developer",
        "Engineer": "Developer",
        "Software Developer": "Developer",
        "Senior Developer": "Developer",
        "Software Dev": "Developer",
        "HR Executive": "Executive",
        "Sales Executive": "Executive",
        "Support Staff": "Support",
        "Marketing Executive": "Executive",
        "Consulting Engineer": "Consultant",
        "Accountant": "Analyst"
    }

    department_mapping = {
        "Software": "Engineering",
        "Tech": "Engineering",
        "Technical": "Engineering",
        "IT": "Engineering",
        "People": "HR",
        "Customer Service": "Support",
        "Business": "Sales"
    }

    data["Position"] = data["Position"].replace(position_mapping)
    data["Department"] = data["Department"].replace(department_mapping)

    # Encode labels
    for col in label_encoders:
        if col in data.columns:
            le = label_encoders[col]
            if data[col].iloc[0] not in le.classes_:
                return f"❌ Unknown label '{data[col].iloc[0]}' in column '{col}'"
            data[col] = le.transform(data[col])

    # Scale numerical values
    scaled_data = scaler.transform(data)

    # Predict
    pred = model.predict(scaled_data)
    return f"💰 Predicted Salary (Local Model): ₹{pred[0]:,.2f}"

# Watsonx API prediction function
def predict_watsonx(data):
    if not API_KEY:
        return "❌ API key not found. Set it in a .env file with key: WATSON_API_KEY"

    # Step 1: Get IAM token
    token_response = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={
            "apikey": API_KEY,
            "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'
        }
    )

    if token_response.status_code != 200:
        return f"❌ Token fetch failed: {token_response.status_code} - {token_response.text}"

    mltoken = token_response.json()["access_token"]
    header = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {mltoken}'
    }

    # Step 2: Use expected fields and dummy values for required ones
    payload_scoring = {
        "input_data": [
            {
                "fields": [
                    "Employee_ID",
                    "Employee_Name",
                    "Age",
                    "Country",
                    "Department",
                    "Position",
                    "Joining_Date"
                ],
                "values": [[
                    "EMP102",                        # dummy ID
                    "Test User",                     # dummy name
                    int(data["Age"].iloc[0]),
                    data["Country"].iloc[0],
                    data["Department"].iloc[0],
                    data["Position"].iloc[0],
                    "2023-01-01"                     # dummy joining date
                ]]
            }
        ]
    }

    # Step 3: Make request
    response = requests.post(
        url='https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/23558892-2ba7-4318-a1c3-ad980f07df57/predictions?version=2021-05-01',
        json=payload_scoring,
        headers=header
    )

    try:
        result = response.json()
        prediction = result["predictions"][0]["values"][0][0]
        return f"💰 Predicted Salary (Watsonx API): ₹{prediction:,.2f}"
    except Exception as e:
        return f"❌ Error from Watson API: {e}\n\nRaw Response: {response.text}"


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("💼 Salary Prediction App")
st.markdown("Use either your local trained model or IBM Watsonx API to predict employee salary.")

mode = st.radio("Select Prediction Method", ["Local Model", "Watsonx API"])

st.subheader("Enter Employee Details:")
age = st.number_input("Age", 18, 65, value=30)
country = st.selectbox("Country", ["India", "USA", "UK", "Germany", "Canada"])
department = st.selectbox("Department", ["IT", "Software", "Technical", "HR", "Customer Service", "Business"])
position = st.selectbox("Position", [
    "Software Engineer", "Engineer", "Senior Developer", "HR Executive", "Sales Executive",
    "Support Staff", "Marketing Executive", "Consulting Engineer", "Accountant"
])
years_exp = st.slider("Years of Experience", 0, 40, value=5)

if st.button("🔮 Predict"):
    user_data = pd.DataFrame([{
        "Age": age,
        "Country": country,
        "Department": department,
        "Position": position,
        "YearsExperience": years_exp
    }])

    if mode == "Local Model":
        result = predict_local(user_data)
    else:
        result = predict_watsonx(user_data)

    st.success(result)
