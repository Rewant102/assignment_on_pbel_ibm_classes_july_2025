import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
API_KEY = os.getenv("WATSON_API_KEY")

@st.cache_resource
def load_local_model():
    model = joblib.load("salary_model_clean.pkl")
    scaler = joblib.load("scaler_clean.pkl")
    label_encoders = joblib.load("label_encoders_clean.pkl")
    return model, scaler, label_encoders

# Local model prediction
def predict_local(data):
    model, scaler, label_encoders = load_local_model()

    # Mapping for normalization
    position_mapping = {
        "Software Engineer": "Developer", "Engineer": "Developer",
        "Software Developer": "Developer", "Senior Developer": "Developer",
        "Software Dev": "Developer", "HR Executive": "Executive",
        "Sales Executive": "Executive", "Support Staff": "Support",
        "Marketing Executive": "Executive", "Consulting Engineer": "Consultant",
        "Accountant": "Analyst"
    }
    department_mapping = {
        "Software": "Engineering", "Tech": "Engineering",
        "Technical": "Engineering", "IT": "Engineering",
        "People": "HR", "Customer Service": "Support", "Business": "Sales"
    }

    data["Position"] = data["Position"].replace(position_mapping)
    data["Department"] = data["Department"].replace(department_mapping)

    for col in label_encoders:
        if col in data.columns:
            le = label_encoders[col]
            if data[col].iloc[0] not in le.classes_:
                return None, f"❌ Unknown label '{data[col].iloc[0]}' in column '{col}'"
            data[col] = le.transform(data[col])

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return prediction[0], None

# Watsonx prediction
def predict_watsonx(data):
    if not API_KEY:
        return None, "❌ API key not found in .env (WATSON_API_KEY)"

    token_response = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
    )

    if token_response.status_code != 200:
        return None, f"❌ Token fetch failed: {token_response.status_code} - {token_response.text}"

    mltoken = token_response.json()["access_token"]
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {mltoken}'}

    payload = {
        "input_data": [{
            "fields": ["Employee_ID", "Employee_Name", "Age", "Country", "Department", "Position", "Joining_Date"],
            "values": [[
                data["Employee_ID"].iloc[0],
                data["Employee_Name"].iloc[0],
                int(data["Age"].iloc[0]),
                data["Country"].iloc[0],
                data["Department"].iloc[0],
                data["Position"].iloc[0],
                data["Joining_Date"].iloc[0]
            ]]
        }]
    }

    response = requests.post(
        url='https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/23558892-2ba7-4318-a1c3-ad980f07df57/predictions?version=2021-05-01',
        json=payload,
        headers=headers
    )

    try:
        result = response.json()
        prediction = result["predictions"][0]["values"][0][0]
        return prediction, None
    except Exception as e:
        return None, f"❌ Watson API Error: {e}\n\nRaw: {response.text}"

# -------------------------------
# 🌟 Streamlit UI
# -------------------------------
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Salary Prediction App")

mode = st.radio("Select Prediction Method", ["Local Model", "Watsonx API"])

with st.form("salary_form"):
    st.header("📋 Input Details")

    if mode == "Local Model":
        # Inputs for Local Model
        age = st.number_input("🎂 Age", 18, 65, value=30)
        country = st.selectbox("🌍 Country", ["India", "USA", "UK", "Germany", "Canada"])
        department = st.selectbox("🏢 Department", ["IT", "Software", "Technical", "HR", "Customer Service", "Business"])
        position = st.selectbox("💼 Position", [
            "Software Engineer", "Engineer", "Senior Developer", "HR Executive", "Sales Executive",
            "Support Staff", "Marketing Executive", "Consulting Engineer", "Accountant"
        ])
        years_exp = st.slider("⌛ Years of Experience", 0, 40, value=5)

    else:
        # Inputs for Watsonx API
        emp_id = st.text_input("🆔 Employee ID", value="EMP102")
        emp_name = st.text_input("👤 Employee Name", value="Test User")
        age = st.number_input("🎂 Age", 18, 65, value=30)
        country = st.selectbox("🌍 Country", ["India", "USA", "UK", "Germany", "Canada"])
        department = st.selectbox("🏢 Department", ["IT", "Software", "Technical", "HR", "Customer Service", "Business"])
        position = st.selectbox("💼 Position", [
            "Software Engineer", "Engineer", "Senior Developer", "HR Executive", "Sales Executive",
            "Support Staff", "Marketing Executive", "Consulting Engineer", "Accountant"
        ])
        joining_date = st.date_input("📅 Joining Date")

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    if mode == "Local Model":
        input_df = pd.DataFrame([{
            "Age": age,
            "Country": country,
            "Department": department,
            "Position": position,
            "YearsExperience": years_exp
        }])
        prediction, error = predict_local(input_df)
    else:
        input_df = pd.DataFrame([{
            "Employee_ID": emp_id,
            "Employee_Name": emp_name,
            "Age": age,
            "Country": country,
            "Department": department,
            "Position": position,
            "Joining_Date": str(joining_date)
        }])
        prediction, error = predict_watsonx(input_df)

    if error:
        st.error(error)
    else:
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
