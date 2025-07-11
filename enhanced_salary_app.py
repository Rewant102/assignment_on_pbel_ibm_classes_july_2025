import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import altair as alt
import xgboost
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("WATSON_API_KEY")

@st.cache_resource
def load_local_model():
    model = joblib.load("salary_model_clean.pkl")
    scaler = joblib.load("scaler_clean.pkl")
    label_encoders = joblib.load("label_encoders_clean.pkl")
    return model, scaler, label_encoders

def predict_local(data):
    model, scaler, label_encoders = load_local_model()

    position_mapping = {
        "Software Engineer": "Developer", "Engineer": "Developer",
        "Software Developer": "Developer", "Senior Developer": "Developer",
        "Software Dev": "Developer", "HR Executive": "Executive",
        "Sales Executive": "Executive", "Support Staff": "Support",
        "Marketing Executive": "Executive", "Consulting Engineer": "Consultant",
        "Accountant": "Analyst"
    }
    department_mapping = {
        "Software": "Engineering", "Tech": "Engineering", "Technical": "Engineering",
        "IT": "Engineering", "People": "HR", "Customer Service": "Support", "Business": "Sales"
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
    pred = model.predict(scaled_data)
    return pred[0], None

def predict_watsonx(data):
    if not API_KEY:
        return None, "❌ API key not found. Set it in .env with WATSON_API_KEY"

    token_response = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
    )
    if token_response.status_code != 200:
        return None, f"❌ Token fetch failed: {token_response.status_code} - {token_response.text}"

    mltoken = token_response.json()["access_token"]
    header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {mltoken}'}

    payload_scoring = {
        "input_data": [{
            "fields": ["Employee_ID", "Employee_Name", "Age", "Country", "Department", "Position", "Joining_Date"],
            "values": [[
                "EMP102", "Test User",
                int(data["Age"].iloc[0]), data["Country"].iloc[0],
                data["Department"].iloc[0], data["Position"].iloc[0], "2023-01-01"
            ]]
        }]
    }

    response = requests.post(
        url='https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/23558892-2ba7-4318-a1c3-ad980f07df57/predictions?version=2021-05-01',
        json=payload_scoring,
        headers=header
    )

    try:
        result = response.json()
        prediction = result["predictions"][0]["values"][0][0]
        return prediction, None
    except Exception as e:
        return None, f"❌ Error from Watson API: {e}\n\nRaw Response: {response.text}"


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Salary Prediction App")
st.markdown("Use your **Local ML Model** or **IBM Watsonx Cloud Model** to predict employee salary.")

with st.sidebar:
    st.header("🔧 Input Employee Details")
    age = st.number_input("🎂 Age", 18, 65, value=30)
    country = st.selectbox("🌍 Country", ["India", "USA", "UK", "Germany", "Canada"])
    department = st.selectbox("🏢 Department", ["IT", "Software", "Technical", "HR", "Customer Service", "Business"])
    position = st.selectbox("👤 Position", [
        "Software Engineer", "Engineer", "Senior Developer", "HR Executive", "Sales Executive",
        "Support Staff", "Marketing Executive", "Consulting Engineer", "Accountant"
    ])
    years_exp = st.slider("⌛ Years of Experience", 0, 40, value=5)
    mode = st.radio("⚙️ Prediction Method", ["Local Model", "Watsonx API"])
    predict_btn = st.button("🔮 Predict Salary")

# Dummy min/max salary ranges
salary_ranges = {
    "Software Engineer": (50000, 120000),
    "Engineer": (45000, 100000),
    "Senior Developer": (80000, 1800000),
    "HR Executive": (40000, 700000),
    "Sales Executive": (35000, 800000),
    "Support Staff": (25000, 500000),
    "Marketing Executive": (40000, 850000),
    "Consulting Engineer": (70000, 1300000),
    "Accountant": (50000, 950000)
}

if predict_btn:
    user_data = pd.DataFrame([{
        "Age": age,
        "Country": country,
        "Department": department,
        "Position": position,
        "YearsExperience": years_exp
    }])

    if mode == "Local Model":
        prediction, error = predict_local(user_data)
    else:
        prediction, error = predict_watsonx(user_data)

    if error:
        st.error(error)
    else:
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

        # Salary trend line chart
        min_sal, max_sal = salary_ranges.get(position, (500000, 1000000))
        chart_data = pd.DataFrame({
            "Label": ["Minimum", "Predicted", "Maximum"],
            "Salary": [min_sal, prediction, max_sal]
        })

        st.markdown("### 📈 Where You Stand in Salary Range")
        chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("Label", sort=["Minimum", "Predicted", "Maximum"]),
            y="Salary",
            color=alt.value("steelblue")
        ).properties(width=500, height=300)

        st.altair_chart(chart, use_container_width=True)
