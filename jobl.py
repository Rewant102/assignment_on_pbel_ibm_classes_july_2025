import joblib
import pandas as pd
import numpy as np

try:
    # 1. Load model and preprocessing tools
    model = joblib.load("salary_model_clean.pkl")
    scaler = joblib.load("scaler_clean.pkl")
    label_encoders = joblib.load("label_encoders_clean.pkl")
    print("✅ Model and preprocessors loaded.")

    # 2. Input new employee data — use raw/unmapped input (we’ll map in code)
    new_data = pd.DataFrame([{
        'Age': 30,
        'Country': 'India',
        'Department': 'IT',                # Raw input
        'Position': 'Software Engineer',  # Raw input
        'YearsExperience': 5
    }])
    print("📥 Input:")
    print(new_data)

    # 3. Normalize raw categorical inputs to match model training
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
        
        "Accountant" : "Analyst",  

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

    new_data["Position"] = new_data["Position"].replace(position_mapping)
    new_data["Department"] = new_data["Department"].replace(department_mapping)

    # 4. Encode categorical columns
    for col in label_encoders:
        if col in new_data.columns:
            le = label_encoders[col]
            # Handle unknown values
            unknowns = new_data[~new_data[col].isin(le.classes_)][col].tolist()
            if unknowns:
                raise ValueError(f"❌ Unknown label in column '{col}': {unknowns}")
            new_data[col] = le.transform(new_data[col])

    # 5. Scale numerical values
    new_data_scaled = scaler.transform(new_data)

    # 6. Predict
    predicted_salary = model.predict(new_data_scaled)
    print(f"\n💰 Predicted Salary: ₹{predicted_salary[0]:,.2f}")

except Exception as e:
    print("❌ Error:", e)
