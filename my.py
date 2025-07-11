import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# === 1. Load Raw Data ===
df = pd.read_csv("employee_records.csv")

# === 2. Drop irrelevant columns ===
df.drop(columns=["Employee_ID", "Employee_Name"], inplace=True)

# === 3. Extract YearsExperience ===
df["Joining_Date"] = pd.to_datetime(df["Joining_Date"], errors="coerce")
df["YearsExperience"] = datetime.now().year - df["Joining_Date"].dt.year
df.drop(columns=["Joining_Date"], inplace=True)

# === 4. Normalize categorical values ===
# Position mapping
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
    "Consulting Engineer": "Consultant"
}

# Department mapping
department_mapping = {
    "Software": "Engineering",
    "Tech": "Engineering",
    "Technical": "Engineering",
    "IT": "Engineering",
    "People": "HR",
    "Customer Service": "Support",
    "Business": "Sales"
}

df["Position"] = df["Position"].replace(position_mapping)
df["Department"] = df["Department"].replace(department_mapping)

# === 5. Drop missing values ===
df.dropna(inplace=True)

# === 6. Separate features and target ===
target_column = "Salary"
X = df.drop(columns=[target_column])
y = df[target_column]

# === 7. Encode categoricals ===
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# === 8. Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 9. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 10. Train XGBoost ===
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# === 11. Evaluate ===
y_pred = model.predict(X_test)
print(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"✅ RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# === 12. Save model and tools ===
joblib.dump(model, "salary_model_clean.pkl")
joblib.dump(scaler, "scaler_clean.pkl")
joblib.dump(label_encoders, "label_encoders_clean.pkl")
print("✅ Model and preprocessors saved successfully.")
