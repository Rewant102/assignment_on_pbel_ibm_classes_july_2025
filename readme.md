# 💼 Salary Prediction App

An intelligent Streamlit app that predicts employee salaries using either:
- ✅ A locally trained Machine Learning model (XGBoost)
- 🌐 IBM Watsonx deployed model via API

This project showcases real-world MLOps with hybrid prediction options, built for smart HR analytics and salary forecasting.

---
### Project Structure
📁 my-smart-predictor/
│
├── app.py # Streamlit app
├── requirements.txt # Python dependencies
├── runtime.txt # Optional: Python version pinning
├── .gitignore # Clean Git tracking
├── .env # Contains API key (excluded from GitHub)
│
├── salary_model_clean.pkl # Local XGBoost model
├── scaler_clean.pkl # Feature scaler
├── label_encoders_clean.pkl # Encoders for department/position
│
└── .streamlit/
└── config.toml # Cloud config file

## 🚀 Live App

🔗 [Click to launch the app](https://rewant102-assignment-on-pbel-ibm-classes-july-2025.streamlit.app)

> Predict salaries instantly using either your own model or IBM’s cloud deployment!

---

## 🧠 Features

✅ Dual Prediction Mode:
- `Local Model`: Runs with `.pkl` files stored in the project
- `Watsonx API`: Sends input data to IBM Cloud and returns predicted salary

🎯 User Inputs:
- Age
- Country
- Department
- Position
- Years of Experience

📊 Output:
- Predicted Salary displayed in ₹ (Indian Rupees)

---

## 📦 Technologies Used

| Tech           | Usage                            |
|----------------|----------------------------------|
| `Streamlit`    | UI for app                       |
| `scikit-learn` | Preprocessing + Label Encoding   |
| `XGBoost`      | Local ML model                   |
| `joblib`       | Load model + encoders            |
| `IBM Watsonx`  | Cloud prediction endpoint        |
| `python-dotenv`| Local `.env` support             |

---

## 🛠️ Project Structure



---

## 🧪 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Rewant102/assignment_on_pbel_ibm_classes_july_2025.git
cd assignment_on_pbel_ibm_classes_july_2025

# 2. (Optional) Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Watson API key in a .env file
echo WATSON_API_KEY=your_actual_key_here > .env

# 5. Run the app
streamlit run app.py

🌐 Deploy to Streamlit Cloud
Push your app to GitHub

Visit streamlit.io/cloud

Connect your GitHub repo and select:

main branch

app.py as entry point

Go to "Advanced Settings" → Secrets and paste:

Copy and edit 
WATSON_API_KEY = "your_actual_key_here"
✅ Done — your app will deploy in seconds!

✨ Credits
Built by Rewant Prajapati
Project under: PBEL IBM AI Cloud Training – July 2025

📬 Contact
📧 rewantprajapati102@gmail.com
🔗 LinkedIn
💻 GitHub 