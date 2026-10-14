import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Employee Prediction Portal",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Modern Minimalist & Professional Styling (CSS)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
    }
    .result-success {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 1.25rem;
        border-radius: 12px;
        color: #065F46;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.05);
    }
    .result-danger {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        padding: 1.25rem;
        border-radius: 12px;
        color: #991B1B;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.05);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Model Function (Using naive.pkl)
@st.cache_resource
def load_model():
  with open("naive.pkl", "rb") as file:
    model = pickle.load(file)
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      f"Error loading naive.pkl. Ensure your file is in the root directory."
      f" Details: {e}"
  )
  st.stop()

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1>Employee Profile Analysis</h1>
        <p>Gaussian Naive Bayes Model Prediction Portal</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Form matching exact features from your model[cite: 1]
with st.form("prediction_form"):
  st.markdown("### 📋 Candidate Information")

  col1, col2 = st.columns(2)

  with col1:
    education = st.selectbox(
        "Education", ["Bachelors", "Masters", "PHD"], index=0
    )
    joining_year = st.number_input(
        "Joining Year", min_value=2010, max_value=2026, value=2018, step=1
    )
    city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"], index=0)
    payment_tier = st.selectbox(
        "Payment Tier", [1, 2, 3], help="1: Highest, 3: Lowest"
    )

  with col2:
    age = st.number_input("Age", min_value=18, max_value=65, value=27, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    ever_benched = st.selectbox("Ever Benched", ["No", "Yes"], index=0)
    experience = st.number_input(
        "Experience in Current Domain (Years)",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
    )

  st.markdown("<br>", unsafe_allow_html=True)
  submit_button = st.form_submit_button(label="Evaluate Profile 🚀")

# Prediction Execution
if submit_button:
  # Standard Ordinal Encoding mapping for typical categorical inputs matching this specific dataset schema
  edu_mapping = {"Bachelors": 0, "Masters": 1, "PHD": 2}
  city_mapping = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}

  # Build DataFrame ensuring columns precisely match the model's feature names[cite: 1]
  input_data = pd.DataFrame(
      {
          "Education": [edu_mapping.get(education, 0)],
          "JoiningYear": [joining_year],
          "City": [city_mapping.get(city, 0)],
          "PaymentTier": [payment_tier],
          "Age": [age],
          "Gender": [1 if gender == "Male" else 0],
          "EverBenched": [1 if ever_benched == "Yes" else 0],
          "ExperienceInCurrentDomain": [experience],
      }
  )

  try:
    prediction = model.predict(input_data)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
      st.markdown(
          """
                <div class="result-success">
                    ✨ Result: Positive Outcome / High Retention Probability
                </div>
            """,
          unsafe_allow_html=True,
      )
    else:
      st.markdown(
          """
                <div class="result-danger">
                    ⚠️ Result: Risk Flagged / Attrition Indicated
                </div>
            """,
          unsafe_allow_html=True,
      )
  except Exception as e:
    st.error(
        f"Prediction execution error: {e}. Please verify input encoding formats."
    )
