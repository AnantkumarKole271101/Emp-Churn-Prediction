import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load Model
try:
  with open("naive.pkl", "rb") as file:
    model = pickle.load(file)
except Exception as e:
  print(f"Error loading model: {e}")

# HTML Template embedded directly into Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Analytics Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #F8FAFC;
            font-family: 'Inter', sans-serif;
            color: #1E293B;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 650px;
            background: #FFFFFF;
            border-radius: 16px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.05);
            border: 1px solid #E2E8F0;
            overflow: hidden;
            margin: 20px 0;
        }
        .header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            padding: 2.5rem 2rem;
            color: white;
            text-align: center;
        }
        .header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.4rem; }
        .header p { color: #94A3B8; font-size: 0.95rem; }
        
        .form-content { padding: 2rem; }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }
        .form-group {
            margin-bottom: 1.25rem;
        }
        label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #475569;
        }
        select, input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            font-size: 0.95rem;
            background: #F8FAFC;
            transition: all 0.2s ease;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #2563EB;
            background: #FFFFFF;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        .btn-submit {
            width: 100%;
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: white;
            font-weight: 600;
            padding: 0.85rem;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
            transition: all 0.3s ease;
            margin-top: 1rem;
            font-size: 1rem;
        }
        .btn-submit:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
        }
        .result-box {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 1.05rem;
        }
        .success {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            color: #065F46;
        }
        .danger {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            color: #991B1B;
        }
        @media (max-width: 600px) {
            .grid-2 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Employee Profile Analysis</h1>
            <p>Gaussian Naive Bayes Prediction Portal</p>
        </div>
        <div class="form-content">
            <form method="POST">
                <div class="grid-2">
                    <div class="form-group">
                        <label>Education Level</label>
                        <select name="Education">
                            <option value="Bachelors">Bachelors</option>
                            <option value="Masters">Masters</option>
                            <option value="PHD">PHD</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Joining Year</label>
                        <input type="number" name="JoiningYear" min="2010" max="2026" value="2018" required>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="form-group">
                        <label>City</label>
                        <select name="City">
                            <option value="Bangalore">Bangalore</option>
                            <option value="New Delhi">New Delhi</option>
                            <option value="Pune">Pune</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Payment Tier (1: High, 3: Low)</label>
                        <select name="PaymentTier">
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                        </select>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="Age" min="18" max="65" value="27" required>
                    </div>
                    <div class="form-group">
                        <label>Gender</label>
                        <select name="Gender">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="form-group">
                        <label>Ever Benched</label>
                        <select name="EverBenched">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Experience in Current Domain (Years)</label>
                        <input type="number" name="ExperienceInCurrentDomain" min="0" max="20" value="2" required>
                    </div>
                </div>

                <button type="submit" class="btn-submit">Evaluate Profile 🚀</button>
            </form>

            {% if prediction_result %}
                <div class="result-box {{ result_type }}">
                    {{ prediction_result }}
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction_result = None
  result_type = None

  if request.method == "POST":
    try:
      # Extract form inputs
      education = request.form.get("Education")
      joining_year = int(request.form.get("JoiningYear"))
      city = request.form.get("City")
      payment_tier = int(request.form.get("PaymentTier"))
      age = int(request.form.get("Age"))
      gender = request.form.get("Gender")
      ever_benched = request.form.get("EverBenched")
      experience = int(request.form.get("ExperienceInCurrentDomain"))

      # Categorical Encodings matching training layout
      edu_mapping = {"Bachelors": 0, "Masters": 1, "PHD": 2}
      city_mapping = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}

      # Build input dataframe matching model feature names
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

      # Make Prediction
      prediction = model.predict(input_data)[0]

      if prediction == 1:
        prediction_result = (
            "✨ Result: Positive Outcome / High Retention Probability"
        )
        result_type = "success"
      else:
        prediction_result = "⚠️ Result: Risk Flagged / Attrition Indicated"
        result_type = "danger"

    except Exception as e:
      prediction_result = f"Prediction Error: {str(e)}"
      result_type = "danger"

  return render_template_string(
      HTML_TEMPLATE,
      prediction_result=prediction_result,
      result_type=result_type,
  )


if __name__ == "__main__":
  app.run(debug=True)
