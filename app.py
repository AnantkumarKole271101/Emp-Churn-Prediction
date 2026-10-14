import pickle
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__)

# Load Model
try:
  with open("naive.pkl", "rb") as file:
    model = pickle.load(file)
except Exception as e:
  print(f"Error loading model: {e}")


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

      # Categorical Encodings matching training data layout
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

  return render_template(
      "index.html",
      prediction_result=prediction_result,
      result_type=result_type,
  )


if __name__ == "__main__":
  app.run(debug=True)
