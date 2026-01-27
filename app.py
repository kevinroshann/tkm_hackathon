from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
resource_model = joblib.load("hospital_forecast.pkl")
patient_model = joblib.load("patient_forecast.pkl")

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Healthcare AI API online"})

# ---------------------
# RESOURCE FORECAST API
# ---------------------
@app.route("/predict/resources", methods=["POST"])
def predict_resources():
    data = request.get_json()
    df = pd.DataFrame([data])  # expects JSON like: {"patients":80, "icu_cases":10, ...}
    y = resource_model.predict(df)[0]
    return jsonify({
        "med_A": y[0],
        "med_B": y[1],
        "med_C": y[2],
        "oxygen": y[3],
        "syringes": y[4]
    })

# ---------------------
# PATIENT SPIKE FORECAST API
# ---------------------
@app.route("/predict/patients", methods=["POST"])
def predict_patients():
    data = request.get_json()
    df = pd.DataFrame([data])
    y = patient_model.predict(df)[0]
    return jsonify({
        "predicted_patients": int(y)
    })

if __name__ == "__main__":
    app.run(debug=True)
