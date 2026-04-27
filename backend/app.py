from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# 📌 Base directory (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📌 Load trained model
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "✅ Fraud Detection API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # 🔥 EXPECT EXACT 13 FEATURES (same as training)
        features = np.array([[
            float(data['merchant']),
            float(data['category']),
            float(data['amt']),
            float(data['merch_lat']),
            float(data['merch_long']),
            float(data['lat']),
            float(data['long']),
            float(data['city_pop']),
            float(data['job']),
            float(data['dob_year']),
            float(data['trans_hour']),
            float(data['trans_day']),
            float(data['age'])
        ]])

        prediction = model.predict(features)[0]
        result = "Fraud" if prediction == 1 else "Normal"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# 🚀 Required for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
