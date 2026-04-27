from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

# 📌 Get correct base path (VERY IMPORTANT for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load dataset safely ─────────────────────────────
data_path = os.path.join(BASE_DIR, "fraudTrain.csv")
data = pd.read_csv(data_path)

data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street',
                   'city', 'state', 'zip', 'trans_num'], inplace=True)

data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day']  = data['trans_date_trans_time'].dt.dayofweek
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year
data.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

# ── Fit encoders ───────────────────────────────────
label_encoders = {}
for col in ['merchant', 'category', 'job']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# ── Fit scaler ─────────────────────────────────────
scaler = StandardScaler()
scaler.fit(data[['amt']])

# ── Load model safely ─────────────────────────────
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
model = joblib.load(model_path)

# ── Routes ────────────────────────────────────────
@app.route("/")
def home():
    return "Fraud Detection API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.json

        merchant_enc = int(label_encoders['merchant'].transform([body['merchant']])[0])
        category_enc = int(label_encoders['category'].transform([body['category']])[0])
        job_enc      = int(label_encoders['job'].transform([body['job']])[0])

        amt_scaled = float(scaler.transform([[float(body['amt'])]])[0][0])

        features = np.array([[
            merchant_enc,
            category_enc,
            amt_scaled,
            float(body['merch_lat']),
            float(body['merch_long']),
            float(body['lat']),
            float(body['long']),
            float(body['city_pop']),
            job_enc,
            float(body['dob_year']),
            float(body['trans_hour']),
            float(body['trans_day']),
            float(body['age']),
        ]])

        prediction = model.predict(features)[0]
        result = "Fraud" if prediction == 1 else "Normal"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# 🚀 IMPORTANT FOR RENDER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
