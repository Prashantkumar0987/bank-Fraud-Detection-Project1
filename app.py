from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

# ── Replicate the exact same preprocessing used in save_model.py ──────────────
data = pd.read_csv("fraudTrain.csv")
data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street',
                   'city', 'state', 'zip', 'trans_num'], inplace=True)

data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day']  = data['trans_date_trans_time'].dt.dayofweek
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year
data.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

# Fit encoders
label_encoders = {}
for col in ['merchant', 'category', 'job']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Fit scaler
scaler = StandardScaler()
scaler.fit(data[['amt']])

# Load model
model = joblib.load("model/fraud_model.pkl")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Fraud Detection API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.json

        # Encode categorical fields
        merchant_enc = int(label_encoders['merchant'].transform([body['merchant']])[0])
        category_enc = int(label_encoders['category'].transform([body['category']])[0])
        job_enc      = int(label_encoders['job'].transform([body['job']])[0])

        # Scale amount
        amt_scaled = float(scaler.transform([[float(body['amt'])]])[0][0])

        # Build feature vector in the exact order the model was trained on:
        # merchant, category, amt, merch_lat, merch_long, lat, long,
        # city_pop, job, dob_year, trans_hour, trans_day, age
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

if __name__ == "__main__":
    app.run(debug=True)