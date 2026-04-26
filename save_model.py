import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
data = pd.read_csv("fraudTrain.csv")

# Drop unnecessary columns
data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street',
                   'city', 'state', 'zip', 'trans_num'], inplace=True)

# Feature engineering
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day'] = data['trans_date_trans_time'].dt.dayofweek
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year

data.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

# Encoding
for col in ['merchant', 'category', 'job']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Scaling
scaler = StandardScaler()
data['amt'] = scaler.fit_transform(data[['amt']])

# Split
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Save
import os
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.pkl")

print("Model saved successfully in /model folder")