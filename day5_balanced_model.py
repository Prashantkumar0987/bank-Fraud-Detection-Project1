import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("fraudTrain.csv")

# Drop unnecessary columns
data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street',
                   'city', 'state', 'zip', 'trans_num'], inplace=True)

# Convert dates
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

# Feature engineering
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day']  = data['trans_date_trans_time'].dt.dayofweek
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year

data.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

# Encode categorical columns
for col in ['merchant', 'category', 'job']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Scale amount
scaler = StandardScaler()
data['amt'] = scaler.fit_transform(data[['amt']])

# 🔥 Handle Imbalance (Undersampling)
fraud = data[data['is_fraud'] == 1]
normal = data[data['is_fraud'] == 0].sample(len(fraud), random_state=42)

balanced_data = pd.concat([fraud, normal])

# Shuffle data
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Split features & target
X = balanced_data.drop('is_fraud', axis=1)
y = balanced_data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Improved Model Trained")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

print("\n🎉 Day 5 Completed Successfully!")