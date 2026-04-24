import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("fraudTrain.csv")

print("✅ Dataset Loaded Successfully")
print("Shape:", data.shape)

# Check missing values
print("\n🔍 Checking Missing Values:")
print(data.isnull().sum())

# Drop irrelevant columns
data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street',
                   'city', 'state', 'zip', 'trans_num'], inplace=True)

# Convert date columns
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

# Feature Engineering — extract useful info from dates
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day']  = data['trans_date_trans_time'].dt.dayofweek
data['age']        = data['trans_date_trans_time'].dt.year - data['dob'].dt.year

# Drop original date columns (no longer needed)
data.drop(columns=['trans_date_trans_time', 'dob', 'dob_year'], inplace=True)

print("\n✅ Date columns processed, new features: trans_hour, trans_day, age")

# Encode categorical columns
le = LabelEncoder()
for col in ['merchant', 'category', 'job']:
    data[col] = le.fit_transform(data[col].astype(str))

print("✅ Categorical columns encoded")

# Feature Scaling (Normalize 'amt')
scaler = StandardScaler()
data['amt'] = scaler.fit_transform(data[['amt']])

print("✅ 'amt' column scaled")

# Split features and target
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

print("\n✅ Features and target separated")
print("Features:", list(X.columns))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display shapes
print("\n📊 Data Split:")
print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)

print("\n🎉 Day 3 Completed Successfully!")
