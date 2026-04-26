import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_data = pd.read_csv("fraudTrain.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows of dataset:")
print(train_data.head())

# Display shape (rows, columns)
print("\nDataset Shape:")
print(train_data.shape)

# Display column names
print("\nColumn Names:")
print(train_data.columns.tolist())

# Display data types
print("\nData Types:")
print(train_data.dtypes)

# Preprocessing
print("\n--- Preprocessing ---")

# Convert date columns
train_data["trans_date_trans_time"] = pd.to_datetime(train_data["trans_date_trans_time"])
train_data["dob"] = pd.to_datetime(train_data["dob"])

# Drop unnecessary columns
train_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'], inplace=True)

# Drop rows with missing values
train_data.dropna(ignore_index=True, inplace=True)

print("Preprocessing complete!")
print("\nFinal dataset shape:")
print(train_data.shape)
print("\nFinal dataset:")
print(train_data.head())