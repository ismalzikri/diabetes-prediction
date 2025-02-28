import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import ssl
import certifi

# Ensure proper SSL verification
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Dataset URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Try downloading dataset with SSL verification
try:
    response = requests.get(url, verify=True)  # Enable SSL verification
    response.raise_for_status()  # Raise error if request fails
    data = pd.read_csv(StringIO(response.text), header=None, names=columns)
    print("✅ Dataset loaded from URL successfully.")
except requests.exceptions.RequestException as e:
    print(f"⚠️ Failed to fetch dataset from URL: {e}\nUsing local dataset instead.")
    data = pd.read_csv("pima-indians-diabetes.data.csv", header=None, names=columns)

# Split data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'diabetes_model.pkl'")
