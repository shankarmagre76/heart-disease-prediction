import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your local dataset
df = pd.read_csv("heart.csv")  # Update path if needed

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
print("✅ Model trained with accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Input guide dictionary
feature_guide = {
    "age": "Enter your age (e.g., 30-90): ",
    "sex": "Sex (0 = Female, 1 = Male): ",
    "cp": "Chest Pain Type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic): ",
    "trestbps": "Resting Blood Pressure (e.g., 90–180 mm Hg): ",
    "chol": "Cholesterol level in mg/dl (e.g., 150–400): ",
    "fbs": "Fasting Blood Sugar > 120 mg/dl? (1 = Yes, 0 = No): ",
    "restecg": "Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy): ",
    "thalach": "Maximum Heart Rate Achieved (e.g., 70–210): ",
    "exang": "Exercise-Induced Angina? (1 = Yes, 0 = No): ",
    "oldpeak": "ST depression compared to rest (e.g., 0.0–6.0): ",
    "slope": "Slope of ST segment (0 = upsloping, 1 = flat, 2 = downsloping): ",
    "ca": "Number of major vessels (0–3): ",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect): ",
}

# User input section
print("\n🔍 Please provide the following medical information:\n")
user_input = []

for feature, prompt in feature_guide.items():
    while True:
        try:
            value = float(input(prompt))
            user_input.append(value)
            break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

# Create DataFrame and scale it
input_df = pd.DataFrame([user_input], columns=feature_guide.keys())
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

# Output
print("\n🔎 Prediction Result:")
if prediction[0] == 1:
    print("💔 The person is likely to have heart disease.")
else:
    print("❤️ The person is unlikely to have heart disease.")
