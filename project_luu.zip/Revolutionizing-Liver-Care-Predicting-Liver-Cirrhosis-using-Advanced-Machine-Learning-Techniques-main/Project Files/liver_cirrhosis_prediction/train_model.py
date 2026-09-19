# FINAL train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("indian_liver_patient.csv")
df = df.dropna()

# Encode Gender
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Map labels to binary
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

# Balance dataset
df_majority = df[df['Dataset'] == 1]
df_minority = df[df['Dataset'] == 0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1).reset_index(drop=True)

# Use correct column order
feature_cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                'Aspartate_Aminotransferase', 'Total_Protiens',
                'Albumin', 'Albumin_and_Globulin_Ratio']
X = df_balanced[feature_cols]
y = df_balanced['Dataset']

print("\n🧾 Training feature order:", feature_cols)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Evaluate model
print("\n📊 Classification Report:")
y_pred = model.predict(X_scaled)
print(classification_report(y, y_pred))

# Save model and scaler
pickle.dump(model, open("rf_acc_68.pkl", "wb"))
pickle.dump(scaler, open("normalizer.pkl", "wb"))

print("\n✅ Model and scaler saved successfully.")