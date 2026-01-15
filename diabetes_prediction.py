import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv("diabetes.csv")

#data has nan's as zeroes , the following cols shouldnt have 0's 

cols_with_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

#replacing 0's by nan's

df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

#imputing the nans by median 

for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col].fillna(df[col].median(), inplace=True)

#i test-train split

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#scaing for logistic regression 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training for logistic regression 

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"  # important for recall
)

log_reg.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.40).astype(int)

joblib.dump(
    {
        "model": log_reg,
        "scaler": scaler,
        "feature_cols": list(X.columns),
        "threshold": 0.40
    },
    "models/diabetes_logreg_model.pkl"
)

print("done")