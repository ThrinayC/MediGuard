
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

from xgboost import XGBClassifier


df = pd.read_csv("heart_data.csv")

y = df["diagnosis"]

y = (y > 0).astype(int)

X = df.drop(columns=["diagnosis"])

#slplitting for trainign and testin

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#training xgboost 

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.4).astype(int)

#defining a function for easier ui use 

def predict_heart_disease_probability(
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
):
   
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    input_df = pd.DataFrame(
        input_data,
        columns=X.columns
    )

    prob = xgb_model.predict_proba(input_df)[0][1]

    return float(round(prob, 3))

