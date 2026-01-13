import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("heart_disease.csv")

# assgining genders, male to 1 and female to 0 


df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})


#dropping NAN's in gender
df = df.dropna(subset=["Gender"])

#dropping nan in bp 
df = df.dropna(subset=["Blood Pressure"])

#Mapping excersie habits
df["Exercise Habits"] = df["Exercise Habits"].map({
    "High": 2,
    "Medium": 1,
    "Low": 0
})

#dropping nans in excersice habbits 
df = df.dropna(subset=["Exercise Habits"])

# mapping and dropping smoking 
df["Smoking"] = df["Smoking"].map({
    "Yes": 1,
    "No": 0
})

df = df.dropna(subset=["Smoking"])

#mapping and dropping in family history
df["Family Heart Disease"] = df["Family Heart Disease"].map({
    "Yes": 1,
    "No": 0
})

df = df.dropna(subset=["Family Heart Disease"])
 
#mapping and dropping in diabetes

df["Diabetes"] = df["Diabetes"].map({
    "Yes": 1,
    "No": 0
})
df = df.dropna(subset=["Diabetes"])

# dropping nans in bmi 

df = df.dropna(subset=["BMI"])

# mapping and dropping in bp

df["High Blood Pressure"] = df["High Blood Pressure"].map({
    "Yes": 1,
    "No": 0
})
df = df.dropna(subset=["High Blood Pressure"])
 

# mapping alcohol , high -> 2 , mid -> 1 , low -> 0  and dropping 
# dropping whole alcohol column , high missingness 25% of data unreported

df = df.drop(columns=["Alcohol Consumption"])

# mapping stress , high -> 2 , mid -> 1 , low -> 0  and dropping 

df["Stress Level"] = df["Stress Level"].map({
    "High": 2,
    "Medium": 1,
    "Low": 0
})

df = df.dropna(subset=["Stress Level"])

# dropping in sleep hrs 
df = df.dropna(subset=["Sleep Hours"])

#mapping sugar , high -> 2 , mid -> 1 , low -> 0  and dropping

df["Sugar Consumption"] = df["Sugar Consumption"].map({
    "High": 2,
    "Medium": 1,
    "Low": 0
})

df = df.dropna(subset=["Sugar Consumption"])

#dropping nans in fbs 

df = df.dropna(subset=["Fasting Blood Sugar"])

#dropping homocystene and crp level columns , - need labwork 

df = df.drop(columns=[
    "CRP Level",
    "Homocysteine Level"
])

#mapping heart status yes/no to 1/0
df["heart_disease"] = df["Heart Disease Status"].map({
    "Yes": 1,
    "No": 0
})
#dropping irrelevant columns 

df = df.drop(columns=[
    "Heart Disease Status"
])
df = df.dropna(subset=["Age"])

lipid_cols = [
    "Cholesterol Level",
    "Triglyceride Level",
    "High LDL Cholesterol",
    "Low HDL Cholesterol"
]

df = df.drop(columns=[c for c in lipid_cols if c in df.columns])



#building x and y 

feature_cols = [
    "Age",                    # fixed
    "Gender",                 # fixed
    "Blood Pressure",         # optional 
    "BMI",                    # optional
    "Smoking",                # changeable
    "Exercise Habits",        # changeable
    "Sleep Hours",            # changeable
    "Stress Level",           # changeable
    "Sugar Consumption",      # changeable
    "Diabetes",               # fixed
    "Family Heart Disease",   # fixed
    "High Blood Pressure"     # fixed flag
]

X = df[feature_cols]
y = df["heart_disease"]


#traning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

model.fit(X, y)
# for ui 

def assess_baseline_risk(
    age_x,
    gender_x,
    blood_pressure_x,
    bmi_x,
    smoking_x,
    exercise_x,
    sleep_hours_x,
    stress_x,
    sugar_x,
    diabetes_x,
    family_history_x,
    high_bp_x
):
    user_dict = {
        "Age": age_x,
        "Gender": gender_x,
        "Blood Pressure": blood_pressure_x,
        "BMI": bmi_x,
        "Smoking": smoking_x,
        "Exercise Habits": exercise_x,
        "Sleep Hours": sleep_hours_x,
        "Stress Level": stress_x,
        "Sugar Consumption": sugar_x,
        "Diabetes": diabetes_x,
        "Family Heart Disease": family_history_x,
        "High Blood Pressure": high_bp_x
    }

    user_df = pd.DataFrame([user_dict], columns=feature_cols)
    risk = model.predict_proba(user_df)[0][1]

    return round(risk, 3), user_dict

def assess_updated_risk(baseline_user_dict, updated_fields_dict):
    updated_user = baseline_user_dict.copy()

    # override only selected fields
    for key, value in updated_fields_dict.items():
        updated_user[key] = value

    updated_df = pd.DataFrame([updated_user], columns=feature_cols)
    updated_risk = model.predict_proba(updated_df)[0][1]

    return round(updated_risk, 3)


if __name__ == "__main__":
    pass
