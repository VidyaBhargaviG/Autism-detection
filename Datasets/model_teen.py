#teen
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_teen_model():
    global best_model, X, class_encoder, label_encoders

    df = pd.read_csv("Datasets\\teen.csv")

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col != "Class":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(df["Class"])

    X = df.drop("Class", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric='logloss', verbosity=0)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    print("\nModel Performance:")
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    a_score_questions = {
        "A1": "I prefer to do things the same way every time.",
        "A2": "I find it difficult to work out people’s intentions.",
        "A3": "I like to focus on small details rather than the whole picture.",
        "A4": "I find it easy to socialize with others.",
        "A5": "I enjoy doing things spontaneously.",
        "A6": "I often notice patterns or details that others miss.",
        "A7": "I feel comfortable in social situations.",
        "A8": "I find it easy to read facial expressions.",
        "A9": "I enjoy spending time with others.",
        "A10": "I prefer structured environments to unpredictable ones.",
    }

    print("\nAnswer the following questions to assess autism risk.")
    user_input = {}
    question_counter = 1

    for col in X.columns:
        if col in a_score_questions:
            while True:
                try:
                    value = int(input(f"Q{question_counter}: {a_score_questions[col]} (0 = No, 1 = Yes)?:"))
                    if value in [0, 1]:
                        user_input[col] = value
                        break
                    else:
                        print("Please enter 0 or 1.")
                except ValueError:
                    print("Invalid input. Enter 0 or 1.")
        elif col == "age":
            while True:
                try:
                    age = int(input(f"Q{question_counter}: Please enter the patient's age (between 12 and 17): "))
                    if 12 <= age <= 17:
                        user_input[col] = age
                        break
                    else:
                        print("Age must be between 12 and 17 years. Please try again.")
                except ValueError:
                    print("Please enter a valid numeric value for age.")
        
        elif col == "gender":
            while True:
                val = input(f"Q{question_counter}: What is the patient's gender? (m/f): ").strip().lower()
                if val in ["m", "f"]:
                    user_input["gender"] = label_encoders["gender"].transform([val])[0]
                    break
                else:
                    print("Invalid input. Please enter 'm' for male or 'f' for female.")
        
        elif col == "ethnicity":
            while True:
                Ethnicity = input(f"Q{question_counter}: Ethnicity (Please enter the ethnicity): ").strip()
                if Ethnicity:
                    if Ethnicity not in label_encoders["ethnicity"].classes_:
                        label_encoders["ethnicity"].classes_ = np.append(label_encoders["ethnicity"].classes_, Ethnicity)
                    user_input[col] = label_encoders["ethnicity"].transform([Ethnicity])[0]
                    break
                else:
                    print("Ethnicity cannot be empty. Please enter a valid ethnicity.")

        elif col == "Jaundice":
            while True:
                val = input(f"Q{question_counter}: Did the patient have any jaundice history during childhood? (yes/no): ").strip().lower()
                if val in ["yes", "no"]:
                    user_input[col] = label_encoders[col].transform([val])[0]
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
        
        elif col == "Family_ASD":
            while True:
                val = input(f"Q{question_counter}: Are the patient's parents autistic? (yes/no): ").strip().lower()
                if val in ["yes", "no"]:
                    user_input["Family_ASD"] = label_encoders["Family_ASD"].transform([val])[0]
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
        
        elif col in label_encoders:
            while True:
                val = input(f"Q{question_counter}: {col}: ").strip()
                try:
                    user_input[col] = label_encoders[col].transform([val])[0]
                    break
                except ValueError:
                    print(f"Invalid input. Expected one of: {list(label_encoders[col].classes_)}")
        else:
            while True:
                try:
                    val = float(input(f"Q{question_counter}: {col}: "))
                    user_input[col] = val
                    break
                except ValueError:
                    print("Please enter a valid number.")
        question_counter += 1

    user_df = pd.DataFrame([user_input])
    pred = best_model.predict(user_df)[0]
    result = class_encoder.inverse_transform([pred])[0]
    print(f"\nPrediction Result: {result}")
