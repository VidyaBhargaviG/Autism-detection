import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Datasets\\adult.csv")

df.drop(columns=["id", "age_desc", "contry_of_res", "relation", "result", "used_app_before"], errors='ignore', inplace=True)

label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        label_encoders[column] = encoder

X = df.drop("Class/ASD", axis=1)
y = df["Class/ASD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(eval_metric='logloss', verbosity=0)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nModel Performance Summary:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

screening_questions = {
    "A1_Score": "I prefer to do things the same way every time.",
    "A2_Score": "I find it difficult to work out people’s intentions.",
    "A3_Score": "I like to focus on small details rather than the whole picture.",
    "A4_Score": "I find it easy to socialize with others.",
    "A5_Score": "I enjoy doing things spontaneously.",
    "A6_Score": "I often notice patterns or details that others miss.",
    "A7_Score": "I feel comfortable in social situations.",
    "A8_Score": "I find it easy to read facial expressions.",
    "A9_Score": "I enjoy spending time with others.",
    "A10_Score": "I prefer structured environments to unpredictable ones."
}

print("\nAnswer following questions to assess autism risk.")
user_responses = {}
question_number = 1

for feature in X.columns:
    if feature in screening_questions:
        while True:
            try:
                response = int(input(f"Q{question_number}: {screening_questions[feature]} (0 = No, 1 = Yes): "))
                if response in [0, 1]:
                    user_responses[feature] = response
                    break
                else:
                    print("Please enter 0 or 1 only.")
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")

    elif feature == "age":
        while True:
            try:
                age = int(input(f"Q{question_number}: Please enter the patient's age (17–64): "))
                if 17 <= age <= 64:
                    user_responses[feature] = age
                    break
                else:
                    print("Age must be between 17 and 64.")
            except ValueError:
                print("Please enter a valid number for age.")

    elif feature== "gender":
     while True:
        val = input(f"Q{question_number}: What is the patient's gender? (m/f): ").strip().lower()
        if val in ["m", "f"]:
            user_responses["gender"] = label_encoders["gender"].transform([val])[0]
            break
        else:
            print("Invalid input. Please enter 'm' for male or 'f' for female.")
    
    elif feature == "ethnicity":
      while True:
        Ethnicity = input(f"Q{question_number}: Ethnicity (Please enter the ethnicity): ").strip()
        if Ethnicity:
            # Check if the ethnicity is already in the encoder, if not, add it
            if Ethnicity not in label_encoders["ethnicity"].classes_:
                label_encoders["ethnicity"].classes_ = np.append(label_encoders["ethnicity"].classes_, Ethnicity)
            user_responses[feature] = label_encoders["ethnicity"].transform([Ethnicity])[0]
            break
        else:
            print("Ethnicity cannot be empty. Please enter a valid ethnicity.")

    elif feature == "jundice":
        while True:
            history = input(f"Q{question_number}: Did the patient have jaundice as a child? (yes/no): ").strip().lower()
            if history in ["yes", "no"]:
                user_responses[feature] = label_encoders[feature].transform([history])[0]
                break
            else:
                print("Please answer 'yes' or 'no'.")

    elif feature == "austim":
        while True:
            family_history = input(f"Q{question_number}: Is there a family history of autism? (yes/no): ").strip().lower()
            if family_history in ["yes", "no"]:
                user_responses[feature] = label_encoders[feature].transform([family_history])[0]
                break
            else:
                print("Please answer 'yes' or 'no'.")

    elif feature in label_encoders:
        while True:
            try:
                response = int(input(f"Q{question_number}: {feature.replace('_', ' ').capitalize()} (0 = No, 1 = Yes): "))
                if response in [0, 1]:
                    user_responses[feature] = response
                    break
                else:
                    print("Please enter 0 or 1 only.")
            except:
                print("Invalid input. Please enter 0 or 1.")

    else:
        while True:
            try:
                response = int(input(f"Q{question_number}: {feature.replace('_', ' ').capitalize()} (0 = No, 1 = Yes): "))
                if response in [0, 1]:
                    user_responses[feature] = response
                    break
                else:
                    print("Please enter 0 or 1 only.")
            except:
                print("Invalid input. Please enter 0 or 1.")
    
    question_number += 1

# Make a prediction based on user input
input_df = pd.DataFrame([user_responses])
prediction = best_model.predict(input_df)[0]
predicted_label = label_encoders["Class/ASD"].inverse_transform([prediction])[0]

print(f"\nPrediction Result: {predicted_label}")
