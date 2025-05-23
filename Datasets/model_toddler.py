#toddler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_toddler_model(input_data=None):
    # Load the dataset
    df = pd.read_csv("Datasets\\toddler.csv")

    # Preprocess the data
    df.drop(columns=["Qchat-10-Score", "Who completed the test"], errors='ignore', inplace=True)

    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            label_encoders[column] = encoder

    X = df.drop("Class/ASD", axis=1)
    y = df["Class/ASD"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with GridSearchCV for hyperparameter tuning
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

    # Print the performance summary
    print("\nModel Performance Summary:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Screening questions
    screening_questions = {
        "A1": "Does your child respond when their name is called?",
        "A2": "Does your child make eye contact when interacting?",
        "A3": "Does your child enjoy being held or cuddled?",
        "A4": "Does your child show interest in other children?",
        "A5": "Does your child imitate sounds or facial expressions?",
        "A6": "Does your child use gestures to communicate (e.g., waving)?",
        "A7": "Does your child point to objects to show interest?",
        "A8": "Does your child smile back when someone smiles at them?",
        "A9": "Does your child engage in pretend play (e.g., feeding a doll)?",
        "A10": "Does your child appear sensitive to loud noises?"
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

        elif feature == "Age_Months":
            while True:
                try:
                    age =float(input(f"Q{question_number}: Please enter the patient's age (1–3): "))
                    if 1<= age <=3:
                       user_responses[feature] = age
                       break
                    else:
                       print("Age must be between 1 and 3.")
                except ValueError:
                    print("Please enter a valid number for age.")
        elif feature == "gender":
            while True:
                val = input(f"Q{question_number}: What is the patient's gender? (m/f): ").strip().lower()
                if val in ["m", "f"]:
                    user_responses["gender"] = label_encoders["gender"].transform([val])[0]
                    break
                else:
                    print("Invalid input. Please enter 'm' for male or 'f' for female.")

        elif feature == "Ethnicity":
            while True:
                Ethnicity = input(f"Q{question_number}: Ethnicity (Please enter the ethnicity): ").strip()
                if Ethnicity:
            # Check if the ethnicity is already in the encoder, if not, add it
                    if Ethnicity not in label_encoders["Ethnicity"].classes_:
                        label_encoders["Ethnicity"].classes_ = np.append(label_encoders["Ethnicity"].classes_, Ethnicity)
                    user_responses[feature] = label_encoders["Ethnicity"].transform([Ethnicity])[0]
                    break
                else:
                    print("Ethnicity cannot be empty. Please enter a valid ethnicity.")

        elif feature == "Jaundice":
            while True:
                history = input(f"Q{question_number}: Did the patient have jaundice as a child? (yes/no): ").strip().lower()
                if history in ["yes", "no"]:
                    user_responses[feature] = label_encoders[feature].transform([history])[0]
                    break
                else:
                    print("Please answer 'yes' or 'no'.")

        elif feature == "Family_mem_with_ASD":
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
    predicted_label = label_encoders["Class/ASD "].inverse_transform([prediction])[0]

    print(f"\nPrediction Result: {predicted_label}")
