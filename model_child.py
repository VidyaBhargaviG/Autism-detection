#child
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_child_model():
    df = pd.read_csv("child.csv")

    df = df.drop(columns=["contry_of_res", "age_desc", "relation", "used_app_before", "result", "relation"], errors='ignore')

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier(eval_metric='logloss', verbosity=0)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    questions = {
        "A1_Score": "I often notice small sounds when others do not (0 = No, 1 = Yes)",
        "A2_Score": "I usually concentrate more on the whole picture rather than the small details (0 = No, 1 = Yes)",
        "A3_Score": "I find it easy to do more than one thing at once (0 = No, 1 = Yes)",
        "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly (0 = No, 1 = Yes)",
        "A5_Score": "I find it easy to understand when someone is bored (0 = No, 1 = Yes)",
        "A6_Score": "I like collecting information about categories of things. (0 = No, 1 = Yes)",
        "A7_Score": "I find it easy to understand what people mean by their tone (0 = No, 1 = Yes)",
        "A8_Score": "I enjoy social chit-chat(0 = No, 1 = Yes)",
        "A9_Score": "I am good at working out what people are thinking or feeling(0 = No, 1 = Yes)",
        "A10_Score": "I find it difficult to work out people’s intentions (0 = No, 1 = Yes)",
        "age": "Please enter the patient's age (between 4 and 11)"
    }

    print("\nAnswer the following questions to assess autism risk.")
    user_input = {}
    question_counter = 1

    for col in X.columns:
        if col in questions:
            while True:
                try:
                    if col == "age":
                        age = int(input(f"Q{question_counter}: {questions[col]}: "))
                        if 4 <= age <= 11:
                            user_input[col] = age
                            break
                        else:
                            print("Age must be between 4 and 11. Please try again.")
                    else:
                        value = int(input(f"Q{question_counter}: {questions[col]}: "))
                        if value in [0, 1]:
                            user_input[col] = value
                            break
                        else:
                            print("Please enter 0 or 1.")
                except ValueError:
                    print("Invalid input. Please enter a valid number (0 or 1 for questions).")
        
        elif col == "jundice":
            while True:
                val = input(f"Q{question_counter}: Did the patient have any jaundice history during childhood? (yes/no): ").strip().lower()
                if val in ["yes", "no"]:
                    user_input[col] = label_encoders[col].transform([val])[0]
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
        elif col == "austim":
            while True:
                val = input(f"Q{question_counter}: Are the patient's parents autistic? (yes/no): ").strip().lower()
                if val in ["yes", "no"]:
                    user_input[col] = label_encoders[col].transform([val])[0]
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
    print(f"\nPrediction Result: {label_encoders['Class'].inverse_transform([pred])[0]}")

# Example usage
# run_child_model()
