import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("diet_data.csv")

# Convert categorical values
df["Gender"] = df["Gender"].str.lower().map({"male": 0, "female": 1})
df["Chronic_Disease"] = df["Chronic_Disease"].astype("category").cat.codes  

X = df[["Age", "Gender", "BMI", "Chronic_Disease", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Cholesterol_Level", "Blood_Sugar_Level"]]
y = df[["Recommended_Calories", "Recommended_Protein", "Recommended_Carbs", "Recommended_Fats"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open("diet_model.pkl", "wb"))

print("Model trained and saved successfully!")
