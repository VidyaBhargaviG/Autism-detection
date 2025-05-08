import requests

url = "http://127.0.0.1:5000/get_diet"
data = {
    "age": 30,
    "bmi": 25,
    "chronic_disease": "Diabetes",
    "blood_pressure_sys": 130,
    "blood_pressure_dia": 85,
    "cholesterol": 190,
    "blood_sugar": 110
}
response = requests.post(url, json=data)
print(response.json())
