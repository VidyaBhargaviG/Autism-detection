#main llm
from model_toddler import run_toddler_model
from model_child import run_child_model
from model_teen import run_teen_model
from model_adult import run_adult_model

def get_age_group(age):
    if 1 <= age <= 3:
        return "toddler"
    elif 4 <= age <= 11:
        return "child"
    elif 12 <= age <= 16:
        return "teen"
    elif 17 <= age <= 64:
        return "adult"
    else:
        return None

def main():
    try:
        age = int(input("Please enter the patient's age: "))
        group = get_age_group(age)

        print(f"Redirecting to {group} model...")

        if group == "toddler":
            run_toddler_model()
        elif group == "child":
            run_child_model()
        elif group == "teen":
            run_teen_model()
        elif group == "adult":
            run_adult_model()
        else:
            print("Sorry, we currently support ages between 1 and 64 only.")
    except ValueError:
        print("Invalid input. Please enter a valid age.")

if __name__ == "__main__":
    main()
