from fastapi import FastAPI
import pickle
from backend.pydantic_m import Passenger, PredictionResponse
import numpy as np
import pandas as pd





app = FastAPI()


## load the model 
model = pickle.load(open("backend/titanic_model.pkl","rb"))
scaler = pickle.load(open("backend/scaler.pkl","rb"))

@app.get("/")
def Home():
    return {"message":"Titanic survival API is Running"}




@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: Passenger):

    features = prepare_features(passenger)

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return {
        "prediction": int(prediction[0]),
        "survival_probability": float(probability[0][1])
    }


def prepare_features(passenger: Passenger):

    # Normalize input
    sex = passenger.Sex.value
    embarked = passenger.Embarked.value
    family_type = passenger.FamilyType.value

    # Estimate Fare automatically based on Pclass (mean values from training data)
    if passenger.Pclass == 1:
        estimated_fare = 84.0
    elif passenger.Pclass == 2:
        estimated_fare = 20.0
    else:
        estimated_fare = 13.0

    # Encode Pclass
    Pclass_2 = 1 if passenger.Pclass == 2 else 0
    Pclass_3 = 1 if passenger.Pclass == 3 else 0

    # Encode Embarked
    Embarked_Q = 1 if embarked == "Q" else 0
    Embarked_S = 1 if embarked == "S" else 0

    # Encode Family Type
    family_type_Large = 1 if family_type == "large" else 0
    family_type_Medium = 1 if family_type == "medium" else 0

    # Encode Title from Sex (simplified logic)
    Title_Mr = 1 if sex == "male" else 0
    Title_Miss = 1 if sex == "female" else 0
    Title_Mrs = 0
    Title_Rare = 0

    # Create DataFrame with correct feature names and order
    features = pd.DataFrame([{
        "Age": passenger.Age,
        "Fare": estimated_fare,
        "Pclass_2": Pclass_2,
        "Pclass_3": Pclass_3,
        "Embarked_Q": Embarked_Q,
        "Embarked_S": Embarked_S,
        "family_type_Large": family_type_Large,
        "family_type_Medium": family_type_Medium,
        "Title_Miss": Title_Miss,
        "Title_Mr": Title_Mr,
        "Title_Mrs": Title_Mrs,
        "Title_Rare": Title_Rare
    }])

    # Scale only Age & Fare
    features[["Age", "Fare"]] = scaler.transform(features[["Age", "Fare"]])

    return features
