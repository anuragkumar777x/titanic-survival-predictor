from pydantic import BaseModel, Field
from enum import Enum

# ---- ENUMS ----

class SexEnum(str, Enum):
    male = "male"
    female = "female"

class EmbarkedEnum(str, Enum):
    S = "S"
    Q = "Q"
    C = "C"

class FamilyTypeEnum(str, Enum):
    alone = "alone"
    medium = "medium"
    large = "large"

# ---- REQUEST MODEL ----

class Passenger(BaseModel):
    Age: float = Field(..., gt=0, lt=100)
    Pclass: int = Field(..., ge=1, le=3)
    Sex: SexEnum
    Embarked: EmbarkedEnum
    FamilyType: FamilyTypeEnum

# ---- RESPONSE MODEL ----

class PredictionResponse(BaseModel):
    prediction: int
    survival_probability: float