from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.staticfiles import StaticFiles

# Adjust path to reach models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessor
model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))

# Define schema for incoming data
class Passenger(BaseModel):
    Age: float
    Fare: float
    Sex: str
    Embarked: str

# Initialize FastAPI app
app = FastAPI()

# Mount static frontend assets
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")

# Single, enhanced /predict endpoint
@app.post("/predict")
def predict(passenger: Passenger):
    df = pd.DataFrame([passenger.dict()])
    print("Received input:", df)

    try:
        X = preprocessor.transform(df)
        prediction = model.predict(X)
        result = "Survived" if prediction[0] == 1 else "Did not survive"
        print("Prediction result:", result)
        return {"prediction": result}
    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)