from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    pr_lines_changed: int = Field(..., ge=10, le=2000)
    file_count: int = Field(..., ge=1, le=50)
    author_experience_months: int = Field(..., ge=1, le=120)
    is_refactor: int = Field(..., ge=0, le=1)

@app.get("/status")
def status():
    return {"alive": True, "service": "ReviewBot bug_detection_count API"}

@app.post("/infer")
def infer(data: InputData):
    features = [[
        data.pr_lines_changed,
        data.file_count,
        data.author_experience_months,
        data.is_refactor
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}