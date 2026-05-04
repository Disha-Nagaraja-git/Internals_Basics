import joblib

# Load model
model = joblib.load("../models/best_model.pkl")

# Example input (same format as your dataset)
sample = [[300, 49, 83, 0]]  

# Predict
prediction = model.predict(sample)

print("Prediction:", prediction)