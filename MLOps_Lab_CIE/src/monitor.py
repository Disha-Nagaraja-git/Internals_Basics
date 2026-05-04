import json
import numpy as np
import os

file_path = "logs/predictions.jsonl"

if not os.path.exists(file_path):
    print("No predictions file found!")
    exit()

preds = []

with open(file_path) as f:
    for line in f:
        data = json.loads(line)
        preds.append(data["prediction"])

if len(preds) == 0:
    print("No predictions found!")
    exit()

preds = np.array(preds)

print("📊 Monitoring Report")
print("---------------------")
print("Total predictions:", len(preds))
print("Mean:", np.mean(preds))
print("Std Dev:", np.std(preds))
print("Min:", np.min(preds))
print("Max:", np.max(preds))

# Simple drift check
if np.std(preds) > 10:
    print("⚠️ WARNING: Possible data drift detected!")
else:
    print("✅ Model looks stable")