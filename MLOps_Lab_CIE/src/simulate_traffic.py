import requests
import json
import random
import time
import os

url = "http://127.0.0.1:9000/infer"

os.makedirs("logs", exist_ok=True)

for i in range(50):
    data = {
        "pr_lines_changed": random.randint(500, 3000),
        "file_count": random.randint(5, 60),
        "author_experience_months": random.randint(1, 120),
        "is_refactor": random.randint(0, 1)
    }

    response = requests.post(url, json=data)

    res = response.json()

    if "prediction" in res:
        log_entry = {
            "input": data,
            "prediction": res["prediction"]
        }

        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    else:
        print("Error response:", res)

    time.sleep(0.1)

print("Traffic simulation done")