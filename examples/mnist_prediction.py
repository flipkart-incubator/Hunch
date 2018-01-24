import os
import json
import requests


model_id = "TensorflowMnistExample"
model_version = "1.0.0"
params = {"model_id": model_id, "model_version": model_version}
data = [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]]
json_payload = json.dumps(data)
response = requests.post("http://localhost:8000/predict", params=params, data=json_payload)
result = json.loads(response.text)['result']
print result