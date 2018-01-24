import requests
import json

model_server_url = "http://localhost:8000/predict"
params = {"model_id":"HelloWorldExample", "model_version":"1.0.0"}
data = json.dumps(None)
response = requests.post(model_server_url, params = params, json = data)
print "Result:", response.content

# Expected Output
# Result: {"stack_trace": "NA", "result": "\"Hello World\""}