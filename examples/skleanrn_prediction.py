import requests
import simplejson

# Pick one example for prediction
from sklearn import datasets
digits = datasets.load_digits()
test_data_X = digits.data[-1:]

model_server_url = "http://localhost:8000/predict"
params = {"model_id":"SKLearnExample", "model_version":"1.0.0"}
data = simplejson.dumps(test_data_X.tolist())
response = requests.post(model_server_url, params = params, json = data)
print "Result:", response.content

# Expected Output
# Result: {"stack_trace": "NA", "result": "[8]"}