import json
import os
from hunchsdk.HunchApi import HunchApi

os.environ["HUNCH_API_CONFIG"]="../config/hunch_api_config.yaml"

class HelloWorldModel:
    def predict(self, input):
        return json.dumps("Hello World")

model = HelloWorldModel()
print "Local Prediction:", model.predict(None)
HunchApi().publish_model(model, "HelloWorldExample", "1.0.0")