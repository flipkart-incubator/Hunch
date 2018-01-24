import os
import numpy
import simplejson
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import Normalizer
from hunchsdk.HunchApi import HunchApi


os.environ["HUNCH_API_CONFIG"]="../config/hunch_api_config.yaml"

class Model:
    def train(self, training_data_X, training_data_Y):
        self.normalizer = Normalizer()
        self.svc = svm.SVC(gamma=0.001, C=100.)
        normalised_training_data_X = self.normalizer.fit_transform(training_data_X)
        self.svc.fit(normalised_training_data_X, training_data_Y)

    def predict(self, given_input):
        input_for_prediction = numpy.array(simplejson.loads(given_input))
        prediction = self.svc.predict(self.normalizer.fit_transform(input_for_prediction))
        return simplejson.dumps(prediction.tolist())

    def predict_local(self, given_input):
        prediction = self.svc.predict(self.normalizer.fit_transform(given_input))
        return prediction


# Prepare the dataset
digits = datasets.load_digits()
training_data_X = digits.data
training_data_Y = digits.target

# Pick an example for local prediction
test_data_X = digits.data[-1:]
test_data_Y = digits.target[-1:]

# Train the model and make a local prediction
model = Model()
model.train(training_data_X, training_data_Y)
print "Local Prediction: ", model.predict_local(test_data_X)
print "Actual Label:", test_data_Y
hunch_api = HunchApi()
hunch_api.publish_model(model, "SKLearnExample", "1.0.0")