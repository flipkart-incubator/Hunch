import os
from hunchsdk.HunchApi import HunchApi

os.environ["HUNCH_API_CONFIG"]="../config/hunch_api_config.yaml"


dir_path = os.path.dirname(os.path.realpath(__file__))
print dir_path
hunch_api = HunchApi()
hunch_api.publish_asm_model(path_to_prediction_module=dir_path + '/resources/prediction_modules/tf/mnist_canned_estimator/iris_estimator_predict.py',
            path_to_model_resources_dir=dir_path + '/resources/model_resources/tf/mnist_canned_estimator',
            model_id="TensorflowMnistExample",
            model_version="1.0.0")

