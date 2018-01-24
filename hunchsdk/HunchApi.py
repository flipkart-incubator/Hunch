import os
import yaml
from hunch_publisher import Publisher

class HunchApi:

    def __init__(self, config="client_config.yaml"):
        config_file = config
        env_value = os.getenv("HUNCH_API_CONFIG", None)
        if env_value:
            config_file = env_value

        if os.path.exists(config_file):
            with open(config_file, 'rt') as f:
                client_config = yaml.load(f.read())
                self.publisher = Publisher(client_config)
        else:
            raise Exception("Given config file doesn't exist")


    def publish_model(self, model_instance, model_id, model_version, path_to_setup_py = None, custom_package_name = None):
        """
        Publishes the given model to Model Blob Storage. It uses Cloudpickle (https://github.com/cloudpipe/cloudpickle) based serialization.
        Args:
            model_instance: Model object which has predict method implemented
            model_id: Model ID
            model_version: Model Version. Model Version is not auto generated. Model Repository integration is needed for Auto version increment.
            path_to_setup_py: Path to setup.py of the custom package on which model is dependent
            custom_package_name: Name of the custom package on which the model is dependent.

        Returns:

        """
        self.publisher.publish_model(model_instance, model_id, model_version, path_to_setup_py, custom_package_name)

    def publish_asm_model(self, path_to_prediction_module, path_to_model_resources_dir, model_id, model_version, path_to_setup_py=None, custom_package_name=None):
        """
        Publishes the given model to Model Blob Storage. Models which are not pure Python objects ( Eg: Models built using frameworks like tensorflow, fasttext, crf etc.) are not serializable using Cloudpickle.

        Args:
            path_to_prediction_module: Prediction module is where you write the model class and implement load_model
            path_to_model_resources_dir: Model resources directory. All the files needed to load a model should be there in this directory.
            model_id: Model ID
            model_version: Model Version. Model Version is not auto generated. Model Repository integration is needed for Auto version increment.
            path_to_setup_py: Path to setup.py of the custom package on which model is dependent
            custom_package_name: Name of the custom package on which the model is dependent.
        Returns:

        """
        self.publisher.publish_asm_model(path_to_prediction_module, path_to_model_resources_dir, model_id, model_version, path_to_setup_py, custom_package_name)