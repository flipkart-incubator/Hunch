import importlib
import os
import sys
from io import BytesIO
from tarfile import TarFile
import shutil
import cloudpickle
from custom_package_deployer import CustomPackageDeployer
from hunchsdk.storage_clients.get_storage_client import get_storage_client

class ModelLoader:
    def __init__(self, hunch_server_config):
        self.__working_dir = hunch_server_config["model_loader"]["working_dir"]
        self.__prediction_module_dir = self.__working_dir + '/prediction_modules'
        self.__prediction_module_path = self.__prediction_module_dir + '/{model_id}_{model_version}/'
        self.__model_resource_dir = self.__working_dir + '/model_resources'
        self.__model_resource_path = self.__model_resource_dir + '/{model_id}_{model_version}/'
        self.__prediction_module = '{model_id}_{model_version}'
        self.__custom_package_deployer = CustomPackageDeployer(hunch_server_config["custom_package_deployer"])
        self.__model_storage_backend = hunch_server_config["model_storage"]["backend"]
        self.__model_storage_client_config = hunch_server_config["model_storage"][self.__model_storage_backend]


    def deserialize_model(self, blob, model_id, model_version):
        """
        Deserializes the given blob to Model object which can be used for predictions
        :param blob:
        :param model_id:
        :param model_version:
        :return:
        """
        model_obj = cloudpickle.loads(blob)

        if not isinstance(model_obj, dict):  # Is a plain cloud-pickled model
            return model_obj

        if isinstance(model_obj, dict) and 'custom_package_blob' in model_obj.keys():
            self.__custom_package_deployer.install_custom_package(blob, model_id, model_version, delete_previous = True)

        if 'serialization_mechanism' in model_obj and model_obj['serialization_mechanism'] == 'asm':  # Is an ASM model
            self.__extract_model_resources(model_obj, model_id, model_version)
            self.__extract_prediction_module(model_obj, model_id, model_version)
            return self.__deserialize_asm_model(model_id, model_version)

        # tar_file_content = model_obj['custom_package_blob']
        # custom_package_name = model_obj['custom_package_name']
        # custom_package_version = model_obj['custom_package_version']
        return cloudpickle.loads(model_obj['model_blob'])  # Is a cloud-pickled model with custom code

    def get_model(self, model_id, model_version):
        """
        Downloads the model from Blob storage. Blob storage client implmentation is chosen based on the configuration
        Args:
            model_id:
            model_version:

        Returns:

        """
        storage_client = get_storage_client(self.__model_storage_backend, self.__model_storage_client_config)
        return storage_client.get_model_blob(model_id, model_version)

    def get_models_from_list(self, models_list):
        """
        Returns deseriliazed objects of the given models.
        :param models_list:
        :return:
        """
        models = {}
        for model_tuple in models_list:
            model_blob = self.get_model(model_tuple[0], model_tuple[1])
            models[(model_tuple[0], model_tuple[1])] =  self.deserialize_model(model_blob, model_tuple[0], model_tuple[1])
        return models

    def __deserialize_asm_model(self, model_id, model_version):
        this_model_resources_dir = self.___model_resource_path(model_id, model_version)
        this_prediction_module_path = self.___prediction_module_path(model_id, model_version)
        sys.path.append(this_prediction_module_path)
        prediction_module = importlib.import_module(self.__prediction_module.format(model_id=model_id, model_version=model_version.replace('.', '_')))
        load_model = getattr(prediction_module, 'load_model')
        model_instance = load_model(this_model_resources_dir)
        return model_instance

    def __extract_prediction_module(self, model_obj, model_id, model_version):
        prediction_module_path = self.___prediction_module_path(model_id, model_version)
        if not os.path.exists(prediction_module_path):
            os.makedirs(prediction_module_path)

        prediction_module_tar_contents = model_obj['model_predict_module']
        with TarFile.open(fileobj=BytesIO(prediction_module_tar_contents), mode='r:bz2') as tar:
            tar.extractall(prediction_module_path)
        os.rename(prediction_module_path +'/model.py', prediction_module_path +'/' + self.__prediction_module.format(model_id=model_id, model_version=model_version.replace('.', '_')) + '.py')

    def __extract_model_resources(self, model_obj, model_id, model_version):
        model_resource_path = self.___model_resource_path(model_id, model_version)
        if not os.path.exists(model_resource_path):
            os.makedirs(model_resource_path)

        model_resources_tar_contents = model_obj['modeldir_blob']
        with TarFile.open(fileobj=BytesIO(model_resources_tar_contents), mode='r:bz2') as tar:
            tar.extractall(model_resource_path)

    def ___model_resource_path(self, model_id, model_version):
        return self.__model_resource_path.format(model_id=model_id, model_version=model_version)

    def ___prediction_module_path(self, model_id, model_version):
        return self.__prediction_module_path.format(model_id=model_id, model_version=model_version)

    def clean_up(self):
        """
        Cleans up all model resources and prediction modules.
        Returns:

        """
        if os.path.exists(self.__model_resource_dir):
            shutil.rmtree(self.__model_resource_dir)
        if os.path.exists(self.__prediction_module_dir):
            shutil.rmtree(self.__prediction_module_dir)


