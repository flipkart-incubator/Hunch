import importlib
import inspect
import os
import platform
import stat
import sys
from io import BytesIO
from tarfile import TarFile
import cloudpickle
from pkg_resources import find_distributions, DistributionNotFound
from setuptools import sandbox
from hunchsdk.storage_clients.get_storage_client import get_storage_client
class Publisher:
    def __init__(self, hunch_client_config):
        self.__model_storage_backend = hunch_client_config["model_storage"]["backend"]
        self.__model_storage_client_config = hunch_client_config["model_storage"][self.__model_storage_backend]

    def write_model_to_blob_storage(self, model_blob, model_id, model_version):
        storage_client = get_storage_client(self.__model_storage_backend, self.__model_storage_client_config)
        storage_client.write_model_blob(model_blob, model_id, model_version)

    def publish_model(self, model_instance, model_id, model_version, path_to_setup_py = None, custom_package_name = None):
        """

        Args:
            model_instance:
            path_to_setup_py:
            custom_package_name:

        Returns:

        """
        model_blob, model_meta = self._create_model_blob(None, None, path_to_setup_py, custom_package_name, cloudpickle.dumps(model_instance))
        self.write_model_to_blob_storage(cloudpickle.dumps(model_blob), model_id, model_version)


    def publish_asm_model(self, path_to_prediction_module, path_to_model_resources_dir, model_id, model_version, path_to_setup_py=None, custom_package_name=None):
        self._prediction_module_guardrails_breach(path_to_prediction_module, path_to_model_resources_dir, path_to_setup_py)
        model_blob, model_meta = self._create_model_blob(path_to_prediction_module, path_to_model_resources_dir, path_to_setup_py, custom_package_name)
        self.write_model_to_blob_storage(cloudpickle.dumps(model_blob), model_id, model_version)

    def create_package_from_setup_py(self, path_to_setup_py, custom_package_name):
        """
        Creates a package from Setup.py specified in the the given path
        :param path_to_setup_py:
        :return: Package Information: Contains package name, version and path
        """
        if os.path.isfile(path_to_setup_py):
            path_to_setup_py = os.path.abspath(path_to_setup_py)
            setup_py_directory = os.path.dirname(path_to_setup_py)
        elif os.path.isdir(path_to_setup_py):
            setup_py_directory = os.path.abspath(path_to_setup_py)
            path_to_setup_py = setup_py_directory + '/setup.py'
        else:
            raise Exception("Given path/file: " + path_to_setup_py + " doesn't exist")

        if not os.path.exists(path_to_setup_py):
            raise Exception("setup.py doesn't exist in the given path: " + path_to_setup_py)

        sandbox.run_setup(path_to_setup_py, ['sdist'])
        eggs = find_distributions(setup_py_directory)

        version = None
        for egg in eggs:
            if custom_package_name in egg.egg_name():
                version = egg.version

        if not version:
            raise Exception("{package_name} not found".format(package_name=custom_package_name))

        package_info = dict()
        package_info['path'] = setup_py_directory + "/dist/" + custom_package_name + "-" + version + ".tar.gz"
        package_info['version'] = version
        package_info['name'] = custom_package_name
        return package_info

    def create_model_blob_with_custom_setup(self, model_blob, custom_package_name, custom_package_version,
                                            custom_package_path):
        return cloudpickle.dumps(
            self._create_model_blob_details_with_custom_setup(
                model_blob, custom_package_name, custom_package_version, custom_package_path))

    def _create_model_blob_details_with_custom_setup(self, model_blob, custom_package_name, custom_package_version,
                                                     custom_package_path):
        if not os.path.exists(custom_package_path) or not custom_package_path.endswith('.tar.gz'):
            raise Exception("Tar file doesn't exit at: " + custom_package_path)

        with open(custom_package_path) as fp:
            custom_package_blob = fp.read()

        model_blob_with_custom_code = {
            'custom_package_version': custom_package_version,
            'custom_package_name': custom_package_name,
            'custom_package_blob': custom_package_blob
        }

        if model_blob is not None:
            model_blob_with_custom_code['model_blob'] = model_blob

        return model_blob_with_custom_code

    def _prediction_module_guardrails_breach(self, prediciton_module_file_path, saved_model_path, path_to_setup_py):
        """
        This method checks if the prediction module, breaches any of the following guardrails.
        1. Prediction module implements a load_model function outside of the Model class, to instantiate and load the model.
        2. load_model should just take one and only argument, named model_dir.
        3. model_dir should be a directory, not a file.
        4. load_model should be able to load the model using the saved_model_path.
        5. predict method should be implemented in the model class.

        :param saved_model_path:
        :param prediciton_module_file_path
        :return: False, if there are no breaches. Raises exception otherwise.
        """
        if path_to_setup_py:
            setup_py_directory = None
            if os.path.isfile(path_to_setup_py):
                path_to_setup_py = os.path.abspath(path_to_setup_py)
                setup_py_directory = os.path.dirname(path_to_setup_py)
            elif os.path.isdir(path_to_setup_py):
                setup_py_directory = os.path.abspath(path_to_setup_py)
                path_to_setup_py = setup_py_directory + '/setup.py'
            sys.path.append(setup_py_directory)

        basepath, module_name = os.path.split(prediciton_module_file_path)
        extension = module_name[-3:]
        if extension != '.py':
            raise TypeError('Please provide the full path to python hunch prediction file')
        module_name = module_name[:-3]
        sys.path.insert(0, basepath)
        prediction_module = importlib.import_module(module_name)
        load_model = getattr(prediction_module, 'load_model', None)
        if not load_model:
            raise NotImplementedError('Load method not implemented in the prediction module.')

        arguments = None
        try:
            arguments = inspect.getargspec(load_model).args
        except TypeError:
            raise NotImplementedError("Implement Load function correctly. Refer to docs.")

        if len(arguments) != 1:
            raise AttributeError('load_model method should have one and only one argument: model_dir')

        if not os.path.isdir(saved_model_path):
            raise RuntimeError('{!r} is not a directory'.format(saved_model_path))

        try:
            model_instance = load_model(saved_model_path)
        except Exception as e:
            raise RuntimeError('Local load not possible with the model_dir that was provided and load_model that was implemented')

        if not callable(getattr(model_instance, 'predict')):
            raise NotImplementedError('predict not implemented in the model class.')

        return False

    def _create_model_blob(self, path_to_prediction_module, path_to_model_resources_dir, path_to_setup_py=None, custom_package_name=None, model_dump=None):
        """
        Creates the blob to be stored for the alternate serialization model.
        :param self:
        :param path_to_prediction_module:
        :param path_to_model_resources_dir:
        :param path_to_setup_py:
        :param custom_package_name:
        :return:
        """
        model_blob = {}
        model_meta_data = {}
        if path_to_setup_py is not None:
            package_info = self.create_package_from_setup_py(path_to_setup_py, custom_package_name)
            custom_package_blob = self._create_model_blob_details_with_custom_setup(None, package_info['name'],
                                                                                    package_info['version'],
                                                                                    package_info['path'])
            model_blob.update(custom_package_blob)

        if path_to_prediction_module is not None:
            prediction_blob = self._get_prediction_module_tar_byte_buffer(path_to_prediction_module)
            resources_blob = self._get_model_resources_tar_byte_buffer(path_to_model_resources_dir)
            model_blob['model_predict_module'] = prediction_blob.getvalue()
            model_blob['modeldir_blob'] = resources_blob.getvalue()
            model_blob['serialization_mechanism'] = 'asm'
        else:
            model_blob['model_blob'] = model_dump

        model_blob['platform_details'] = {
            'architecture': platform.machine(),
            'os_name': platform.system(),
            'processor': platform.processor(),
            'os_version': platform.dist()
        }
        model_blob['python_details'] = {
            'python_version': platform.python_version(),
            'python_compiler': platform.python_compiler(),
            'python_implementation': platform.python_implementation()
        }
        model_blob['gcc_details'] = {
            'gcc_version': platform.python_compiler().split(' ')[1],
            'glibc_version': platform.libc_ver()[0]
        }
        model_blob['specification_version'] = '1.0'
        model_meta_data['python_version'] = model_blob['python_details']['python_version']
        model_meta_data['gcc_version'] = model_blob['gcc_details']['gcc_version']
        model_meta_data['os_name'] = model_blob['platform_details']['os_name']
        model_meta_data['os_flavour'] = model_blob['platform_details']['os_version'][0]
        model_meta_data['os_version'] = model_blob['platform_details']['os_version'][1]
        return model_blob, model_meta_data

    def _get_prediction_module_tar_byte_buffer(self, path_to_prediction_module):
        """
        Returns a Byte buffer of a tar file containing the prediction module. The tar file is compressed using bz2
        :param path_to_prediction_module: Path to prediction module file
        :return: Byte buffer with the tar data
        """
        prediction_module_stat = os.stat(path_to_prediction_module)
        if stat.S_ISDIR(prediction_module_stat.st_mode):
            raise Exception("Expected a file but got a directory for arg 'path_to_prediction_module' = '{}'".format(
                path_to_prediction_module))

        file_out = BytesIO()
        with TarFile.open(mode="w:bz2", fileobj=file_out) as tar:
            tar.add(name=path_to_prediction_module, arcname='model.py')

        return file_out

    def _get_model_resources_tar_byte_buffer(self, path_to_model_resources_dir):
        """
        Returns a byte buffer of a tar file containing the model resources. The tar file is compressed using bz2.
        The topmost folder is named 'model_resource'
        :param path_to_model_resources_dir:  Path to resources directory
        :return: Byte buffer with the tar data
        """
        model_resources_stat = os.stat(path_to_model_resources_dir)
        if not stat.S_ISDIR(model_resources_stat.st_mode):
            raise Exception(
                "Expected a directory for arg 'path_to_model_resources_dir' = {}".format(path_to_model_resources_dir))

        file_out = BytesIO()
        with TarFile.open(mode="w:bz2", fileobj=file_out) as tar:
            tar.add(name=path_to_model_resources_dir, recursive=True, arcname='')

        return file_out