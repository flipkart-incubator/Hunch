import os

class StorageClient(object):
    def __init__(self, storage_client_config):
        self.__modelrepo_dir = storage_client_config["modelrepo_dir"]
        self.__model_repo_local_path = self.__modelrepo_dir + '/{model_id}_{model_version}'

    def ___model_repo_local_path(self, model_id, model_version):
        """
        Returns the local path of model Repository for the given model id and model version.
        Args:
            model_id:
            model_version:

        Returns:

        """
        return self.__model_repo_local_path.format(model_id=model_id, model_version=model_version.replace('.', '_'))

    def get_model_blob(self, model_id, model_version):
        """
        Returns model blob for the given model id and model version
        Args:
            model_id:
            model_version:

        Returns:

        """
        model_file_path = self.___model_repo_local_path(model_id=model_id, model_version=model_version)
        if os.path.exists(model_file_path):
            with open(model_file_path, "r") as f:
                model_blob = f.read()
                return model_blob
        else:
            raise Exception("Model ({}, {}) doesn't exist. ".format(model_id, model_version))

    def write_model_blob(self, model_blob, model_id, model_version):
        """
        Write model blob with the given model id and model version to Model Repository storage.
        Args:
            model_blob:
            model_id:
            model_version:

        Returns:

        """
        model_file_path = self.___model_repo_local_path(model_id=model_id, model_version=model_version)
        if os.path.exists(model_file_path):
            raise Exception("Model ({}, {}) already exists. ".format(model_id, model_version))
        else:
            with open(model_file_path, "wb") as f:
                f.write(model_blob)

