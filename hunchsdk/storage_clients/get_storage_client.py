import importlib

def get_storage_client(model_storage_backend, model_storage_client_config):
    """
    Returns the storage client instance created with given storage backend and storage config
    Args:
        model_storage_backend: Blob storage backend to be used. For ex: local_fs,s3, azure_blob_storage etc.
        model_storage_client_config: Configuration required to connect to and load from Blob Storage. For ex: S3 Configuration parameters like endpoint, bucket, secret key, access key etc.

    Returns: storage client instance which allows to write model blobs and load model blobs.

    """
    package = "hunchsdk.storage_clients." + model_storage_backend
    storage_client_module = importlib.import_module(".storage_client", package)
    StorageClient = getattr(storage_client_module, 'StorageClient')
    return StorageClient(model_storage_client_config)
