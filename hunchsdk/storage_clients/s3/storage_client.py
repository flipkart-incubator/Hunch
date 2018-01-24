import boto
import boto.s3.connection
from boto.exception import AWSConnectionError
from retrying import retry
import sys
import math
import StringIO

class StorageClient(object):
    def __init__(self, storage_client_config):
        self.__storage_client_config = storage_client_config
        self.__key = '{model_id}_{model_version}'

    def __model_key(self, model_id, model_version):
        return self.__key.format(model_id=model_id, model_version=model_version.replace('.', '_'))

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def __put_to_s3(self, key, value):
        try:
            conn = boto.connect_s3(aws_access_key_id=self.__storage_client_config["access_key"],
                               aws_secret_access_key=self.__storage_client_config["secret_key"],
                               host=self.__storage_client_config["endpoint"],
                               port=80,
                               is_secure=False,
                               calling_format=boto.s3.connection.OrdinaryCallingFormat())
            bucket = conn.get_bucket(self.__storage_client_config["bucket"])

            bucket.delete_key(key)
            store_key = bucket.new_key(key)

            #Get size of the bytestream
            source_size = sys.getsizeof(value)

            if source_size > self.__storage_client_config["size_limit"]:
                # Ceph throws EntityTooSmall Error when chunk size is less than 5 MB.
                if self.__storage_client_config["chunk_size"] < 5242880:
                    raise Exception("chunk_size can't be smaller than 5MB.")

                chunk_size = self.__storage_client_config["chunk_size"]
                multipart_upload = bucket.initiate_multipart_upload(store_key)
                chunk_count = int(math.ceil(source_size / float(chunk_size)))

                chunks = lambda byte_array, chunk_size: [byte_array[x: x + chunk_size] for x in xrange(0, len(byte_array), chunk_size)]

                chunk_list = chunks(value, chunk_size)

                assert chunk_count == len(chunk_list)

                for idx in range(chunk_count):
                    multipart_upload.upload_part_from_file(StringIO.StringIO(chunk_list[idx]), part_num = idx + 1)

                multipart_upload.complete_upload()
            else:
                store_key.set_contents_from_string(value)
            conn.close()
        except AWSConnectionError as e:
            raise AWSConnectionError("Unable to connect to AWS")
        except Exception as e:
            raise Exception("Exception occured" + str(e))

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def __get_from_s3(self, key):
        try:
            conn = boto.connect_s3(aws_access_key_id=self.__storage_client_config["access_key"],
                               aws_secret_access_key=self.__storage_client_config["secret_key"],
                               host=self.__storage_client_config["endpoint"],
                               port=80,
                               is_secure=False,
                               calling_format=boto.s3.connection.OrdinaryCallingFormat())
            bucket = conn.get_bucket(self.__storage_client_config["bucket"])
            store_key = bucket.get_key(key)
            result = store_key.get_contents_as_string()
            conn.close()
            return result
        except AWSConnectionError as e:
            raise AWSConnectionError("Unable to connect to AWS")
        except Exception as e:
            raise Exception("Exception occured" + str(e))



    def get_model_blob(self, model_id, model_version):
        """
        Returns model blob for the given model id and model version
        Args:
            model_id:
            model_version:

        Returns:

        """
        return self.__get_from_s3(self.__model_key(model_id, model_version))

    def write_model_blob(self, model_blob, model_id, model_version):
        """
        Write model blob with the given model id and model version to Model Repository storage.
        Args:
            model_blob:
            model_id:
            model_version:

        Returns:

        """
        self.__put_to_s3(self.__model_key(model_id, model_version), model_blob)

