import cloudpickle
import os
import time
import shutil
import logging
import tarfile

class CustomPackageDeployer:
    def __init__(self, custom_package_deployer_config):
        self.working_dir = custom_package_deployer_config["working_dir"]
        self.pkg_install_dir = self.working_dir + '/custom_libs'
        self.logger = logging.getLogger(__name__)

    def extend_path_in_initpy(self, init_py_file_path):
        """
        Extends path to support multiple packages with same namespace.
        :param init_py_file_path:
        :return:
        """
        path_extended = False
        if (os.path.isfile(init_py_file_path)):
            with open(init_py_file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1 and lines[-1] == "__path__ = extend_path(__path__, __name__)":
                   path_extended = True
        if not path_extended:
            lines_to_add = "\nfrom pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)"
            with open(init_py_file_path, 'a') as file:
                file.write(lines_to_add)

    def check_package_version(self, custom_package_version, model_id, model_version):
        """
        Checks whether custom package version is valid or not.
        :param custom_package_version: The version to check
        :param model_id: For logging purpose
        :param model_version: For logging purpose
        :return: InvalidCustomPackageVersionException if the version is not valid else None
        """

        custom_package_version_details = custom_package_version.split(".")
        for single_version in custom_package_version_details:
            if not single_version.isdigit():
                self.logger.error(
                    "Custom Package version is invalid for Model %s . The version is %s" % ((model_id + ":" + model_version), custom_package_version))
                raise Exception(
                    message="Custom Package version is invalid for Model %s . The version is %s" % ((model_id + ":" + model_version), custom_package_version))

    def check_namespace_clashes(self, new_pkg_file_path, custom_package_name, custom_package_version):
        """
        Raises NamespaceClashException in case of any namespace clashes with already installed custom packages.
        :param new_pkg_file_path:
        :param custom_package_name:
        :param custom_package_version:
        :return:
        """
        tar = tarfile.open(new_pkg_file_path)
        tar.extractall(self.working_dir)
        tar.close()

        package_path = self.working_dir + '/' + custom_package_name + '-' + custom_package_version
        all_files_path = set()
        for directory_tuple in os.walk(package_path):
            directory_path = directory_tuple[0]
            if directory_path == package_path or (custom_package_name + '.egg-info') in directory_path:
                continue
            for file_name in directory_tuple[2]:
                if not file_name.endswith('.py') or file_name == '__init__.py':
                    continue
                file_path = directory_path + "/" + file_name
                all_files_path.add(file_path[(len(package_path) + 1):])
            init_file_path = directory_path + "/__init__.py"
            self.extend_path_in_initpy(init_file_path)

        for dct in os.walk(self.pkg_install_dir):
            if dct[0] == self.pkg_install_dir:
                continue
            for file_name in dct[2]:
                if not file_name.endswith('.py') or file_name == '__init__.py':
                    continue
                file_path = dct[0] + "/" + file_name
                file_path = file_path[len(self.pkg_install_dir) + 1:]
                file_path = file_path[file_path.find("/") + 1:]
                if file_path in all_files_path:
                    self.logger.error("Namespace clash while installing %s. %s exists in the same namespace of an already installed custom package" %(custom_package_name, file_name))
                    raise Exception(message="Namespace clash while installing %s. %s exists in the same namespace of an already installed custom package" %(custom_package_name, file_name))

        try:
            os.remove(new_pkg_file_path)
        except OSError as e:
            self.logger.exception("rm file failed with exception " + e)
        tar = tarfile.open(new_pkg_file_path, "w:gz")
        tar.add(package_path, arcname="TarName")
        tar.close()
        try:
            shutil.rmtree(package_path)
        except OSError as e:
            self.logger.exception("rmtree failed with exception " + e)

        return

    def install_custom_package(self, model_blob, model_id, model_version, delete_previous):
        """

        Args:
            model_blob:
            model_id:
            model_version:
            delete_previous:

        Returns:

        """
        model_obj = cloudpickle.loads(model_blob)

        if isinstance(model_obj, dict) and 'custom_package_blob' in model_obj.keys():
            (custom_package_name, custom_package_version, tar_file_content) = self.unpack_pkg(model_obj)
            self.check_package_version(custom_package_version, model_id, model_version)

            self.install_custom_module(custom_package_name, custom_package_version, model_id, model_version, tar_file_content, delete_previous=delete_previous)

    def unpack_pkg(self, model_obj):
        """
        Package details along with content are returned as tuple.
        Args:
            model_obj: Model object received from Model Repository

        Returns:

        """
        if isinstance(model_obj, dict):
            tar_file_content = model_obj['custom_package_blob']
            custom_package_name = model_obj['custom_package_name']
            custom_package_version = model_obj['custom_package_version']
        else:
            raise ValueError('Cannot extract pkg - model is not a dictionary.')

        return (custom_package_name, custom_package_version, tar_file_content)

    def lock(self, lockfile):
        import fcntl
        lockfd = open(lockfile, 'w+')
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lockfd

    def unlock(self, lockfd):
        import fcntl
        fcntl.flock(lockfd, fcntl.LOCK_UN)

    def install_custom_module(self, custom_package_name, custom_package_version, model_id, model_version, tar_file_content, delete_previous=False):
        lockfile = self.working_dir + '/' + custom_package_name + '.lock'
        pkg_file_path = self.working_dir + '/' + custom_package_name + '-' + custom_package_version + '-' + str(time.time()) + '.tgz'

        if not os.path.exists(lockfile):
            fd = open(lockfile, 'w+')
            fd.close()

        lockfd = self.lock(lockfile)

        if not os.path.exists(pkg_file_path):
            with open(pkg_file_path, 'w') as fout:
                fout.write(tar_file_content)

        pkg_install_path = self.pkg_install_dir+'/'+custom_package_name

        if os.path.exists(pkg_install_path) and delete_previous==True:
            import shutil
            shutil.rmtree(pkg_install_path, ignore_errors=True)
        elif os.path.exists(pkg_install_path) and delete_previous==False:
            self.unlock(lockfd)
            raise ValueError("Already installed. Set delete_previous to true to override")

        self.check_namespace_clashes(pkg_file_path, custom_package_name, custom_package_version)

        if not os.path.exists(pkg_install_path):
            import pip
            pip.main(['install', pkg_file_path, '--upgrade', '--target', pkg_install_path, '--no-deps'])

        self.unlock(lockfd)
        return pkg_install_path

    def uninstall_custom_packages(self):
        """
        Uninstalls all the custom packages installed.
        Returns:

        """
        if os.path.isdir(self.pkg_install_dir):
            for file in os.listdir(self.pkg_install_dir):
                file_path = os.path.join(self.pkg_install_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    self.logger.exception("Exception occurred: " + str(e))

        self.logger.debug("Custom packages are uninstalled. ")

    def cleanup_custom_packages(self):
        """
        Uninstalls all the custom packages installed and removes all the temp files created while installing custom packages
        Returns:

        """
        # Remove Tar files in hunchsdk_custom_libs directory
        if os.path.isdir(self.working_dir):
            for file in os.listdir(self.working_dir):
                file_path = os.path.join(self.working_dir, file)
                try:
                    if os.path.isfile(file_path) and (file_path.endswith(".tgz") or file_path.endswith(".lock")):
                        os.unlink(file_path)
                except Exception as e:
                    self.logger.exception("Exception occurred: " + str(e))
        self.uninstall_custom_packages()

    def pythonpath_for_custom_packages(self):
        """
        Python path needed for all the custom packages installed.
        Returns:

        """
        python_paths = []
        if os.path.isdir(self.pkg_install_dir):
            packages = next(os.walk(self.pkg_install_dir))[1]
            for package in packages:
                python_paths.append(self.pkg_install_dir + "/" + package)

        return ",".join(python_paths)