from typing import *
import io, json, yaml, os, errno, sys, glob, pathlib, math, copy, time, shutil, pickle
from synthergent.base.util.language import as_list, is_list_like, format_exception_msg, remove_values
from synthergent.base.util.string import StringUtil


class FileSystemUtil:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')

    @classmethod
    def exists(cls, path: str) -> bool:
        return pathlib.Path(path).exists()

    @classmethod
    def dir_exists(cls, path: str) -> bool:
        try:
            path: str = cls.expand_dir(path)
            return pathlib.Path(path).is_dir()
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                return False
            raise e

    @classmethod
    def dirs_exist(cls, paths: List[str], ignore_files: bool = True) -> bool:
        for path in paths:
            if ignore_files and cls.file_exists(path):
                continue
            if not cls.dir_exists(path):
                return False
        return True

    @classmethod
    def is_path_valid_dir(cls, path: str) -> bool:
        path: str = cls.expand_dir(path)
        path: str = StringUtil.assert_not_empty_and_strip(
            path,
            error_message=f'Following path is not a valid local directory: "{path}"'
        )
        return path.endswith(os.path.sep) or cls.dir_exists(path)

    @classmethod
    def file_exists(cls, path: str) -> bool:
        try:
            path: str = cls.expand_dir(path)
            return pathlib.Path(path).is_file()
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                return False
            raise e

    @classmethod
    def check_file_exists(cls, path: str):
        if cls.file_exists(path) is False:
            raise FileNotFoundError(f'Could not find file at location "{path}"')

    @classmethod
    def check_dir_exists(cls, path: str):
        if cls.dir_exists(path) is False:
            raise FileNotFoundError(f'Could not find dir at location "{path}"')

    @classmethod
    def files_exist(cls, paths: List[str], ignore_dirs: bool = True) -> bool:
        for path in paths:
            if ignore_dirs and cls.dir_exists(path):
                continue
            if not cls.file_exists(path):
                return False
        return True

    @classmethod
    def get_dir(cls, path: str) -> str:
        """
        Returns the directory of the path. If the path is an existing dir, returns the input.
        :param path: input file or directory path.
        :return: The dir of the passed path. Always ends in '/'.
        """
        path: str = StringUtil.assert_not_empty_and_strip(path)
        path: str = cls.expand_dir(path)
        if not cls.dir_exists(path):  ## Works for both /home/seldon and /home/seldon/
            path: str = os.path.dirname(path)
        return cls.construct_nested_dir_path(path)

    @classmethod
    def mkdir_if_does_not_exist(cls, path: str, *, raise_error: bool = False) -> bool:
        try:
            path: str = cls.expand_dir(path)
            dir_path: str = cls.get_dir(path)
            if not cls.is_writable(dir_path):
                raise OSError(f'Insufficient permissions to create directory at path "{path}"')
            os.makedirs(dir_path, exist_ok=True)
            if not cls.dir_exists(dir_path):
                raise OSError(f'Could not create directory at path "{path}"')
            return True
        except Exception as e:
            if raise_error:
                raise e
            return False

    @classmethod
    def expand_dir(cls, path: Union[str, pathlib.Path]) -> str:
        is_dir: bool = False
        if isinstance(path, pathlib.Path):
            path: str = str(path)
        if pathlib.Path(path).is_dir() or path.endswith(os.path.sep):
            is_dir: bool = True
        path: str = str(path)
        if path.startswith('~'):
            path: str = os.path.expanduser(path)
        path: str = os.path.abspath(path)
        if is_dir:
            path: str = path if path.endswith(os.path.sep) else path + os.path.sep
        return path

    @classmethod
    def is_writable(cls, path: str) -> bool:
        """
        Checkes whether the current user has sufficient permissions to write files in the passed directory.
        Backs off to checking parent files until it hits the root (this handles cases where the path may not exist yet).
        Ref: modified from https://stackoverflow.com/a/34102855
        :param path: path to check directory. If file path is passed, will check in that file's directory.
        :return: True if the current user has write permissions.
        """
        ## Parent directory of the passed path.
        path: str = cls.expand_dir(path)
        dir: str = cls.get_dir(path)
        if cls.dir_exists(dir):
            return os.access(dir, os.W_OK)
        dir_parents: Sequence = pathlib.Path(dir).parents
        for i in range(len(dir_parents)):
            if cls.dir_exists(dir_parents[i]):
                return os.access(dir_parents[i], os.W_OK)
        return False

    @classmethod
    def list_files_in_dir(cls, *args, **kwargs) -> List[str]:
        return cls.list(*args, **kwargs)

    @classmethod
    def list(
            cls,
            path: str,
            *,
            file_glob: str = StringUtil.DOUBLE_ASTERISK,
            ignored_files: Union[str, List[str]] = None,
            recursive: bool = False,
            only_files: bool = False,
            only_subdirs: bool = False,
            **kwargs
    ) -> List[str]:
        if ignored_files is None:
            ignored_files = []
        ignored_files: List[str] = as_list(ignored_files)
        if not isinstance(file_glob, str):
            raise ValueError(f'`file_glob` must be a string; found {type(file_glob)} with value {file_glob}')
        if only_files and only_subdirs:
            raise ValueError(f'Cannot set both `only_files` and `only_subdir` to True; at most one must be set.')
        path: str = cls.expand_dir(path)
        fpaths: List[str] = glob.glob(os.path.join(path, file_glob), recursive=recursive)
        file_names_map: Dict[str, str] = {file_path: os.path.basename(file_path) for file_path in fpaths}
        file_names_map = remove_values(file_names_map, ignored_files)
        fpaths: List[str] = sorted(list(file_names_map.keys()))
        if only_files:
            fpaths: List[str] = [file_path for file_path in fpaths if cls.file_exists(file_path)]
        if only_subdirs:
            fpaths: List[str] = [file_path for file_path in fpaths if cls.dir_exists(file_path)]
        return fpaths if len(fpaths) > 0 else []

    @classmethod
    def list_first_file_in_dir(cls, path: str, file_glob=StringUtil.ASTERISK, ignored_files=None) -> Optional[str]:
        path: str = cls.expand_dir(path)
        file_paths: List[str] = cls.list_files_in_dir(path, file_glob=file_glob, ignored_files=ignored_files)
        return file_paths[0] if len(file_paths) > 0 else None

    @classmethod
    def list_only_file_in_dir(cls, path: str, file_glob=StringUtil.ASTERISK, ignored_files=None) -> Optional[str]:
        path: str = cls.expand_dir(path)
        if cls.file_exists(path):
            return path  ## Is actually a file
        file_paths: List[str] = cls.list_files_in_dir(path, file_glob=file_glob, ignored_files=ignored_files)
        if len(file_paths) == 0:
            return None
        if len(file_paths) > 1:
            raise FileNotFoundError(f'Multiple matching files are present in the directory')
        return file_paths[0]

    @classmethod
    def get_file_size(
            cls,
            path: Union[List[str], str],
            unit: Optional[str] = None,
            decimals: int = 3,
    ) -> Union[float, str]:
        fpaths: List[str] = as_list(path)
        size_in_bytes: int = int(sum([pathlib.Path(fpath).stat().st_size for fpath in fpaths]))
        if unit is not None:
            return StringUtil.convert_size_from_bytes(size_in_bytes, unit=unit, decimals=decimals)
        return StringUtil.readable_bytes(size_in_bytes, decimals=decimals)

    @classmethod
    def get_time_last_modified(cls, path: str, decimals=3):
        path = StringUtil.assert_not_empty_and_strip(path)
        path: str = cls.expand_dir(path)
        assert cls.exists(path), f'Path {path} does not exist.'
        return round(os.path.getmtime(path), decimals)

    @classmethod
    def get_last_modified_time(cls, path: str):
        path: str = cls.expand_dir(path)
        assert cls.exists(path), f'Path {path} does not exist.'
        return os.path.getmtime(path)

    @classmethod
    def get_seconds_since_last_modified(cls, path: str, decimals=3):
        path: str = cls.expand_dir(path)
        return round(time.time() - cls.get_last_modified_time(path), decimals)

    @classmethod
    def read(
            cls,
            path: str,
            *,
            concat: bool = False,
            concat_sep: str = '\n',
            **kwargs,
    ) -> Optional[Union[Dict[str, str], str]]:
        if cls.file_exists(path):
            return cls.get_file_str(path, **kwargs)
        elif cls.dir_exists(path):
            out = {
                fpath: cls.get_file_str(fpath, **kwargs)
                for fpath in cls.list(path, **kwargs)
            }
            if not concat:
                return out
            return concat_sep.join([
                out[fpath]
                for fpath in sorted(list(out.keys()))
            ])
        raise OSError(f'Path "{path}" is neither an existing file or directory.')

    @classmethod
    def get_file_str(
            cls,
            path: str, *,
            encoding: str = 'utf-8',
            errors: str = 'replace',
            raise_error: bool = False,
            **kwargs
    ) -> Optional[str]:
        path: str = cls.expand_dir(path)
        try:
            with io.open(path, 'r', encoding=encoding, errors=errors) as inp:
                file_str = inp.read()
            StringUtil.assert_not_empty(file_str)
            return file_str
        except Exception as e:
            if raise_error:
                raise e
        return None

    @classmethod
    def get_file_bytes(cls, path: str, *, raise_error: bool = False) -> Optional[bytes]:
        path: str = cls.expand_dir(path)
        try:
            with io.open(path, 'rb') as inp:
                file_bytes = inp.read()
            StringUtil.assert_not_empty_bytes(file_bytes)
            return file_bytes
        except Exception as e:
            if raise_error:
                raise e
        return None

    @classmethod
    def get_file_pickle(cls, path: str, *, raise_error: bool = False) -> Optional[Any]:
        path: str = cls.expand_dir(path)
        try:
            with io.open(path, 'rb') as inp:
                data = pickle.load(inp)
                assert data is not None
                return data
        except Exception as e:
            if raise_error:
                raise e
        return None

    @classmethod
    def get_json(cls, path: str, *, raise_error: bool = False):
        path: str = cls.expand_dir(path)
        try:
            with io.open(path, 'r') as inp:
                return json.load(inp)
        except Exception as e:
            if raise_error:
                raise e
            return None

    @classmethod
    def get_yaml(cls, path: str, *, raise_error: bool = False):
        path: str = cls.expand_dir(path)
        try:
            with io.open(path, 'r') as inp:
                return yaml.safe_load(inp)
        except Exception as e:
            if raise_error:
                raise e
            return None

    @classmethod
    def touch_file(
            cls,
            path: str,
            **kwargs,
    ) -> bool:
        return cls.put_file_str(path=path, file_str='', **kwargs)

    @classmethod
    def put_file_str(
            cls,
            path: str,
            file_str: str,
            overwrite: bool = True,
            raise_error: bool = True,
    ) -> bool:
        path: str = cls.expand_dir(path)
        if cls.file_exists(path) and overwrite is False:
            if raise_error:
                raise FileExistsError(f'File already exists at {path}, set overwrite=True to overwrite it.')
            return False
        try:
            with io.open(path, 'w') as out:
                out.write(file_str)
            return True
        except Exception as e:
            if raise_error:
                raise e
            return False

    @classmethod
    def put_file_pickle(
            cls,
            path: str,
            data: Any,
            overwrite: bool = True,
            raise_error: bool = True,
    ) -> bool:
        path: str = cls.expand_dir(path)
        if cls.file_exists(path) and overwrite is False:
            if raise_error:
                raise FileExistsError(f'File already exists at {path}, set overwrite=True to overwrite it.')
            return False
        try:
            with io.open(path, 'wb') as out:
                pickle.dump(data, out)
            return True
        except Exception as e:
            if raise_error:
                raise e
            return False

    @classmethod
    def rm_file(cls, path: str, *, raise_error: bool = True):
        path: str = cls.expand_dir(path)
        if cls.file_exists(path):
            try:
                os.remove(path)
            except Exception as e:
                if raise_error:
                    raise e
                return False

    @classmethod
    def copy_dir(cls, source: str, destination: str, *, mkdir: bool = True, raise_error: bool = True) -> bool:
        """Copies one dir to another dir, potentially overwriting files in the destination dir."""
        source: str = cls.expand_dir(source)
        if not cls.dir_exists(source):
            if not raise_error:
                return False
            raise OSError(f'Could not find source directory at path "{source}"')
        destination: str = cls.expand_dir(destination)
        if not cls.is_path_valid_dir(destination):
            if not raise_error:
                return False
            raise OSError(f'Destination is not a valid directory path: "{destination}"')
        if mkdir:
            if not cls.mkdir_if_does_not_exist(destination, raise_error=False):
                if not raise_error:
                    return False
                raise OSError(f'Cannot create destination directory at path: "{destination}"')
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return True

    @classmethod
    def construct_path_in_dir(cls, path: str, name: str, is_dir: bool, **kwargs) -> str:
        if not path.endswith(os.path.sep):
            path += os.path.sep
        if is_dir is False:
            out: str = cls.construct_file_path_in_dir(path, name, **kwargs)
        else:
            out: str = cls.construct_subdir_path_in_dir(path, name)
        return out

    @classmethod
    def construct_file_path_in_dir(cls, path: str, name: str, file_ending: Optional[str] = None) -> str:
        """
        If the path is a dir, uses the inputs to construct a file path.
        If path is a file, returns the path unchanged.
        :param path: path to dir (or file) on filesystem.
        :param name: name of the file.
        :param file_ending: (optional) a string of the file ending.
        :return: file path string.
        """
        path: str = cls.expand_dir(path)
        if cls.is_path_valid_dir(path):
            file_name: str = StringUtil.assert_not_empty_and_strip(name)
            if file_ending is not None:
                file_name += StringUtil.assert_not_empty_and_strip(file_ending)
            return os.path.join(cls.get_dir(path), file_name)
        else:
            return path

    @classmethod
    def construct_subdir_path_in_dir(cls, path: str, name: str) -> str:
        """
        Uses the inputs to construct a subdir path.
        :param path: path to dir on filesystem.
        :param name: name of the subdir.
        :return: subdir path string.
        """
        path: str = cls.expand_dir(path)
        if not cls.is_path_valid_dir(path):
            raise ValueError(f'Base dir path "{path}" is not a valid directory.')
        name: str = StringUtil.assert_not_empty_and_strip(name)
        path: str = os.path.join(cls.get_dir(path), name)
        if not path.endswith(os.path.sep):
            path += os.path.sep
        return path

    @classmethod
    def construct_nested_dir_path(cls, path: str, *other_paths: Tuple[str]) -> str:
        StringUtil.assert_not_empty(path)
        other_paths = tuple([str(x) for x in other_paths])
        path = os.path.join(path, *other_paths)
        return path if path.endswith(os.path.sep) else path + os.path.sep
