from typing import *
import pathlib, io, os, requests, tempfile
from synthergent.base.util import Parameters, is_null, auto, StringUtil, Log, FileSystemUtil, safe_validate_arguments, \
    optional_dependency, get_default
from synthergent.base.util.aws import S3Util
from synthergent.base.constants import FileFormat, FileContents, Storage, REMOTE_STORAGES, FILE_ENDING_TO_FILE_FORMAT_MAP, Alias, \
    MLType, MLTypeSchema
from pydantic import validator, root_validator, constr, FilePath

FileMetadata = "FileMetadata"


class FileMetadata(Parameters):
    name: Optional[constr(min_length=1, max_length=63, strip_whitespace=True)]
    path: Union[constr(min_length=1, max_length=1023), Any]
    storage: Optional[Storage]
    format: Optional[FileFormat]
    contents: Optional[FileContents]
    file_glob: Optional[str]
    data_schema: Optional[MLTypeSchema]

    @classmethod
    def of(cls, path: Union[io.IOBase, FileMetadata, Dict, str], **kwargs) -> 'FileMetadata':
        if isinstance(path, FileMetadata):
            path: Dict = path.dict(exclude=None)
        elif isinstance(path, (str, pathlib.Path)):
            path: Dict = dict(path=str(path))
        elif isinstance(path, io.IOBase):
            path: Dict = dict(path=path)
        assert isinstance(path, dict)
        path: Dict = {
            **path,
            **kwargs
        }
        return FileMetadata(**path)

    @root_validator(pre=True)
    def set_params(cls, params: Dict):
        Alias.set_format(params)
        if isinstance(params['path'], pathlib.Path):
            params['path']: str = str(params['path'])
        if isinstance(params['path'], str) and params['path'].startswith('~'):
            params['path']: str = FileSystemUtil.expand_dir(params['path'])

        if 'storage' not in params:
            params['storage']: Storage = cls.detect_storage(params['path'])
        if params['storage'] is Storage.STREAM:
            raise ValueError(f'Storage cannot be a stream.')
        elif params['storage'] is Storage.LOCAL_FILE_SYSTEM:
            params['path']: str = FileSystemUtil.expand_dir(params['path'])

        if 'format' not in params:
            format: Optional[FileFormat] = cls.detect_file_format(params['path'], raise_error=False)
            if format is not None:
                params['format'] = format
        return params

    def is_remote_storage(self, remote_storages: Tuple[Storage, ...] = tuple(REMOTE_STORAGES)) -> bool:
        return self.storage in remote_storages

    @classmethod
    @safe_validate_arguments
    def detect_storage(cls, path: Union[io.IOBase, constr(min_length=1, max_length=1023)]) -> Optional[Storage]:
        if isinstance(path, io.IOBase) and hasattr(path, 'read'):
            return Storage.STREAM
        elif isinstance(path, str):
            if path.startswith(StringUtil.HTTP_PREFIX) or path.startswith(StringUtil.HTTPS_PREFIX):
                return Storage.URL
            if S3Util.is_valid_s3_path(path):
                return Storage.S3
            return Storage.LOCAL_FILE_SYSTEM
        return None

    @classmethod
    @safe_validate_arguments
    def detect_file_ending(
            cls,
            file_path: constr(min_length=1, max_length=1023),
            raise_error: bool = True,
    ) -> Optional[str]:
        if FileSystemUtil.is_path_valid_dir(file_path) or S3Util.is_path_valid_s3_dir(file_path):
            if raise_error:
                raise ValueError(f'Cannot detect file ending of directory: {file_path}')
            return None
        ## Works for both local and S3 paths:
        file_ending: List = pathlib.Path(str(file_path)).suffixes
        if len(file_ending) == 0:
            return None
        return ''.join(file_ending)

    @classmethod
    def path_exists(cls, path: Any) -> bool:
        if not isinstance(path, (str, pathlib.Path)):
            return False
        path: str = str(path)
        storage: Storage = cls.detect_storage(path)
        if storage is Storage.LOCAL_FILE_SYSTEM:
            return FileSystemUtil.exists(path)
        elif storage is Storage.S3:
            return S3Util.s3_object_exists(path)
        elif storage is Storage.URL:
            return str(requests.head(path).status_code).startswith('2')
        raise NotImplementedError(f'Cannot determine whether following path exists on {storage}: "{path}"')

    def exists(self) -> bool:
        return self.path_exists(path=self.path)

    @classmethod
    @safe_validate_arguments
    def detect_file_format(
            cls,
            file_path: constr(min_length=1, max_length=1023),
            raise_error: bool = True,
    ) -> Optional[FileFormat]:
        if FileSystemUtil.is_path_valid_dir(file_path) or S3Util.is_path_valid_s3_dir(file_path):
            if raise_error:
                raise ValueError(f'Cannot detect file format of directory: {file_path}')
            return None
        fpath_stripped: str = str(file_path).rstrip()
        matched_file_endings_longest_first: List[Tuple[str, FileFormat]] = sorted(
            [
                (file_ending, file_format)
                for file_ending, file_format in FILE_ENDING_TO_FILE_FORMAT_MAP.items()
                if fpath_stripped.endswith(file_ending)
            ],
            key=lambda x: len(x[0]),
            reverse=True,
        )
        if len(matched_file_endings_longest_first) == 0:
            if raise_error:
                raise ValueError(f'No matching file format found for file with path: "{file_path}"')
            return None
        return matched_file_endings_longest_first[0][1]

    @safe_validate_arguments
    def open(self, file_name: Optional[str] = None, mode: Optional[str] = None, tmpdir: Optional[str] = None):
        if self.is_path_valid_dir() and file_name is None:
            raise ValueError(
                f'When the path is a directory, you must pass `file_name` '
                f'to {self.class_name}.open(...)'
            )
        elif not self.is_path_valid_dir() and file_name is not None:
            raise ValueError(
                f'When the file metadata path is a file, you must not pass `file_name` '
                f'to {self.class_name}.open(...)'
            )
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            if self.is_path_valid_dir():
                assert file_name is not None
                local_file: FileMetadata = self.file_in_dir(file_name, return_metadata=True)
            else:
                assert file_name is None
                local_file: FileMetadata = self
        elif self.is_remote_storage():
            if self.is_path_valid_dir() and file_name is not None:
                remote_file: FileMetadata = self.file_in_dir(file_name, return_metadata=True)
                if tmpdir is None:
                    tmpdir: str = tempfile.TemporaryDirectory().name
                temp_local_dir: FileMetadata = FileMetadata.of(tmpdir).mkdir(return_metadata=True)
                temp_local_file: FileMetadata = temp_local_dir.file_in_dir(file_name, return_metadata=True)
                if remote_file.storage is Storage.S3:
                    if not S3Util.copy_s3_file_to_local(
                            source_s3_path=remote_file.path,
                            destination_local_path=temp_local_file.path,
                    ):
                        raise OSError(f'Cannot download file from "{remote_file.path}" to "{temp_local_file.path}"')
                else:
                    raise NotImplementedError(f'Can only load from S3, not {self.storage}')
                if not temp_local_file.exists():
                    raise OSError(f'No such file on {temp_local_file.storage}: "{temp_local_file}"')
                local_file: FileMetadata = temp_local_file
            else:
                raise NotImplementedError(f'Cannot yet open a folder from remote location: "{self.path}"')
        else:
            raise NotImplementedError(f'Cannot open storage: {self.storage}')
        return io.open(local_file.path, mode=mode)

    def get_dir(self, return_metadata: bool = False) -> Union[FileMetadata, str]:
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            dir_path: str = FileSystemUtil.get_dir(self.path)
        elif self.storage is Storage.S3:
            dir_path: str = S3Util.get_s3_dir(self.path)
        else:
            raise NotImplementedError(f'Cannot get dir for path on {self.storage} storage.')
        if return_metadata:
            return self.update_params(path=dir_path)
        return dir_path

    def file_in_dir(
            self,
            path: str,
            return_metadata: bool = False,
            touch: bool = False,
            **kwargs,
    ) -> Union[FileMetadata, str]:
        file_in_dir: str = self.path_in_dir(path, is_dir=False, **kwargs)
        if touch:
            if self.storage is Storage.LOCAL_FILE_SYSTEM:
                FileSystemUtil.touch_file(file_in_dir)
            elif self.storage is Storage.S3:
                S3Util.touch_s3_object(file_in_dir)
            else:
                raise ValueError(f'Cannot touch file on {self.storage} storage.')
        if return_metadata:
            return self.update_params(path=file_in_dir)
        return file_in_dir

    def __truediv__(self, subdir_name: str) -> FileMetadata:
        assert isinstance(subdir_name, str)
        return self.subdir_in_dir(
            subdir_name,
            return_metadata=True,
        )

    def subdir_in_dir(
            self,
            path: Optional[str],
            *,
            mkdir: bool = True,
            return_dir_on_none: bool = False,
            raise_error: bool = True,
            return_metadata: bool = False,
            **kwargs,
    ) -> Union[FileMetadata, str]:
        if path is None and return_dir_on_none:
            if return_metadata:
                return self.copy()
            return self.path
        subdir_path: str = self.path_in_dir(path, is_dir=True, **kwargs)
        if mkdir:
            FileMetadata(
                path=subdir_path,
                **self.dict(exclude={'path'})
            ).mkdir(raise_error=raise_error)
        if return_metadata:
            return self.update_params(path=subdir_path)
        return subdir_path

    def path_in_dir(self, path: str, is_dir: bool, **kwargs) -> str:
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            return FileSystemUtil.construct_path_in_dir(self.path, path, is_dir=is_dir, **kwargs)
        elif self.storage is Storage.S3:
            if not S3Util.is_valid_s3_path(self.path):
                raise ValueError(
                    f'Cannot create path of file/subdir with name "{path}" in directory; '
                    f'base directory path "{self.path}" is invalid.'
                )
            return S3Util.construct_path_in_s3_dir(self.path, path, is_dir=is_dir, **kwargs)
        raise NotImplementedError(f'Cannot create path {path} in dir "{self.path}" for storage: {self.storage}')

    def mkdir(self, raise_error: bool = True, return_metadata: bool = False) -> Union[FileMetadata, str]:
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            path: str = self.path
            if not path.endswith(os.path.sep):
                path += os.path.sep
            FileSystemUtil.mkdir_if_does_not_exist(path, raise_error=raise_error)
            if return_metadata:
                return self
            return path
        elif self.storage is Storage.URL:
            raise ValueError(f'Cannot create a directory at URL: {self.path}')
        elif self.storage is Storage.STREAM:
            raise ValueError(f'Cannot create a directory for a stream.')
        elif self.storage is Storage.S3:
            path: str = self.path
            if not path.endswith(StringUtil.SLASH):
                path += StringUtil.SLASH
            if return_metadata:
                return self
            return path  ## Do nothing, S3 dirs do not need to be created.
        raise NotImplementedError(f'Unsupported storage: {self.storage}')

    def mksubdir(self, subdir_name: str, raise_error: bool = True) -> bool:
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            if not FileSystemUtil.is_path_valid_dir(self.path):
                raise ValueError(
                    f'Cannot create subdirectory with name "{subdir_name}" in directory; '
                    f'directory path "{self.path}" is invalid.'
                )
            return FileSystemUtil.mkdir_if_does_not_exist(
                self.subdir_in_dir(subdir_name),
                raise_error=raise_error,
            )
        elif self.storage is Storage.URL:
            raise ValueError(f'Cannot create a directory at URL: {self.path}')
        elif self.storage is Storage.STREAM:
            raise ValueError(f'Cannot create a directory for a stream.')
        elif self.storage is Storage.S3:
            return True  ## Do nothing, S3 dirs do not need to be created.
        raise NotImplementedError(f'Unsupported storage: {self.storage}')

    def is_path_valid_dir(self) -> bool:
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            return FileSystemUtil.is_path_valid_dir(self.path)
        elif self.storage is Storage.S3:
            return S3Util.is_path_valid_s3_dir(self.path)
        elif self.storage is Storage.URL:
            return self.path.endswith(StringUtil.SLASH)
        return False

    def list(self, **kwargs) -> List[str]:
        if not self.is_path_valid_dir():
            raise ValueError(f'Path "{self.path}" is not a valid directory.')
        if self.file_glob is not None:
            kwargs.setdefault('file_glob', self.file_glob)
        if self.storage is Storage.LOCAL_FILE_SYSTEM:
            return FileSystemUtil.list(self.path, **kwargs)
        elif self.storage is Storage.S3:
            return S3Util.list(self.path, **kwargs)
        raise ValueError(f'Cannot list files in {self.storage} path: {self.path}')

    def list_metadata(self, **kwargs) -> List[FileMetadata]:
        if not self.is_path_valid_dir():
            raise ValueError(f'Path "{self.path}" is not a valid directory.')
        files_metadata: List[FileMetadata] = [
            self.update_params(
                name=None,
                path=fpath,
                file_glob=None,
            )
            for fpath in self.list(**kwargs)
        ]
        return files_metadata

    def copy_to_dir(self, destination: Union[FileMetadata, Dict, str]) -> bool:
        if self.is_path_valid_dir() is False:
            raise ValueError(f'Source path is not a valid directory: "{self.path}"')
        destination: FileMetadata = FileMetadata.of(destination)
        if destination.is_path_valid_dir() is False:
            raise ValueError(f'Destination path is not a valid directory: "{destination.path}"')
        if self.storage is Storage.LOCAL_FILE_SYSTEM and destination.storage is Storage.LOCAL_FILE_SYSTEM:
            return FileSystemUtil.copy_dir(self.path, destination.path)
        elif self.storage is Storage.LOCAL_FILE_SYSTEM and destination.storage is Storage.S3:
            return S3Util.copy_local_dir_to_s3(self.path, destination.path)
        elif self.storage is Storage.S3 and destination.storage is Storage.LOCAL_FILE_SYSTEM:
            return S3Util.copy_s3_dir_to_local(self.path, destination.path)
        elif self.storage is Storage.S3 and destination.storage is Storage.S3:
            return S3Util.copy_dir_between_s3_locations(self.path, destination.path)
        raise NotImplementedError(
            f'Copying from source storage "{self.storage}" to destination storage "{destination.storage}" '
            f'is not yet supported.'
        )
