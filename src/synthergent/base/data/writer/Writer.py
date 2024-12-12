from typing import *
import time, io, os
from abc import abstractmethod, ABC
from synthergent.base.constants import FileFormat, MLType, MLTypeSchema, Storage, FileContents, FILE_FORMAT_TO_FILE_ENDING_MAP
from synthergent.base.util.aws import S3Util
from synthergent.base.util import Parameters, AutoEnum, auto, FileSystemUtil, StringUtil, Registry, Log, as_list, \
    filter_kwargs, format_exception_msg, classproperty, safe_validate_arguments
from synthergent.base.data.FileMetadata import FileMetadata
from pydantic import root_validator, constr, confloat, conint, Extra

Writer = "Writer"


class Writer(Parameters, Registry, ABC):
    """
    Abstract base class for file writers to various destinations and file-formats.
    Params:
    - params: params for writing to the particular file-format. These are forwarded to .to_csv, .to_yaml, etc.
    - filter_kwargs: whether to filter out (ignore) args in `params` that are not supported by the writer function.
    """
    ## Ref for homogeneous Tuple typing: https://docs.python.org/3.6/library/typing.html#typing.Tuple
    file_formats: ClassVar[Tuple[FileFormat, ...]]
    file_contents: ClassVar[Tuple[FileContents, ...]]
    streams: ClassVar[Tuple[Type[io.IOBase], ...]]

    class Config(Parameters.Config):
        extra = Extra.ignore

    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        """

        class Config(Parameters.Config):
            ## Allow extra keyword parameters to be used when initializing the writer.
            ## These will be forwarded to the respective writer method like .to_csv, .to_json, etc.
            extra = Extra.allow

    params: Params = {}
    filter_kwargs: bool = True

    @root_validator(pre=True)
    def convert_params(cls, params: Dict):
        params['params'] = cls._convert_params(cls.Params, params.get('params'))
        return params

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return as_list(cls.file_formats) + as_list(cls.file_ending)

    def filtered_params(self, *writer_fn: Union[Callable, Tuple[Callable, ...]]) -> Dict:
        filtered_params: Dict[str, Any] = {
            **self.params.dict(),
        }
        if self.filter_kwargs:
            filtered_params: Dict[str, Any] = filter_kwargs(writer_fn, **filtered_params)
        return filtered_params

    @classmethod
    @safe_validate_arguments
    def of(cls, file_format: Union[FileFormat, str], **kwargs) -> Writer:
        WriterClasses: List[Type[Writer]] = as_list(cls.get_subclass(file_format, raise_error=True))
        WriterClass: Type[Writer] = WriterClasses[0]
        return WriterClass(**kwargs)

    @safe_validate_arguments
    def write(
            self,
            destination: Union[io.IOBase, FileMetadata, str],
            data: Any,
            raise_error: bool = True,
            **kwargs,
    ) -> bool:
        """Wrapper function to read all data using file contents"""
        if isinstance(destination, FileMetadata):
            return self.write_metadata(destination, data, raise_error=raise_error, **kwargs)
        storage: Storage = FileMetadata.detect_storage(destination)
        if storage is Storage.STREAM:
            return self.write_stream(destination, data, raise_error=raise_error, **kwargs)
        elif storage is Storage.S3:
            return self.write_s3(destination, data, raise_error=raise_error, **kwargs)
        ## Add any other remote storage solutions here.
        elif storage is Storage.LOCAL_FILE_SYSTEM:
            return self.write_local(destination, data, raise_error=raise_error, **kwargs)
        elif raise_error:
            raise IOError(f'Cannot write object of type {type(data)} to destination of type {type(destination)}')
        return False

    @safe_validate_arguments
    def write_stream(
            self,
            stream: io.IOBase,
            data: Any,
            file_contents: Optional[FileContents] = None,
            overwrite: bool = False,
            log_perf: bool = True,
            raise_error: bool = True,
            **kwargs,
    ) -> bool:
        """
        Method to write data to text and binary input streams. Ref: https://docs.python.org/3/library/io.html
        :param stream: Python stream object. Must be an instance of io.IOBase.
        :param data: data to write. Must be string or bytes.
        :param file_contents: (Optional) informs the writer what the contents of the stream is.
        :param overwrite: (default=False) whether to overwrite if the stream already has contents.
        :param log_perf: whether to log performance information or not.
        :param raise_error: whether to raise an error if writing does not succeed. If this is set to False,
         a True/False flag is returned.
        :return: True if write is successful, false if unsuccessful.
        :raises: ,
        :raises: if raise_error=True, this function will raise:
         - ValueError if file_contents are not None and not one of the supported values.
         - TypeError if passed stream is not a supported stream type.
         - IOError if stream is not writable.
         - FileExistsError if overwrite=False and a file or non-empty folder already exists at the path.
         - PermissionError if file or folder is not writable.
         - IOError if writing the file failed.
        """
        try:
            self._check_file_contents(file_contents)
            if not isinstance(stream, self.streams):
                raise TypeError(
                    f'Input stream should be one of following types: {self.streams}; '
                    f'found object of type "{type(stream)}"'
                )
            if not stream.writable() or stream.closed:
                raise IOError(f'Cannot write to stream of type "{str(type(stream))}"')
            start = time.perf_counter()
            if overwrite:
                stream.seek(0)  ## Write from start of stream, overwriting data currently in stream.
            start_point: int = stream.tell()
            self._write_stream(
                stream,
                data,
                file_contents=file_contents,
                log_perf=log_perf,
                **kwargs,
            )
            stream.seek(start_point)  ## So that a subsequent call to .read() will return the contents we have written.
            end = time.perf_counter()
            if log_perf:
                Log.debug(
                    f'Took {StringUtil.readable_seconds(end - start)} '
                    f'to write to stream of type {str(type(stream))}'
                )
            return True
        except Exception as e:
            if raise_error:
                raise e
            Log.error(format_exception_msg(e))
            return False

    @abstractmethod
    def _write_stream(
            self,
            stream: io.IOBase,
            data: Any,
            file_contents: FileContents,
            **kwargs,
    ) -> NoReturn:
        pass

    @safe_validate_arguments
    def write_local(
            self,
            local_path: str,
            data: Any,
            file_contents: Optional[FileContents] = None,
            overwrite: bool = False,
            check_permissions: bool = True,
            log_perf: bool = True,
            raise_error: bool = True,
            **kwargs,
    ) -> bool:
        """
        Write data to the local filesystem.
        :param local_path: Path to file or dir. Must be writable.
        :param data: data to write.
        :param file_contents: (Optional) informs the writer what the contents of the stream is.
        :param overwrite: (default=False) whether to overwrite if the file or folder already has contents.
        :param check_permissions: whether to check if the process has permissions to write to the file or dir.
        :param log_perf: whether to log performance information or not.
        :param raise_error: whether to raise an error if writing does not succeed. If this is set to False,
         a True/False flag is returned.
        :return: True if writing succeeds, False if writing fails and raise_error=False.
        :raises: if raise_error=True, this function will raise:
         - ValueError if file_contents are not None and not one of the supported values.
         - PermissionError if file or folder is not writable.
         - FileExistsError if overwrite=False and a file or non-empty folder already exists at the path.
         - IOError if writing the file failed.
        """
        try:
            if log_perf:
                Log.debug(f'Writing to "{local_path}" using {str(self)}')
            start = time.perf_counter()
            self._check_file_contents(file_contents)
            if check_permissions and not FileSystemUtil.is_writable(local_path):
                raise PermissionError(f'Insufficient permissions to write to "{local_path}"')
            if overwrite is False:
                if FileSystemUtil.file_exists(local_path):
                    raise FileExistsError(f'File already exists at path "{local_path}"')
                if FileSystemUtil.dir_exists(local_path) and len(FileSystemUtil.list(local_path)) > 0:
                    raise FileExistsError(f'Non-empty folder already exists at path "{local_path}"')
            FileSystemUtil.mkdir_if_does_not_exist(FileSystemUtil.get_dir(local_path))
            written_paths: List[str] = as_list(self._write_local(
                local_path,
                data,
                file_contents=file_contents,
                log_perf=log_perf,
                **kwargs,
            ))
            end = time.perf_counter()
            if log_perf:
                if FileSystemUtil.is_path_valid_dir(local_path):
                    Log.debug(
                        f'Took {StringUtil.readable_seconds(end - start)} '
                        f'to write {len(written_paths)} file(s) totalling '
                        f'size {FileSystemUtil.get_file_size(written_paths)}, to local directory "{local_path}".'
                    )
                else:
                    assert len(written_paths) == 1
                    written_path: str = written_paths[0]
                    Log.debug(
                        f'Took {StringUtil.readable_seconds(end - start)} to write file '
                        f'of size {FileSystemUtil.get_file_size(written_path)} to local file "{written_path}".'
                    )
            return True
        except Exception as e:
            if raise_error:
                raise e
            Log.error(format_exception_msg(e))
            return False

    @abstractmethod
    def _write_local(
            self,
            local_path: str,
            data: Any,
            file_contents: FileContents,
            **kwargs,
    ) -> Union[str, List[str]]:
        pass

    @safe_validate_arguments
    def write_s3(
            self,
            s3_path: str,
            data: Any,
            file_contents: FileContents = None,
            overwrite: bool = False,
            log_perf: bool = True,
            raise_error: bool = True,
            **kwargs,
    ) -> bool:
        """
        :param s3_path: path to write data. Must be a valid S3 path and user must have S3:PutObject permissions.
        :param data: data to write.
        :param file_contents: (Optional) informs the writer what the contents of the stream is.
        :param overwrite: (default=False) whether to overwrite if the file or folder already has contents.
        :param log_perf: whether to log performance information or not.
        :param raise_error: whether to raise an error if writing does not succeed. If this is set to False,
         a True/False flag is returned.
        :return: True if writing succeeds, False if writing fails and raise_error=False.
        :raises: if raise_error=True, this function will raise:
         - ValueError if file_contents are not None and not one of the supported values.
         - FileExistsError if overwrite=False and a file or non-empty folder already exists at the path.
         - IOError if writing the file failed.
        """
        try:
            if overwrite is False and S3Util.s3_object_exists(s3_path):
                raise FileExistsError(f'File already exists at path "{s3_path}"')
            self._check_file_contents(file_contents)
            if log_perf:
                Log.debug(f'Writing file "{s3_path}" using {str(self)}')
            start = time.perf_counter()
            written_s3_paths: List[str] = as_list(self._write_s3(
                s3_path,
                data,
                file_contents=file_contents,
                log_perf=log_perf,
                **kwargs,
            ))
            end = time.perf_counter()
            if log_perf:
                if S3Util.is_path_valid_s3_dir(s3_path):
                    Log.debug(
                        f'Took {StringUtil.readable_seconds(end - start)} '
                        f'to write {len(written_s3_paths)} file(s) totalling '
                        f'size {S3Util.get_s3_object_size(written_s3_paths)}, to S3 directory "{s3_path}".'
                    )
                else:
                    assert len(written_s3_paths) == 1
                    written_s3_path: str = written_s3_paths[0]
                    Log.debug(
                        f'Took {StringUtil.readable_seconds(end - start)} to write file '
                        f'of size {S3Util.get_s3_object_size(written_s3_path)} to S3 file "{written_s3_path}".'
                    )
            return True
        except Exception as e:
            if raise_error:
                raise e
            Log.error(format_exception_msg(e))
            return False

    @abstractmethod
    def _write_s3(
            self,
            s3_path: str,
            data: Any,
            file_contents: FileContents,
            **kwargs,
    ) -> Union[str, List[str]]:
        pass

    @safe_validate_arguments
    def write_metadata(
            self,
            file: FileMetadata,
            data: Any,
            **kwargs,
    ) -> bool:
        if file.storage is Storage.LOCAL_FILE_SYSTEM:
            return self.write_local(
                local_path=file.path,
                data=data,
                **{
                    **dict(
                        file_name=file.name,
                    ),
                    **kwargs,
                }
            )
        elif file.storage is Storage.S3:
            return self.write_s3(
                s3_path=file.path,
                data=data,
                **{
                    **dict(
                        file_name=file.name,
                    ),
                    **kwargs,
                }
            )
        elif file.storage is Storage.STREAM:
            return self.write_stream(
                stream=file.path,
                data=data,
                **kwargs,
            )
        raise NotImplementedError(f'Writing to {file.storage} is not supported.')

    @classproperty
    def file_ending(cls) -> str:
        ## Get a default file ending, mapped from the file-format:
        file_format: FileFormat = as_list(cls.file_formats)[0]
        file_endings: List[str] = as_list(FILE_FORMAT_TO_FILE_ENDING_MAP.get(file_format, []))
        if len(file_endings) == 0:
            raise ValueError(f'No file ending registered for supported file formats of class {cls.class_name}.')
        return file_endings[0]

    @classmethod
    def _check_file_contents(cls, file_contents: Optional[FileContents]) -> NoReturn:
        if file_contents is not None:
            if file_contents not in cls.file_contents:
                raise ValueError(f'{cls.class_name} only supports contents {cls.file_contents}')
