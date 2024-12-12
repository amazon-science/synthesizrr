from typing import *
from abc import abstractmethod, ABC
import time, io, json, pathlib, numpy as np
from math import inf
from synthergent.base.constants import FileFormat, Alias, Storage, FileContents, MLTypeSchema, FILE_FORMAT_TO_FILE_ENDING_MAP
from synthergent.base.util.language import as_list, classproperty, format_exception_msg, safe_validate_arguments, filter_kwargs, \
    get_default, shuffle_items
from synthergent.base.util import StringUtil, FileSystemUtil, Parameters, Registry, Log
from synthergent.base.util.aws import S3Util
from synthergent.base.data.FileMetadata import FileMetadata
from pydantic import conint, confloat, constr, root_validator, Extra

Reader = "Reader"


class Reader(Parameters, Registry, ABC):
    """
    Abstract base class for readers from various sources and file-formats.
    Params:
    - params: params for reading from the particular file-format. These are forwarded to .read_csv, .read_yaml, etc.
    - filter_kwargs: whether to filter out (ignore) args in `params` that are not supported by the reader function.
    """
    _allow_multiple_subclasses = True
    _allow_subclass_override = False
    file_formats: ClassVar[Tuple[FileFormat, ...]]
    file_contents: ClassVar[Tuple[FileContents, ...]]
    streams: ClassVar[Tuple[Type[io.IOBase], ...]]
    retry: conint(ge=1) = 1
    retry_wait: confloat(ge=0.0) = 5.0
    shuffled_multi_read: bool = True

    class Config(Parameters.Config):
        extra = Extra.ignore

    class Params(Parameters):
        """
        BaseModel for parameters. Expected to be overridden by subclasses.
        """

        class Config(Parameters.Config):
            ## Allow extra keyword parameters to be used when initializing the reader.
            ## These will be forwarded to the respective reader method like .read_csv, .read_json, etc.
            extra = Extra.allow

    params: Params = {}
    filter_kwargs: bool = True

    @root_validator(pre=True)
    def convert_params(cls, params: Dict) -> Dict:
        Alias.set_retry(params)
        params['params'] = cls._convert_params(cls.Params, params.get('params'))
        return params

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return as_list(cls.file_formats) + as_list(cls.file_ending)

    def filtered_params(self, *reader_fn: Union[Callable, Tuple[Callable, ...]]) -> Dict:
        filtered_params: Dict[str, Any] = {
            **self.params.dict(),
        }
        if self.filter_kwargs:
            filtered_params: Dict[str, Any] = filter_kwargs(reader_fn, **filtered_params)
        return filtered_params

    @classmethod
    @safe_validate_arguments
    def of(cls, file_format: Union[FileFormat, str], **kwargs) -> Reader:
        ReaderClasses: List[Type[Reader]] = as_list(cls.get_subclass(file_format, raise_error=True))
        ReaderClass: Type[Reader] = ReaderClasses[0]
        return ReaderClass(**kwargs)

    @classmethod
    @safe_validate_arguments
    def any(cls, source: Union[FileMetadata, io.IOBase, str], **kwargs) -> Any:
        if isinstance(source, FileMetadata):
            return Reader.of(source.format, **kwargs).read(source, **kwargs)
        elif isinstance(source, str):
            return Reader.of(FileMetadata.detect_file_format(source), **kwargs).read(source, **kwargs)
        raise ValueError(
            f'Cannot detect file format from data of type {type(source)}; '
            f'please instantiate a reader using {Reader}.of(file_format=...)'
        )

    @safe_validate_arguments
    def read(
            self,
            source: Union[FileMetadata, io.IOBase, str],
            **kwargs
    ) -> Any:
        """Wrapper function to read all data using file contents"""
        if isinstance(source, FileMetadata):
            return self.read_metadata(source, **kwargs)
        else:
            storage: Optional[Storage] = FileMetadata.detect_storage(source)
            if storage is Storage.STREAM:
                return self.read_stream(source, **kwargs)
            elif storage is Storage.S3:
                return self.read_s3(source, **kwargs)
            ## Add any other remote storage solutions here.
            elif storage is Storage.URL:
                return self.read_url(source, **kwargs)
            elif storage is Storage.LOCAL_FILE_SYSTEM:
                return self.read_local(source, **kwargs)
        raise IOError(f'Cannot read object of type: {type(source)}: {source}')

    @safe_validate_arguments
    def read_stream(
            self,
            stream: io.IOBase,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            log_perf: bool = True,
            **kwargs
    ) -> Any:
        """
        Method to read data from text and binary input streams. Ref: https://docs.python.org/3/library/io.html
        :param stream: Python stream object. Must be an instance of io.IOBase.
        :param file_contents: (Optional) informs the reader what the contents of the stream is.
        :param data_schema: (Optional) the columns which should be read by the reader. Only for reading dataframes.
        :param log_perf: whether to log performance information or not.
        :return: Python object of stream contents.
        :raises: TypeError if passed stream is not a supported stream type, IOError if stream cannot be read.
        """
        self._check_file_contents(file_contents)
        if not isinstance(stream, tuple(list(self.streams))):
            raise TypeError(
                f'Input stream should be one of the following types: {self.streams}; '
                f'found object of type: "{type(stream)}"'
            )
        if not stream.readable() or stream.closed:
            raise IOError(f'Cannot read from stream of type "{type(stream)}"')

        try:
            start = time.perf_counter()
            ## Do NOT log a long message at the start, as that can make reading very slow during realtime deployment.
            obj = self._read_stream(
                stream,
                file_contents=file_contents,
                data_schema=data_schema,
                log_perf=log_perf,
                **kwargs
            )
            end = time.perf_counter()
            if log_perf:
                Log.debug(f'Took {StringUtil.readable_seconds(end - start)} to read from stream.')
            return obj
        except Exception as e:
            Log.error(format_exception_msg(e))
        raise IOError(f'Cannot read from stream of type "{type(stream)}"')

    @abstractmethod
    @safe_validate_arguments
    def _read_stream(
            self,
            stream: io.IOBase,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs
    ) -> Any:
        pass

    @safe_validate_arguments
    def read_url(
            self,
            url: constr(min_length=1),
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            log_perf: bool = True,
            **kwargs
    ) -> Any:
        """
        Method to read a URL over HTTP/HTTPS/SCP/SFTP/etc.
        :param url: Python URL string.
        :param file_contents: (Optional) informs the reader what the contents of the URL is.
        :param data_schema: (Optional) the columns which should be read by the reader. Only for reading dataframes.
        :param log_perf: whether to log performance information or not.
        :return: Python object of URL contents.
        :raises: TypeError if passed URL is not a string, IOError if URL cannot be read.
        """
        self._check_file_contents(file_contents)
        try:
            start = time.perf_counter()
            ## Do NOT log a long message at the start, as that can make reading very slow during realtime deployment.
            obj = self._read_url(
                url,
                file_contents=file_contents,
                data_schema=data_schema,
                log_perf=log_perf,
                **kwargs
            )
            end = time.perf_counter()
            if log_perf:
                Log.debug(f'Took {StringUtil.readable_seconds(end - start)} to read from url: "{url}"')
            return obj
        except Exception as e:
            Log.error(format_exception_msg(e))
        raise IOError(f'Cannot read from url: "{url}"')

    @abstractmethod
    @safe_validate_arguments
    def _read_url(
            self,
            url: str,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> Any:
        pass

    @safe_validate_arguments
    def read_local(
            self,
            local_path: Union[str, pathlib.Path],
            *,
            file_glob: str = StringUtil.DOUBLE_ASTERISK,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            files_to_ignore: List[str] = StringUtil.FILES_TO_IGNORE,
            log_perf: bool = True,
            **kwargs
    ) -> Any:
        """
        Reads data from the local filesystem.
        :param local_path: Path to file, dir or glob.
        :param file_glob: if the passed path is a dir, this matches a particular file name. Defaults to "**"
        :param file_contents: (Optional) informs the reader what the contents of the file(s) are.
        :param data_schema: (Optional) the columns which should be read by the reader. Only for reading dataframes.
        :param files_to_ignore: (Optional) if passing a directory or glob, ignores these files.
        :param log_perf: whether to log performance information or not.
        :return: Python object of file contents.
        :raises: FileNotFoundError if invalid path or no file(s) at path, IOError if file(s) cannot be read.
        """
        self._check_file_contents(file_contents)
        try:
            if isinstance(local_path, pathlib.Path):
                local_path: str = str(local_path)
            if FileSystemUtil.file_exists(local_path):
                local_path_to_read: str = local_path
                num_files_to_read: int = 1
            elif FileSystemUtil.dir_exists(local_path):
                local_path_to_read: List[str] = FileSystemUtil.list(
                    local_path,
                    file_glob=file_glob,
                    **{
                        **dict(
                            ignored_files=files_to_ignore,
                            recursive=True,
                            only_files=True,
                        ),
                        **kwargs,
                    }
                )
                num_files_to_read = len(local_path_to_read)
                if num_files_to_read == 0:
                    raise FileNotFoundError(f'No files found in directory at local path "{local_path}"')
                if self.shuffled_multi_read:
                    local_path_to_read: List[str] = list(shuffle_items(local_path_to_read))
            else:
                ## Check if input was a glob:
                local_path_to_read: List[str] = FileSystemUtil.list(
                    local_path,
                    **{
                        **dict(
                            file_glob=StringUtil.EMPTY,
                            ignored_files=files_to_ignore,
                            recursive=True,
                        ),
                        **kwargs,
                    }
                )
                num_files_to_read = len(local_path_to_read)
                if num_files_to_read == 0:
                    raise FileNotFoundError(f'No file or directory at local path "{local_path}"')
                if self.shuffled_multi_read:
                    local_path_to_read: List[str] = list(np.random.permutation(local_path_to_read))
            if log_perf:
                Log.debug(f'Reading {num_files_to_read} file(s) from local path "{local_path}" using {str(self)}')
            start = time.perf_counter()
            obj = self._read_local(
                local_path_to_read,
                file_contents=file_contents,
                data_schema=data_schema,
                log_perf=log_perf,
                **kwargs
            )
            end = time.perf_counter()
            if log_perf:
                Log.debug(
                    f'Took {StringUtil.readable_seconds(end - start)} '
                    f'to read {num_files_to_read} file(s) from local path "{local_path}"'
                )
            return obj
        except Exception as e:
            Log.error(format_exception_msg(e))
            raise IOError(f'Cannot read from local path "{local_path}"')

    @abstractmethod
    @safe_validate_arguments
    def _read_local(
            self,
            local_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs
    ) -> Any:
        pass

    @safe_validate_arguments
    def read_s3(
            self,
            s3_path: str,
            *,
            file_glob: str = StringUtil.DOUBLE_ASTERISK,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            files_to_ignore: List[str] = StringUtil.FILES_TO_IGNORE,
            log_perf: bool = True,
            **kwargs
    ) -> Any:
        """
        Reads data from AWS S3.
        :param s3_path: Path to S3 file or dir.
        :param file_glob: if the passed path is a dir, this matches a particular file name. Defaults to "**"
        :param file_contents: (Optional) informs the reader what the contents of the file(s) are.
        :param data_schema: (Optional) the columns which should be read by the reader. Only for reading dataframes.
        :param files_to_ignore: (Optional) if passing a directory or glob, ignores these files.
        :param log_perf: whether to log performance information or not.
        :return: Python object of file contents.
        :raises: FileNotFoundError if invalid path or no file(s) at path, IOError if file(s) cannot be read.
        """
        if not S3Util.is_valid_s3_path(s3_path):
            raise FileNotFoundError(f'Not a valid S3 path: "{s3_path}"')
        self._check_file_contents(file_contents)
        try:
            if S3Util.is_path_valid_s3_dir(s3_path):
                s3_path_to_read: List[str] = S3Util.list(
                    s3_path,
                    ignored_files=files_to_ignore,
                    file_glob=file_glob,
                    **kwargs
                )
                num_files_to_read = len(s3_path_to_read)
                if num_files_to_read == 0:
                    raise FileNotFoundError(f'No files found in directory at S3 path "{s3_path}"')
                if self.shuffled_multi_read:
                    s3_path_to_read: List[str] = list(np.random.permutation(s3_path_to_read))
            elif S3Util.s3_object_exists(s3_path):
                s3_path_to_read: str = s3_path
                num_files_to_read: int = 1
            else:
                ## Globs are not supported by S3:
                raise FileNotFoundError(
                    f'Cannot read file or directory at S3 path "{s3_path}";\n'
                    f'Note that glob-like patterns are not supported by S3'
                )
            if log_perf:
                Log.debug(f'Reading {num_files_to_read} file(s) from S3 path "{s3_path}" using {str(self)}')
            start = time.perf_counter()
            obj = self._read_s3(
                s3_path_to_read,
                file_contents=file_contents,
                data_schema=data_schema,
                log_perf=log_perf,
                **kwargs
            )
            end = time.perf_counter()
            if log_perf:
                Log.debug(
                    f'Took {StringUtil.readable_seconds(end - start)} '
                    f'to read {num_files_to_read} file(s) from S3 path "{s3_path}"'
                )
            return obj
        except Exception as e:
            Log.error(format_exception_msg(e))
            raise IOError(f'Cannot read file from S3 path "{s3_path}":\n{format_exception_msg(e)}')

    @abstractmethod
    @safe_validate_arguments
    def _read_s3(
            self,
            s3_path: str,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs
    ) -> Any:
        pass

    @safe_validate_arguments
    def read_metadata(
            self,
            file: FileMetadata,
            *,
            file_glob: Optional[str] = None,
            **kwargs,
    ) -> Any:
        file_glob: Optional[str] = get_default(file_glob, file.file_glob)
        if file_glob is not None:
            kwargs['file_glob'] = file_glob

        if file.storage is Storage.LOCAL_FILE_SYSTEM:
            return self.read_local(
                file.path,
                file_contents=file.contents,
                data_schema=file.data_schema,
                **kwargs,
            )
        elif file.storage is Storage.URL:
            return self.read_url(
                file.path,
                file_contents=file.contents,
                data_schema=file.data_schema,
                **kwargs,
            )
        elif file.storage is Storage.S3:
            return self.read_s3(
                file.path,
                file_contents=file.contents,
                data_schema=file.data_schema,
                **kwargs,
            )
        raise NotImplementedError(f'Reading from {file.storage} is not supported.')

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
                raise ValueError(f'{cls.class_name} only supports contents {cls.file_contents}.')
