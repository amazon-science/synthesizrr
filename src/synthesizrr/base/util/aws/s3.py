from typing import *
import io, random, re, math, os, time, boto3, botocore, fnmatch, pickle
from urllib.parse import urlparse, ParseResult
from synthesizrr.base.util.language import as_list, is_list_like, format_exception_msg, any_are_none, remove_values
from synthesizrr.base.util import Utility, StringUtil, FileSystemUtil, Log, Timer, shuffle_items


class S3Util(Utility):
    S3_BUCKET = 'Bucket'
    OBJECT_KEY = 'Key'
    ACL = 'ACL'
    S3_PATH_REGEX = StringUtil.CARET + StringUtil.S3_PREFIX + '(\S+?)' + StringUtil.SLASH + '(\S+)' + StringUtil.DOLLAR
    ## Permissions:
    S3_BUCKET_GET_OBJ_PERMISSION = 's3:GetObject'
    S3_BUCKET_LIST_PERMISSION = 's3:ListBucket'
    S3_BUCKET_PUT_OBJ_PERMISSION = 's3:PutObject'
    S3_BUCKET_DELETE_OBJ_PERMISSION = 's3:DeleteObject'

    S3_BUCKET_OWNER_FULL_CONTROL_ACL = 'bucket-owner-full-control'

    @classmethod
    def s3_path_exploder(cls, s3_path: str) -> Tuple[str, str]:
        s3_path = StringUtil.assert_not_empty_and_strip(s3_path)
        s3_parsed_result: ParseResult = urlparse(s3_path)
        s3_bucket, object_key = None, None
        if StringUtil.is_not_empty(s3_parsed_result.netloc) and StringUtil.SPACE not in s3_parsed_result.netloc:
            s3_bucket = s3_parsed_result.netloc
        if StringUtil.is_not_empty(s3_parsed_result.path) and s3_bucket is not None:
            object_key = StringUtil.remove_prefix(s3_parsed_result.path, StringUtil.SLASH)
        return s3_bucket, object_key

    @classmethod
    def s3_path_exploder_dict(cls, s3_path: str) -> Dict[str, str]:
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        return {
            cls.S3_BUCKET: s3_bucket,
            cls.OBJECT_KEY: object_key
        }

    @classmethod
    def is_valid_s3_path(cls, s3_path: str) -> bool:
        s3_path = StringUtil.assert_not_empty_and_strip(s3_path)
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        return StringUtil.is_not_empty(s3_bucket) and StringUtil.is_not_empty(object_key)

    @classmethod
    def is_path_valid_s3_dir(cls, s3_path: str) -> bool:
        return cls.is_valid_s3_path(s3_path) and s3_path.endswith(StringUtil.SLASH)

    @classmethod
    def get_s3_dir(cls, s3_path: str) -> str:
        if cls.is_path_valid_s3_dir(s3_path):
            return s3_path
        return StringUtil.SLASH.join(s3_path.split(StringUtil.SLASH)[:-1]) + StringUtil.SLASH

    @classmethod
    def check_bucket_permission(cls, s3_path: str, action_names: Union[str, List[str]]) -> bool:
        s3_bucket, _ = cls.s3_path_exploder(s3_path)
        if isinstance(action_names, str):
            action_names: List[str] = [action_names]
        ## Ref: https://stackoverflow.com/a/47058571
        iam = boto3.client('iam')
        sts = boto3.client('sts')
        # Get the arn represented by the currently configured credentials
        arn = sts.get_caller_identity()['Arn']
        # Create an arn representing the objects in a bucket
        bucket_objects_arn = f'arn:aws:s3:::{s3_bucket}/*'
        # Run the policy simulation for the basic s3 operations
        results = iam.simulate_principal_policy(
            PolicySourceArn=arn,
            ResourceArns=[bucket_objects_arn],
            ActionNames=action_names
        )
        for policy_result in results["EvaluationResults"]:
            if policy_result["EvalDecision"] != "allowed":
                return False
        return True

    @classmethod
    def can_read_bucket(cls, s3_path: str) -> bool:
        return cls.check_bucket_permission(s3_path, [cls.S3_BUCKET_GET_OBJ_PERMISSION, cls.S3_BUCKET_LIST_PERMISSION])

    @classmethod
    def can_write_bucket(cls, s3_path: str) -> bool:
        return cls.check_bucket_permission(s3_path, cls.S3_BUCKET_PUT_OBJ_PERMISSION)

    @classmethod
    def can_delete_from_bucket(cls, s3_path: str) -> bool:
        return cls.check_bucket_permission(s3_path, cls.S3_BUCKET_DELETE_OBJ_PERMISSION)

    @classmethod
    def s3_object_exists(cls, s3_path: str):
        return cls.get_s3_object_details(s3_path, log_error=False) is not None

    @classmethod
    def s3_object_does_not_exist(cls, s3_path: str):
        StringUtil.assert_not_empty(s3_path)
        return not cls.s3_object_exists(s3_path)

    @classmethod
    def get_s3_object_details(cls, s3_path: str, log_error: bool = True):
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        s3 = boto3.client('s3')
        try:
            return s3.head_object(Bucket=s3_bucket, Key=object_key)
        except Exception as e:
            if log_error:
                if e.response.get('Error').get('Code') == 404:
                    Log.error('Bucket %s does not contain key %s' % (s3_bucket, object_key))
                else:
                    Log.error(str(e))
        return None

    @classmethod
    def get_s3_object_etag(cls, s3_path: str):
        s3_resp = cls.get_s3_object_details(s3_path)
        if s3_resp is None:
            return None
        etag = s3_resp['ETag'].strip('"')
        ## To handle etags of multi-part uploads. Ref: https://stackoverflow.com/q/6591047/4900327
        etag = etag.split('-')[0]
        return etag

    @classmethod
    def get_s3_object_size(
            cls,
            s3_path: Union[List[str], str],
            unit: Optional[str] = None,
            decimals: int = 3,
            ignore_missing: bool = True
    ) -> Optional[Union[float, str]]:
        s3_obj_details: List = [
            cls.get_s3_object_details(s3_fpath)
            for s3_fpath in as_list(s3_path)
        ]
        if any_are_none(*s3_obj_details) and not ignore_missing:
            return None
        size_in_bytes: int = int(sum([
            s3_resp['ContentLength']
            for s3_resp in s3_obj_details
            if s3_resp is not None
        ]))
        if unit is not None:
            return StringUtil.convert_size_from_bytes(size_in_bytes, unit=unit, decimals=decimals)
        return StringUtil.readable_bytes(size_in_bytes, decimals=decimals)

    @classmethod
    def list_recursive_objects_in_dir(cls, *args, **kwargs) -> List[str]:
        return cls.list(*args, **kwargs)

    @classmethod
    def list(
            cls,
            s3_path: str,
            *,
            file_glob: str = StringUtil.DOUBLE_ASTERISK,
            ignored_files: Union[str, List[str]] = StringUtil.FILES_TO_IGNORE,
            **kwargs
    ) -> List[str]:
        ignored_files: List[str] = as_list(ignored_files)
        s3 = boto3.resource('s3')
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        s3_bucket_resource = s3.Bucket(s3_bucket)
        objs_in_dir: List[str] = [obj_path for obj_path in s3_bucket_resource.objects.filter(Prefix=object_key)]
        if len(objs_in_dir) == 0:
            return []
        objs_in_dir: List[str] = [
            os.path.join(StringUtil.S3_PREFIX, obj_path.bucket_name, obj_path.key)
            for obj_path in objs_in_dir
        ]
        objs_in_dir: List[str] = [
            obj_path for obj_path in objs_in_dir
            if fnmatch.fnmatch(StringUtil.remove_prefix(obj_path, s3_path), file_glob)
        ]
        obj_names_map: Dict[str, str] = {obj_path: os.path.basename(obj_path) for obj_path in objs_in_dir}
        obj_names_map = remove_values(obj_names_map, ignored_files)
        objs_in_dir = list(obj_names_map.keys())
        return objs_in_dir

    @classmethod
    def list_subdirs_in_dir(cls, *args, **kwargs) -> List[str]:
        return cls.list_subdirs(*args, **kwargs)

    @classmethod
    def list_subdirs(
            cls,
            s3_path: str,
            *,
            names_only: bool = False,
    ) -> List[str]:
        s3 = boto3.resource('s3')
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        if not object_key.endswith(StringUtil.SLASH):
            object_key += StringUtil.SLASH
        s3_bucket_resource = s3.Bucket(s3_bucket)
        paginator = s3_bucket_resource.meta.client.get_paginator('list_objects')
        pagination_params: Dict = dict(Prefix=object_key, Delimiter=StringUtil.SLASH)
        subdirs: List[str] = []
        ## Ref: https://stackoverflow.com/a/51372405
        for resp in paginator.paginate(Bucket=s3_bucket_resource.name, **pagination_params):
            if 'CommonPrefixes' in resp:
                subdirs.extend([
                    os.path.join(StringUtil.S3_PREFIX, s3_bucket,
                                 f['Prefix']) if not names_only
                    else StringUtil.remove_suffix(
                        StringUtil.remove_prefix(f['Prefix'], prefix=object_key),
                        suffix=StringUtil.SLASH,
                    )
                    for f in resp['CommonPrefixes']]
                )
        return sorted(subdirs)

    @classmethod
    def touch_s3_object(cls, s3_path: str):
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        s3 = boto3.client('s3')
        s3_resp = s3.put_object(Bucket=s3_bucket, Key=object_key)
        return s3_resp

    @classmethod
    def get_s3_object_str(cls, s3_path: str, retry: int = 1) -> str:
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        out_str = cls.stream_s3_object(s3_path, retry=retry).read().decode('utf-8')
        if StringUtil.is_empty(out_str):
            raise IOError(f'Object in bucket "{s3_bucket}" with key "{object_key}" seems to be empty')
        return out_str

    @classmethod
    def get_s3_object_pickle(cls, s3_path: str, retry: int = 1) -> Any:
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        pickled_data = cls.stream_s3_object(s3_path, retry=retry).read()
        loaded_data = pickle.loads(pickled_data)
        if loaded_data is None:
            raise IOError(f'Object in bucket "{s3_bucket}" with key "{object_key}" seems to be empty')
        return loaded_data

    @classmethod
    def stream_s3_object(cls, s3_path: str, retry: int = 1) -> io.IOBase:
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        s3 = boto3.resource('s3')
        retry_wait_times = cls.__get_retry_wait_times_list(retry)
        for retry_wait_time in retry_wait_times:
            try:
                time.sleep(retry_wait_time)
                obj = s3.Object(s3_bucket, object_key)
                stream = obj.get()['Body']
                return stream
            except Exception as e:
                Log.debug(format_exception_msg(e))
                if retry_wait_time != retry_wait_times[-1]:
                    Log.debug(f'Retrying retrieval of object at bucket "{s3_bucket}" with key "{object_key}"...')
        raise IOError(f'Cannot retrieve S3 object after {retry} attempts; bucket="{s3_bucket}", key="{object_key}"')

    @classmethod
    def put_s3_object_str(
            cls,
            s3_path: str,
            obj_str: str,
            overwrite: bool = True,
            num_attempts: int = 1,
    ):
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        s3 = boto3.client('s3')
        retry_wait_times = cls.__get_retry_wait_times_list(num_attempts)
        for retry_wait_time in retry_wait_times:
            try:
                time.sleep(retry_wait_time)
                if cls.s3_object_exists(s3_path) and overwrite is False:
                    raise FileExistsError(f'File already exists at {s3_path}, set overwrite=True to overwrite it.')
                s3.put_object(Body=obj_str, Bucket=s3_bucket, Key=object_key)
                if cls.s3_object_exists(s3_path) is False:
                    raise IOError(f'Could not put object at bucket {s3_bucket} and key {object_key}')
                return True
            except Exception as e:
                Log.error(format_exception_msg(e))
                if retry_wait_time != retry_wait_times[-1]:
                    Log.info(f'Retrying put of object at bucket {s3_bucket} and key {object_key}...')
        raise IOError('Could not successfully put object at bucket %s and key %s, after %s attempts' % (
            s3_bucket, object_key, num_attempts))

    @classmethod
    def put_s3_object_pickle(
            cls,
            s3_path: str,
            obj_data: Any,
            overwrite: bool = True,
            num_attempts: int = 1,
    ):
        s3_bucket, object_key = cls.s3_path_exploder(s3_path)
        s3 = boto3.client('s3')
        retry_wait_times = cls.__get_retry_wait_times_list(num_attempts)
        for retry_wait_time in retry_wait_times:
            try:
                time.sleep(retry_wait_time)
                if cls.s3_object_exists(s3_path) and overwrite is False:
                    raise FileExistsError(f'File already exists at {s3_path}, set overwrite=True to overwrite it.')
                serialized_data = pickle.dumps(obj_data)
                s3.put_object(Body=serialized_data, Bucket=s3_bucket, Key=object_key)
                if cls.s3_object_exists(s3_path) is False:
                    raise IOError(f'Could not put object at bucket {s3_bucket} and key {object_key}')
                return True
            except Exception as e:
                Log.error(format_exception_msg(e))
                if retry_wait_time != retry_wait_times[-1]:
                    Log.info(f'Retrying put of object at bucket {s3_bucket} and key {object_key}...')
        raise IOError('Could not successfully put object at bucket %s and key %s, after %s attempts' % (
            s3_bucket, object_key, num_attempts))

    @classmethod
    def copy_local_file_to_s3(cls, source_local_path: str, destination_s3_path: str, extra_args=None) -> bool:
        if extra_args is None:
            extra_args = {}
        try:
            FileSystemUtil.check_file_exists(source_local_path)
            s3 = boto3.resource('s3')
            ## Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.upload_file
            destination_s3_bucket, destination_object_key = cls.s3_path_exploder(destination_s3_path)
            s3.meta.client.upload_file(
                source_local_path,
                destination_s3_bucket,
                destination_object_key,
                extra_args
            )
            assert cls.s3_object_exists(destination_s3_path), f'Could not find file {destination_s3_path} after copying'
            return True
        except Exception as e:
            Log.error(format_exception_msg(e))
        return False

    @classmethod
    def copy_local_dir_to_s3(
            cls,
            source_local_dir: str,
            destination_s3_dir: str,
            log: bool = False,
            extra_args=None,
    ) -> bool:
        if not FileSystemUtil.dir_exists(source_local_dir):
            raise OSError(f'Source directory does not exist: "{source_local_dir}"')
        source_local_dir: str = FileSystemUtil.expand_dir(source_local_dir)
        local_fpaths: List[str] = FileSystemUtil.list(source_local_dir, recursive=True, only_files=True)
        if not cls.is_path_valid_s3_dir(destination_s3_dir):
            raise OSError(f'Destination is not a valid S3 directory: "{destination_s3_dir}"')
        for local_fpath in shuffle_items(local_fpaths):
            local_fname: str = local_fpath.replace(source_local_dir, '')
            s3_fpath: str = cls.construct_path_in_s3_dir(destination_s3_dir, name=local_fname, is_dir=False)
            if log:
                Log.info(f'Uploading file from "{local_fname}" to "{s3_fpath}"...')
            try:
                with Timer(silent=True) as timer:
                    cls.copy_local_file_to_s3(
                        source_local_path=local_fpath,
                        destination_s3_path=s3_fpath,
                        extra_args=extra_args,
                    )
                if log:
                    Log.info(
                        f'...uploaded file from "{local_fpath}" to "{s3_fpath}" '
                        f'in {timer.time_taken_str}.'
                    )
            except Exception as e:
                Log.error(format_exception_msg(e))
                return False
        return True

    @classmethod
    def copy_s3_file_to_local(cls, source_s3_path: str, destination_local_path: str, extra_args=None) -> bool:
        if extra_args is None:
            extra_args = {}
        try:
            s3 = boto3.resource('s3')
            assert cls.s3_object_exists(source_s3_path), f'Could not find file {source_s3_path} to copy'
            ## Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.download_file
            source_s3_bucket, source_object_key = cls.s3_path_exploder(source_s3_path)
            s3.meta.client.download_file(
                source_s3_bucket,
                source_object_key,
                destination_local_path,
                extra_args
            )
            FileSystemUtil.check_file_exists(destination_local_path)
            return True
        except Exception as e:
            Log.error(format_exception_msg(e))
        return False

    @classmethod
    def copy_s3_dir_to_local(
            cls,
            source_s3_dir: str,
            destination_local_dir: str,
            force_download: bool = False,
            log: bool = False,
            wait_timeout: int = 300,
            extra_args=None,
    ) -> bool:
        s3_fpaths: List[str] = cls.list_recursive_objects_in_dir(source_s3_dir)
        s3_fnames: Set[str] = {
            s3_fpath.split(StringUtil.SLASH)[-1] for s3_fpath in s3_fpaths
        }
        destination_local_dir: str = FileSystemUtil.expand_dir(destination_local_dir)
        local_fpaths: List[str] = FileSystemUtil.list(destination_local_dir)
        if force_download:
            local_fnames: Set[str] = set()
        else:
            local_fnames: Set[str] = {
                local_fpath.split(os.path.sep)[-1]
                for local_fpath in local_fpaths
            }
        if local_fnames < s3_fnames:
            s3_fpaths_to_download: List[str] = [
                s3_fpath
                for s3_fpath in s3_fpaths
                if s3_fpath.split(StringUtil.SLASH)[-1] in (s3_fnames - local_fnames)
            ]
            if len(s3_fpaths_to_download) == 0:
                return True
            time.sleep(random.randint(0, 30000) / 1000)  ## Wait randomly between 0 and 30 seconds to acquire locks.
            lock_file: str = os.path.join(destination_local_dir, 'litmus-download.lock')
            if not FileSystemUtil.file_exists(lock_file):
                ## Acquire lock:
                try:
                    FileSystemUtil.touch_file(lock_file)
                    ## If we don't have a file, download a copy:
                    if log:
                        Log.info(f'Downloading {len(s3_fpaths_to_download)} files from S3...')
                    for s3_fpath_to_download in s3_fpaths_to_download:
                        fname: str = s3_fpath_to_download.split(StringUtil.SLASH)[-1]
                        local_fpath: str = os.path.join(destination_local_dir, fname)
                        if log:
                            Log.info(f'Downloading file from "{s3_fpath_to_download}" to "{local_fpath}"...')
                        with Timer(silent=True) as timer:
                            cls.copy_s3_file_to_local(s3_fpath_to_download, local_fpath, extra_args=extra_args)
                        if log:
                            Log.info(
                                f'...downloaded file from "{s3_fpath_to_download}" to "{local_fpath}" '
                                f'in {timer.time_taken_str}.'
                            )
                finally:
                    FileSystemUtil.rm_file(lock_file)
            else:
                for i in range(wait_timeout // 10):
                    if not FileSystemUtil.file_exists(lock_file):
                        return True
                    time.sleep(10)
                raise SystemError(
                    f'Waited for {wait_timeout} sec but still not completed downloading files to '
                    f'"{destination_local_dir}". Files which we had started downloading: {s3_fpaths_to_download}'
                )

        return True

    @classmethod
    def copy_file_between_s3_locations(cls, source_s3_path: str, destination_s3_path: str, extra_args=None) -> bool:
        if extra_args is None:
            extra_args = {}
        try:
            s3 = boto3.resource('s3')
            assert cls.s3_object_exists(source_s3_path)
            ## Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.copy
            destination_s3_bucket, destination_object_key = cls.s3_path_exploder(destination_s3_path)
            s3.meta.client.copy(
                cls.s3_path_exploder_dict(source_s3_path),
                destination_s3_bucket,
                destination_object_key,
                extra_args
            )
            assert cls.s3_object_exists(destination_s3_path), f'Could not find file {destination_s3_path} after copying'
            return True
        except Exception as e:
            Log.error(format_exception_msg(e))
        return False

    @classmethod
    def copy_dir_between_s3_locations(cls, source_s3_dir: str, destination_s3_dir: str, extra_args=None) -> bool:
        if not cls.is_path_valid_s3_dir(source_s3_dir):
            raise ValueError(f'Invalid s3 source directory: "{source_s3_dir}"')
        if not cls.is_path_valid_s3_dir(destination_s3_dir):
            raise ValueError(f'Invalid s3 destination directory: "{destination_s3_dir}"')
        source_s3_fpaths: List[str] = cls.list(source_s3_dir)
        try:
            for source_s3_fpath in source_s3_fpaths:
                if source_s3_fpath.endswith(StringUtil.SLASH):
                    continue  ## Only copy files
                source_s3_fname: str = source_s3_fpath.replace(source_s3_dir, '')
                destination_s3_fpath: str = cls.construct_path_in_s3_dir(
                    destination_s3_dir,
                    name=source_s3_fname,
                    is_dir=False,
                )
                cls.copy_file_between_s3_locations(source_s3_fpath, destination_s3_fpath, extra_args=extra_args)
            return True
        except Exception as e:
            Log.error(format_exception_msg(e))
            return False

    @staticmethod
    def construct_path_in_s3_dir(
            s3_path: str,
            name: str,
            is_dir: bool,
            file_ending: Optional[str] = None,
    ):
        """
        If the path is a dir, uses the inputs to construct a file path. If path is a file, returns unchanged.
        :param s3_path: path to dir (or file) in S3.
        :param name: name of the file.
        :param is_dir: whether the newly created path should be a directory or file.
        :param file_ending: (optional) a string of the file ending.
        :return: file path string.
        """
        if S3Util.is_path_valid_s3_dir(s3_path):
            file_name: str = StringUtil.assert_not_empty_and_strip(name)
            if file_ending is not None:
                file_name += StringUtil.assert_not_empty_and_strip(file_ending)
            if s3_path.endswith(StringUtil.SLASH):
                out_s3_path: str = StringUtil.EMPTY.join([s3_path, file_name])
            else:
                out_s3_path: str = StringUtil.SLASH.join([s3_path, file_name])
            if is_dir and not out_s3_path.endswith(StringUtil.SLASH):
                ## Add a slash at the end:
                out_s3_path += StringUtil.SLASH
            return out_s3_path
        else:
            return s3_path

    @classmethod
    def __get_retry_wait_times_list(cls, num_attempts):
        return [(math.pow(2, i - 1) - 1) for i in range(1, num_attempts + 1)]  ## 0, 1, 3, 7, 15, ...

    @classmethod
    def generate_presigned_s3_url(cls, s3_path: str, expiry: int = 7 * 24 * 60 * 60) -> str:
        url = boto3.client('s3').generate_presigned_url(
            ClientMethod='get_object',
            Params=S3Util.s3_path_exploder_dict(s3_path),
            ExpiresIn=expiry,
            ## Max expiry time, see: X-Amz-Expires here: https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html
        )
        return url
