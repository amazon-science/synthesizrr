import pathlib
from typing import *
from abc import abstractmethod, ABC
import io, numpy as np, pandas as pd
from synthergent.base.util import is_list_like, StringUtil, run_concurrent, run_parallel, run_parallel_ray, accumulate, Log, \
    format_exception_msg, as_list, retry, dispatch
from synthergent.base.constants import FileContents, Parallelize, Storage, TensorShortHand, SHORTHAND_TO_TENSOR_LAYOUT_MAP, \
    MLType, MLTypeSchema
from synthergent.base.data.reader.Reader import Reader
from synthergent.base.data.asset import Asset


class AssetReader(Reader, ABC):
    file_contents = [
        FileContents.ASSET,
    ]
    streams = [io.BytesIO]
    parallelize: Parallelize = Parallelize.threads
    return_as: Optional[TensorShortHand] = None
    asset_mltype: ClassVar[MLType]
    asset_params: Dict[str, Any] = {}

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return super(AssetReader, cls)._registry_keys() + as_list(cls.asset_mltype)

    def _read_stream(
            self,
            stream: io.BytesIO,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> Asset:
        return self._verify_asset(
            self._read_asset_with_retries(
                source=stream,
                storage=Storage.STREAM,
                **kwargs
            )
        )

    def _read_url(
            self,
            url: str,
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> Asset:
        return self._verify_asset(
            self._read_asset_with_retries(
                source=url,
                storage=Storage.URL,
                **kwargs
            )
        )

    def _read_local(
            self,
            local_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            **kwargs,
    ) -> Union[Asset, List[Asset]]:
        if is_list_like(local_path):
            return [
                self._verify_asset(asset)
                for asset in self._read_asset_multi(
                    source=local_path,
                    storage=Storage.LOCAL_FILE_SYSTEM,
                    **kwargs
                )
            ]
        return self._verify_asset(
            self._read_asset_with_retries(
                source=local_path,
                storage=Storage.LOCAL_FILE_SYSTEM,
                **kwargs
            )
        )

    def _read_s3(
            self,
            s3_path: Union[str, List[str]],
            file_contents: Optional[FileContents] = None,
            data_schema: Optional[MLTypeSchema] = None,
            files_to_ignore: List[str] = StringUtil.FILES_TO_IGNORE,
            **kwargs,
    ) -> Union[Asset, List[Asset]]:
        if is_list_like(s3_path):
            return [
                self._verify_asset(asset)
                for asset in self._read_asset_multi(
                    source=s3_path,
                    storage=Storage.S3,
                    **kwargs
                )
            ]
        return self._verify_asset(
            self._read_asset_with_retries(
                source=s3_path,
                storage=Storage.S3,
                **kwargs
            )
        )

    def _read_asset_multi(
            self,
            source: List[str],
            storage: Storage,
            file_contents: Optional[FileContents] = None,
            **kwargs
    ) -> List[Asset]:
        asset_futures: Dict[str, Asset] = {
            file_path: dispatch(
                self._read_asset_with_retries,
                parallelize=self.parallelize,
                source=file_path,
                storage=storage,
                postprocess=False,
                **kwargs,
            )
            for file_path in source
        }
        asset_futures: Dict[str, Asset] = accumulate(asset_futures)
        assets: List = []
        failed_read_file_paths: List[str] = []
        for file_path, asset_future in asset_futures.items():
            try:
                assets.append(accumulate(asset_future))
            except Exception as e:
                Log.error(f'Error reading from file "{file_path}":\n{format_exception_msg(e, short=True)}')
                failed_read_file_paths.append(file_path)

        if len(failed_read_file_paths) > 0:
            raise IOError(f'Could not read assets from the following paths:\n{sorted(list(failed_read_file_paths))}')
        return assets

    def _read_asset_with_retries(
            self,
            source: Union[str, io.BytesIO],
            storage: Storage,
            file_contents: Optional[FileContents] = None,
            **kwargs
    ) -> Asset:
        return retry(
            self._read_asset,
            retries=self.retry,
            wait=self.retry_wait,
            source=source,
            storage=storage,
            file_contents=file_contents,
            **kwargs,
        )

    @abstractmethod
    def _read_asset(
            self,
            source: Union[str, io.BytesIO],
            storage: Storage,
            file_contents: Optional[FileContents] = None,
            **kwargs
    ) -> Asset:
        pass

    def _verify_asset(self, asset: Asset) -> Asset:
        if not isinstance(asset, Asset):
            raise ValueError(
                f'Expected asset object to be of type {Asset.get_subclass(self.asset_mltype)}; '
                f'found value of type {type(asset)}.'
            )
        if self.return_as is not None and asset.layout is not SHORTHAND_TO_TENSOR_LAYOUT_MAP[self.return_as]:
            raise ValueError(
                f'Expected asset of type {asset.mltype} to have layout {SHORTHAND_TO_TENSOR_LAYOUT_MAP[self.return_as]}; '
                f'found data of type {type(asset)} with layout {asset.layout}'
            )
        return asset
