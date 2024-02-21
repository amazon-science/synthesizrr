from typing import *
from abc import abstractmethod, ABC
import io, numpy as np
from synthesizrr.base.constants import FileContents, MLType, FileFormat, SHORTHAND_TO_TENSOR_LAYOUT_MAP, DataLayout
from synthesizrr.base.util import is_list_like, StringUtil, run_concurrent, run_parallel, run_parallel_ray, accumulate
from synthesizrr.base.data.reader.asset.AssetReader import AssetReader
from pydantic import constr
from pydantic.typing import Literal


class AudioReader(AssetReader, ABC):
    asset_mltype = MLType.AUDIO

    def _postprocess_asset(
            self,
            asset: Any,
            **kwargs
    ) -> Any:
        asset: Any = super()._postprocess_asset(asset, **kwargs)
        if SHORTHAND_TO_TENSOR_LAYOUT_MAP[self.return_as] is DataLayout.NUMPY:
            if self.channels_first:
                asset: np.ndarray = np.moveaxis(asset, -1, 0)
        return asset
