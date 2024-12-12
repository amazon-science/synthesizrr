from typing import *
from abc import abstractmethod, ABC
import io, numpy as np
from synthergent.base.constants import FileContents, MLType, FileFormat, DataLayout, Storage, SHORTHAND_TO_TENSOR_LAYOUT_MAP
from synthergent.base.util import is_list_like, StringUtil, Parameters
from synthergent.base.data.reader.asset.AssetReader import AssetReader
from synthergent.base.data.asset import Image
from pydantic import constr, conint
from pydantic.typing import Literal


class ImageReader(AssetReader, ABC):
    asset_mltype = MLType.IMAGE

    ## Whether to put the color channels first, i.e. get a 512x512 image as an array/tensor of shape (3, 512, 512)
    channels: Literal['first', 'last'] = 'first'

    def _read_asset(
            self,
            source: Union[str, io.BytesIO],
            **kwargs
    ) -> Image:
        return self._read_image(source=source, **kwargs)

    @abstractmethod
    def _read_image(
            self,
            source: Union[str, io.BytesIO],
            storage: Storage,
            file_contents: Optional[FileContents] = None,
            **kwargs
    ) -> Image:
        pass
