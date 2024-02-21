from typing import *
from abc import abstractmethod, ABC
import io, numpy as np
from synthesizrr.base.constants import FileContents, MLType, FileFormat, Parallelize, Storage
from synthesizrr.base.util import is_list_like, StringUtil, FileSystemUtil, run_concurrent, run_parallel, run_parallel_ray, \
    accumulate, optional_dependency
from synthesizrr.base.util.aws import S3Util
from synthesizrr.base.data.reader.asset.audio.AudioReader import AudioReader
from pydantic import constr
from pydantic.typing import Literal

with optional_dependency('torchaudio'):
    import torchaudio as ta


    class TorchAudioReader(AudioReader):
        ## Subset of formats supported by imageio:
        file_formats = [
            FileFormat.PNG,
            FileFormat.JPEG,
            FileFormat.BMP,
            FileFormat.GIF,
            FileFormat.ICO,
            FileFormat.WEBP,
        ]

        class Params(ImageReader.Params):
            mode: constr(min_length=1, max_length=6, strip_whitespace=True) = 'RGB'

        def _read_image(
                self,
                source: Union[str, io.BytesIO],
                storage: Storage,
                file_contents: Optional[FileContents] = None,
                postprocess: bool = True,
                **kwargs
        ) -> np.ndarray:
            if storage is Storage.S3:
                source: io.BytesIO = io.BytesIO(S3Util.stream_s3_object(source).read())
            img: np.ndarray = iio.imread(
                source,
                **self.params.dict(),
            )
            if not postprocess:
                return img
            return self._postprocess_image(img, **kwargs)


    class TIFFImageIOReader(ImageIOReader):
        ## Subset of formats supported by imageio:
        file_formats = [
            FileFormat.TIFF,
        ]

        class Params(ImageReader.Params):
            mode: Literal['r'] = 'r'  ## In imageio's tifffile plugin, mode is 'r' or 'w'

    # def fetch_img_imageio(img_path: str):
    #     storage = FileMetadata.detect_storage(img_path)
    #     if storage is Storage.LOCAL_FILE_SYSTEM:
    #         img_np = iio.imread(
    #             img_path,
    #             mode="RGB"
    #         )
    #     elif storage is Storage.S3:
    #         img_np = iio.imread(
    #             io.BytesIO(S3Util.stream_s3_object(img_path).read()),
    #             mode="RGB"
    #         )
    #     return img_np
    #
    #
    # def np_img_transform(img: np.ndarray, shared_memory=True) -> torch.Tensor:
    #     img: np.ndarray = cv2_resize(img)
    #     img: torch.Tensor = transform(torch.from_numpy(np.moveaxis(img, -1, 0)))
    #     if shared_memory:
    #         img: torch.Tensor = img.share_memory_()
    #     return img
    #
    #
    # def process_task_data_load_imgs_imageio_concurrent(task_data) -> Dataset:
    #     # global task_data_global
    #     # task_data_global.append(task_data)
    #     try:
    #         imgs = task_data.data['First-Image'].apply(
    #             lambda img_path: run_concurrent(fetch_img_imageio, img_path)
    #         ).apply(accumulate).apply(np_img_transform)
    #         if task_data.data.layout is DataLayout.DICT:
    #             imgs = imgs.as_torch()
    #         task_data.data['img'] = imgs
    #         task_data.data_schema.other_schema['img'] = MLType.IMAGE
    #     except Exception as e:
    #         print(f'Failed for data with ids: {task_data.data["id"].tolist()}')
    #         raise e
    #     return task_data
