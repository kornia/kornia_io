from enum import Enum

import cv2  # for reader
import numpy as np

from torch import Tensor
import kornia as K
from kornia.color import rgb_to_grayscale, bgr_to_grayscale

# TODO: make it flexible
__image_reader__ = cv2.imread


class ImageColor(Enum):
    GRAY = 0
    RGB = 1
    BGR = 2


class Image(Tensor):
    @staticmethod 
    def __new__(cls, data, color, *args, **kwargs): 
        return super().__new__(cls, data, *args, **kwargs) 

    def __init__(self, data: Tensor, color: ImageColor) -> None:
        self._color = color

    @property
    def is_batch(self) -> bool:
        return len(self.data.shape) > 3

    @property
    def channels(self) -> int:
        return self.data.shape[-3]

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def color(self) -> ImageColor:
        return self._color

    @classmethod
    def from_tensor(cls, data: Tensor, color: ImageColor = ImageColor.RGB) -> 'Image':
        return cls(data, color)

    @classmethod
    def from_file(cls, file_path: str) -> 'Image':
        data: np.ndarray = __image_reader__(file_path)
        data_t: Tensor = K.utils.image_to_tensor(data)
        data_t = K.color.bgr_to_rgb(data_t)
        # TODO: discuss whether we return normalised
        data_t = data_t.float() / 255.
        return cls(data_t, ImageColor.RGB)

    def _to_grayscale(self, data: Tensor) -> Tensor:
        if self.color == ImageColor.GRAY:
            out = data
        elif self.color == ImageColor.RGB:
            out = rgb_to_grayscale(data)
        elif self.color == ImageColor.BGR:
            out = bgr_to_grayscale(data)
        else:
            raise NotImplementedError(f"Unsupported color: {self.color}.")
        return out

    def grayscale(self) -> 'Image':
        gray = self._to_grayscale(self.data)
        return Image(gray, ImageColor.GRAY)

    def grayscale_(self) -> 'Image':
        self.data = self._to_grayscale(self.data)
        self._color = ImageColor.GRAY
        return self

    # TODO: add the methods we need
    # - lab, hsv, ...
    # - erode, dilate, ...


