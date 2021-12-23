from typing import Optional, Union

import visdom
import numpy as np
from torch import Tensor

from .image import Image


class Visualizer(visdom.Visdom):
    def __init__(self, port: int = 8097) -> None:
        super().__init__(port=port, raise_exceptions=True)
        self._port = port

        if not self.check_connection():
            raise ConnectionError(
                f"Error connecting with the visdom server. Run in your termnal: visdom.")

    def show_image(self, window_name: str, image: Union[Image, Tensor, np.ndarray], opts: Optional[dict] = None) -> None:
        opts_dict = dict(title=window_name)
        if opts is not None:
            opts_dict.update(opts)

        if isinstance(image, np.ndarray):
            image = Image.from_numpy(image)

        if len(image.shape) == 4:
            self.images(image, win=window_name, opts=opts_dict)
        elif len(image.shape) in (2, 3,):
            self.image(image, win=window_name, opts=opts_dict)
        else:
            raise NotImplementedError(f"Unsupported image size. Got: {image.shape}.")