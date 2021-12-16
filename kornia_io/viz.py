from typing import Optional

import torch
import visdom


class Visualizer(visdom.Visdom):
    def __init__(self) -> None:
        super().__init__(port=8097, raise_exceptions=True)

        if not self.check_connection():
            raise ConnectionError('Error connecting with the visdom server.')

    def show_image(self, window_name: str, image: torch.Tensor, opts: Optional[dict] = None) -> None:
        opts_dict = dict(title=window_name)
        if opts is not None:
            opts_dict.update(opts)

        if len(image.shape) == 4:
            self.images(image, win=window_name, opts=opts_dict)
        elif len(image.shape) in (2, 3,):
            self.image(image, win=window_name, opts=opts_dict)
        else:
            raise NotImplementedError(f"Unsupported image size. Got: {image.shape}.")