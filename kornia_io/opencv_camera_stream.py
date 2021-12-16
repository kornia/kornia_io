from typing import Any, Tuple

import cv2
import torch
import kornia as K

from .camera_stream import CameraStreamBase


class OpencvCameraStream(CameraStreamBase):
    def __init__(self, stream: cv2.VideoCapture) -> None:
        super().__init__(stream)

    def get(self) -> torch.Tensor:
        _, frame = self._stream.read()
        frame_t: torch.Tensor = K.utils.image_to_tensor(
            frame).to(self.device)
        if self._map_fn is not None:
            frame_t = self._map_fn(frame_t)
        return frame_t

    def is_opened(self) -> bool:
        return self._stream.isOpened()

    def close(self) -> None:
        self._stream.release()

    @classmethod
    def from_index(cls, camera_index: int) -> 'OpencvCameraStream':
        cap = cv2.VideoCapture(camera_index)
        return cls(cap)

    def set_size(self, size: Tuple[int, int]) -> None:
        """Set the image height and width."""
        height, width = size
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def width(self) -> int:
        """Return the video frame width."""
        return int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Return the video frame height."""
        return int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Return the frame per seconds."""
        return self._stream.get(cv2.CAP_PROP_FPS)

