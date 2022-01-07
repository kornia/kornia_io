from typing import Tuple, Union
from pathlib import Path

import cv2
import numpy as np

from .image import Image, ImageColor


class VideoStreamWriter:
    def __init__(self, file_path: str, fps: float, size: Tuple[int, int]) -> None:
        self._fps = fps
        self._size = size  # should come in h/w for later checks
        self.writer: cv2.VideoWriter = self._create_writer(Path(file_path), fps, size[::-1])

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def is_opened(self) -> bool:
        return self.writer.isOpened()

    @staticmethod
    def _create_writer(file_path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
        if file_path.suffix == '.avi':
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        else:
            raise ValueError(f"Unsupported video format yet - use [.avi,]. Got: {file_path.suffix}.")
        return cv2.VideoWriter(str(file_path), fourcc, fps, size)

    def _check_valid(self, shape: Tuple[int, int]) -> bool:
        if shape != self._size:
            return False
        return True

    def append(self, frame: Union[Image, np.ndarray]) -> bool:
        if isinstance(frame, Image):
            frame_out = frame.convert(ImageColor.BGR).to_numpy()
        elif isinstance(frame, np.ndarray):
            # NOTE: user must be aware that of what colorspace has the image
            frame_out = frame
        if not self._check_valid(frame_out.shape[:2]):
            raise ValueError(
                f"Invalid frame size. Got: {frame_out.shape[:2]}. Expected: {self._size}."
            )
        return self.writer.write(frame_out.astype('uint8'))
    
    def close(self):
        self.writer.release()
        