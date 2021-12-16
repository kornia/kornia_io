from enum import Enum

import torch

# TODO: define in another place to avoid circular dependencies
from .camera_stream import CameraStreamBase, CameraStreamBackend
from .opencv_camera_stream import OpencvCameraStream
from .luxonis_camera_stream import LuxonisCameraStream


class CameraStream:
    @staticmethod
    def create(backend: CameraStreamBackend) -> CameraStreamBase:
        if backend == CameraStreamBackend.OPENCV:
            return OpencvCameraStream.from_index(0)
        elif backend == CameraStreamBackend.LUXONIS_OAKD_RGB:
            return LuxonisCameraStream(backend)
        else:
            raise NotImplementedError(f"Unsupported backend image.")
