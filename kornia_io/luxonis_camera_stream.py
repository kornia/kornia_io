"""Camera module supporting data streaming in PyTorch."""
from typing import Optional, Tuple

import depthai as dai
import torch
import numpy as np
import kornia as K

from .camera_stream import CameraStreamBase, CameraStreamBackend


class LuxonisCameraStream(CameraStreamBase):
    def __init__(self, camera_backend: CameraStreamBackend) -> None:
        self.camera_backend = camera_backend
        self._dai_device: Optional[dai.Device] = None

        self.open()  # start streaming

    def is_opened(self) -> bool:
        return self._dai_device is not None
    
    def close(self) -> None:
        # make sure we close the current stream
        if self._dai_device is not None:
            self._dai_device.close()

    def open(self) -> bool:
        self.close()

        # create a new stream
        if self.camera_backend == CameraStreamBackend.LUXONIS_OAKD_RGB:
            pipeline = dai.Pipeline()
            self.cam = pipeline.createColorCamera()
            self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            self.cam.setInterleaved(False)
            self.cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

            xout = pipeline.createXLinkOut()
            xout.setStreamName("rgb")

            self.cam.preview.link(xout.input)

            self._dai_device = dai.Device(pipeline)
            self._q = self._dai_device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        else:
            raise NotImplementedError(f"Camera type not supported: {self.backend}.")

        return True
    
    def get(self) -> torch.Tensor:
        frame: np.ndarray = self._q.get().getCvFrame()
        frame_t: torch.Tensor = K.utils.image_to_tensor(
            frame).to(self.device)
        if self._map_fn is not None:
            frame_t = self._map_fn(frame_t)
        return frame_t

    def set_size(self, size: Tuple[int, int]) -> None:
        """Set the image height and width."""
        height, width = size
        self.cam.setPreviewSize(width, height)  # width/height

    @property
    def width(self) -> int:
        """Return the video frame width."""
        return self.cam.getPreviewWidth()

    @property
    def height(self) -> int:
        """Return the video frame height."""
        return self.cam.getPreviewHeight()

    @property
    def fps(self) -> float:
        """Return the frame per seconds."""
        return self.cam.getFps()