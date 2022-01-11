"""Camera module supporting data streaming in PyTorch."""
from typing import List, Optional, Tuple

import depthai as dai
import torch
import numpy as np
import kornia as K

# compile to myriad through openvino
import onnx
from onnxsim import simplify
import blobconverter

from .image import Image, ImageColor
from .camera_stream import CameraStreamBase, CameraStreamBackend


class LuxonisCameraStream(CameraStreamBase):
    def __init__(self, camera_backend: CameraStreamBackend) -> None:
        super().__init__(self)
        self.camera_backend = camera_backend
        self._dai_device: Optional[dai.Device] = None

        self.config = None
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

            xout_d = pipeline.createXLinkOut()
            xout_d.setStreamName("data")

            if self.config is not None:
                nn = pipeline.createNeuralNetwork()
                nn.setBlobPath(self.config)
                self.cam.preview.link(nn.input)
                nn.out.link(xout_d.input)
            else:
                self.cam.preview.link(xout.input)

            self._dai_device = dai.Device(pipeline)
            self._q = self._dai_device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            self._d = self._dai_device.getOutputQueue(name="data", maxSize=1, blocking=False)
        else:
            raise NotImplementedError(f"Camera type not supported: {self.backend}.")

        return True
    
    def get(self) -> Image:
        frame: np.ndarray = self._q.get().getCvFrame()
        frame_t: torch.Tensor = K.utils.image_to_tensor(
            frame).to(self.device, torch.float32)
        frame_img = Image.from_tensor(frame_t, ImageColor.BGR).convert(ImageColor.RGB)
        if self._map_fn is not None:
            frame_img = self._map_fn(frame_img)
        return frame_img

    def get_data(self) -> torch.Tensor:
        data: List[float] = self._d.get().getFirstLayerFp16()
        return torch.tensor(data)

    def upload(self, input_pipe: torch.nn.Module) -> bool:
        # convert to onnx
        input_data = torch.ones(1, 3, *self.resolution)
        torch.onnx.export(
            input_pipe,
            input_data, # Dummy input for shape
            "model.onnx",
            opset_version=12,
            do_constant_folding=True,
        )
        # simplify the onnx model
        onnx_model = onnx.load("model.onnx")
        model_simpified, check = simplify(onnx_model)
        onnx.save(model_simpified, "model_sim.onnx")
        # use blobxonverter -> .blob
        blobconverter.from_onnx(
            model="model_sim.onnx",
            output_dir="model",
            data_type="FP16",
            shaves=6,
            use_cache=False,
            optimizer_params=[]
        )
        # create camera pipeline
        self.config = "model/model_sim_openvino_2021.4_6shave.blob"
        assert self.open()
        return self


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