#!/usr/bin/env python3
import cv2

import torch.nn as nn
import kornia as K
import kornia_io as IO
import torchvision

# load a pretrained model for classification and remove the classifier
model = torchvision.models.resnet50(pretrained=True)
model = nn.Sequential(*(list(model.children())[:-6]))

# create the vision pipeline
pipe = nn.Sequential(
    K.contrib.Lambda(lambda x: x.float()),
    K.contrib.Lambda(lambda x: x.unsqueeze(0) if len(x.shape) == 3 else x),
    # K.geometry.Resize(224),
    model,
)

cam = IO.CameraStream.create(IO.CameraStreamBackend.LUXONIS_OAKD_RGB)
assert cam.is_opened(), f"Please, make sure the camera is connected."

# NOTE: read the instruction below before using.
# - .map() will be called in the host and must be used with .get()
# - .upload() will compile in the VPU and run the sequential in the camera.
# Upload must be used with .get_data() passing the shape to reshape since it returns
# a flat vector of data. Use first .map() to debug and know the shape.
use_vpu: bool = True

if not use_vpu:
    cam = cam.map(pipe)  # this is will run in the host
else:
    cam = cam.upload(pipe)  # this is actualy compiling with onnx/myriad

viz = IO.Visualizer()

while cam.is_opened():
    start = cv2.getTickCount()

    # this comes already in [0,1] - torch.float32
    if not use_vpu:
        frame = cam.get()
    else:
        frame = cam.get_data()
        # NOTE: you need to debug to figure out the shape
        frame = frame.view((1, 64, 75, 75))

    fps: float = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    print(fps)

    for i in range(8):
        frame_i = K.enhance.normalize_min_max(frame[:, i:i+1])
        viz.show_image(f"feat_{i}", frame_i)

cam.close()
