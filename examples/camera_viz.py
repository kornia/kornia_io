
import argparse

import cv2
import numpy as np

import torch
import torch.nn as nn

import kornia as K
from kornia_io import CameraStream, CameraStreamBackend, Visualizer


def my_app(args):
    # select the device
    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True

    # create the video capture object
    #cap = CameraStream.create(CameraStreamBackend.LUXONIS_OAKD_RGB)
    cap = CameraStream.create(CameraStreamBackend.OPENCV)
    print(f"Before resize: {cap.meta}")

    # create the visualizer object
    viz = Visualizer()

    # compute scale
    scale: float = 1. * args.image_size / cap.meta['width']
    w, h = int(cap.meta['width'] * scale), int(cap.meta['height'] * scale)
    cap.set_size((h, w))
    print(f"After resize: {cap.meta}")

    # set set some preprocess transforms
    preprocess = nn.Sequential(
        K.contrib.Lambda(lambda x: x[None].float()),
        K.color.BgrToRgb(),
    )

    model = nn.Sequential(
        K.color.RgbToGrayscale(),
        K.filters.Sobel(normalized=False),
    )

    postproces = nn.Sequential(
        K.contrib.Lambda(lambda x: K.tensor_to_image(x.byte()).copy()),
    )

    # set the preprocess to the video stream and device
    cap = cap.map(preprocess).to(device)

    while(True):

        # Capture the video frame by frame
        frame: torch.Tensor = cap.get()

        start = cv2.getTickCount()

        frame_out = model(frame)
        frame_vis: np.ndarray = postproces(frame_out)  # to visualize

        fps: float = cv2.getTickFrequency() / (cv2.getTickCount() - start)

        # show image

        frame_vis = cv2.putText(
            frame_vis, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # Display the resulting frame
        viz.show_image('frame', frame)
        viz.show_image('frame_out', frame_vis)

    # After the loop release the cap and writing objects
    cap.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--image_size', default=640, type=int, help='the image size to process.')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    args = parser.parse_args()
    my_app(args)