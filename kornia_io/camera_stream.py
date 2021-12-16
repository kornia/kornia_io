from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch


class CameraStreamBackend(Enum):
    OPENCV = 0
    LUXONIS_OAKD_RGB = 1
    LUXONIS_OAKD_STEREO = 2
    LUXONIS_OAKD_DEPTH = 3


class CameraStreamBase(ABC):
    """Base class to represent a video stream camera."""
    @abstractmethod
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._map_fn: Optional[Callable] = None
        self._device: torch.device = torch.device('cpu')

    @abstractmethod
    def is_opened(self) -> bool:
        """Whether the camera stream is open."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the camera stream."""
        raise NotImplementedError

    @abstractmethod
    def get(self) -> torch.Tensor:
        """Retrieve a frame from the stream and cast to tensor."""
        raise NotImplementedError

    def map(self, fn: Callable) -> 'CameraStreamBase':
        """Set a callable to be applied after .get()"""
        self._map_fn = fn
        return self

    def to(self, device: torch.device) -> 'CameraStreamBase':
        """Set a torch.device to be applied after .get()"""
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        """Return the current torch device."""
        return self._device

    @property
    def meta(self) -> Dict[str, Any]:
        data = {}
        data['width'] = self.width
        data['height'] = self.height
        data['fps'] = self.fps
        return data

    @abstractproperty
    def width(self) -> int:
        """Return the video frame width."""
        raise NotImplementedError

    @abstractproperty
    def height(self) -> int:
        """Return the video frame height."""
        raise NotImplementedError

    @abstractproperty
    def fps(self) -> float:
        """Return the frame per seconds."""
        raise NotImplementedError