import numpy as np
from jenkspy import jenks_breaks
from numpy import uint8, bool_
from numpy.typing import NDArray

import src.util as util
from src.rgb import compute_excess_green


class RgbImage:
    def __init__(self, path: str):
        self._rgb = util.load_rgb_in_jpg(path)
        assert self._rgb.ndim == 3
        assert self._rgb.dtype == uint8

    @property
    def image(self) -> NDArray[uint8]:
        return self._rgb.copy()

    @property
    def label(self) -> NDArray[bool_]:
        exg = compute_excess_green(self._rgb)
        breaks = jenks_breaks(exg.ravel(), n_classes=2)
        result = np.logical_and(exg >= breaks[1], exg <= breaks[2])

        return result
