import numpy as np
from jenkspy import jenks_breaks
from numpy import uint8, bool_
from numpy.typing import NDArray

import src.util as util


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
        WEIGHTS_PER_CHANNEL = np.array([-1, 2, -1])
        weighted_sum_per_pixel = np.sum(self._rgb * WEIGHTS_PER_CHANNEL, axis=2)

        # replace 0 with 1 to avoid division by 0
        sum_per_pixel = np.sum(self._rgb, axis=2)
        sum_per_pixel[sum_per_pixel == 0] = 1

        exg = weighted_sum_per_pixel / sum_per_pixel

        breaks = jenks_breaks(exg.ravel(), n_classes=2)
        result = np.logical_and(exg >= breaks[1], exg <= breaks[2])

        return result
