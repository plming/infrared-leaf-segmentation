from numpy.typing import NDArray
import numpy as np
from numpy import uint8, bool8
from jenkspy import jenks_breaks
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
    def label(self) -> NDArray[bool8]:
        # region compute excess green
        exg = np.zeros(shape=self._rgb.shape[:-1])

        for row, col, _channel in np.ndindex(self._rgb.shape):
            rgb_sum = self._rgb[row][col].sum()

            if rgb_sum == 0:
                r, g, b = 0, 0, 0
            else:
                r, g, b = self._rgb[row][col] / rgb_sum

            exg[row][col] = 2 * g - r - b
        # endregion

        breaks = jenks_breaks(exg.ravel(), n_classes=2)
        result = np.logical_and(exg >= breaks[1], exg <= breaks[2])

        return result
