import os

import cv2
import numpy as np
from numpy import genfromtxt, ndarray
from numpy.typing import NDArray


def load_ir_in_csv(path: str) -> NDArray[np.float64]:
    assert os.path.splitext(path)[1] == '.csv'

    WIDTH = 320
    HEIGHT = 240

    result = genfromtxt(path,
                        delimiter=',',
                        skip_header=2,
                        usecols=range(1, WIDTH + 1))

    assert result.shape == (HEIGHT, WIDTH)
    return result


def load_rgb_in_jpg(path: str) -> ndarray:
    WIDTH = 160
    HEIGHT = 120

    rgb_image = cv2.imread(path)
    rgb_image = cv2.resize(rgb_image,
                           dsize=(WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA)

    return rgb_image


def get_max_temperature(ir: ndarray, mask: ndarray) -> float:
    assert ir.shape == mask.shape and ir.ndim == 2
    assert mask.dtype == np.bool8

    return np.max(ir[mask])


def get_min_temperature(ir: ndarray, mask: ndarray) -> float:
    assert ir.shape == mask.shape and ir.ndim == 2
    assert mask.dtype == np.bool8

    return np.min(ir[mask])


def get_average_temperature(ir: ndarray, mask: ndarray) -> float:
    assert ir.shape == mask.shape and ir.ndim == 2
    assert mask.dtype == np.bool8

    return float(np.mean(ir[mask]))


def get_masked_image(rgb: ndarray, mask: ndarray) -> ndarray:
    # FIXME: not works. think about how to apply 2d mask at rgb(3d) image
    assert rgb.ndim == 3
    assert rgb.dtype == np.uint8
    assert mask.ndim == 2
    assert mask.dtype == np.bool8

    return rgb[mask]


def crop_img_from_center(img, offset_x, offset_y, cropx, cropy):
    assert img.ndim == 3

    y, x, _ = img.shape
    start_x = x // 2 - (cropx // 2) + offset_x
    start_y = y // 2 - (cropy // 2) + offset_y

    # clamp start_x and start_y in positive range
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    return img[start_y:start_y + cropy, start_x:start_x + cropx]
