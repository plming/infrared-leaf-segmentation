import os
from array import array

import cv2
import jenkspy
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt, ndarray


def load_ir_in_dat(path: str) -> ndarray:
    assert os.path.splitext(path)[1] == '.dat'

    WIDTH = 160
    HEIGHT = 120

    with open(path, 'rb') as file:
        USHORT_BYTE = 2
        byte_stream = file.read(HEIGHT*WIDTH*USHORT_BYTE)

    ir_array = array('H', byte_stream)

    result = np.array(ir_array).reshape(HEIGHT, WIDTH)

    # TODO: fix magic number
    result = (result - 27315)/100
    return result


def load_ir_in_csv(path: str) -> ndarray:
    assert os.path.splitext(path)[1] == '.csv'

    WIDTH = 320
    result = genfromtxt(path, delimiter=',', skip_header=2, usecols=range(1, WIDTH + 1))
    assert result.shape == (320, 240)
    return result


def show_ir(ir: ndarray) -> None:
    plt.imshow(ir)
    plt.show()


def show_rgb(rgb: ndarray) -> None:
    plt.imshow(rgb)
    plt.show()


def get_excess_green(rgb: ndarray) -> ndarray:
    exg = np.zeros(shape=rgb.shape[:-1])

    for row, col, _channel in np.ndindex(rgb.shape):
        b, g, r = rgb[row][col]/rgb[row][col].sum()
        exg[row][col] = 2*g - r - b

    return exg


def load_rgb_in_jpeg(path: str) -> ndarray:
    WIDTH = 160
    HEIGHT = 120

    rgb_image = cv2.imread(path)
    rgb_image = cv2.resize(rgb_image,
                           dsize=(WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA)

    return rgb_image


def get_average_temperature(ir: ndarray, mask: ndarray) -> float:
    assert len(ir.shape) == 2 and len(mask.shape) == 2
    average_per_channels = cv2.mean(ir, mask)
    return average_per_channels[0]


def predict_with_jenks(arr: ndarray) -> ndarray:
    breaks = jenkspy.jenks_breaks(arr.ravel(), nb_class=2)
    return cv2.inRange(arr, lowerb=breaks[1], upperb=breaks[2])


"""
ir -> leaf -> temperature
"""
