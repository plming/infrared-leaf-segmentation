from array import array
from typing import Any  # TODO: remove Any for static type checks
from numpy import genfromtxt
import numpy as np
import os


def load_ir_in_dat(path: str) -> np.ndarray[Any, Any]:
    assert os.path.splitext(path)[1] == '.dat'

    WIDTH = 160
    HEIGHT = 120

    with open(path, 'rb') as file:
        USHORT_BYTE = 2
        bytes = file.read(HEIGHT * WIDTH * USHORT_BYTE)

    ir_array = array('H', bytes)

    result = np.array(ir_array).reshape(HEIGHT, WIDTH)

    # TODO: fix magic number
    result = (result - 27315) / 100
    return result


def load_ir_in_csv(path: str) -> np.ndarray[Any, Any]:
    assert os.path.splitext(path)[1] == '.csv'

    WIDTH = 320
    return genfromtxt(path, delimiter=',',
                      skip_header=2, usecols=range(1, WIDTH+1))


def get_average_tempeature(ir: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> float:
    assert len(ir.shape) == 2
    assert len(mask.shape) == 2

    num_region_pixels = 0
    sum_temperature = 0
    for y, x in np.ndindex(ir.shape):
        if mask[y, x] != 0:
            sum_temperature += ir[y, x]
            num_region_pixels += 1

    return sum_temperature / num_region_pixels
