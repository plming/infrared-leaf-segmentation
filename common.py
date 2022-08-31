from array import array
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt


def load_ir_in_dat(filename: str) -> np.ndarray:
    WIDTH = 160
    HEIGHT = 120

    with open(filename, 'rb') as file:
        USHORT_BYTE = 2
        bytes = file.read(HEIGHT * WIDTH * USHORT_BYTE)

    ir_array = array('H')
    ir_array.frombytes(bytes)

    result = np.array(ir_array).reshape(HEIGHT, WIDTH)

    # TODO: remove magic number
    result = (result - 27315) / 100
    return result


def load_ir_in_csv(filename: str) -> np.ndarray:
    WIDTH = 320
    return genfromtxt(filename, delimiter=',',
                      skip_header=2, usecols=range(1, WIDTH+1))


def show_image(img, title="") -> None:
    plt.imshow(img)
    plt.title(title)
    plt.show()


def get_average_tempeature(ir: np.ndarray, mask: np.ndarray) -> float:
    assert len(ir.shape) == 2
    assert len(mask.shape) == 2

    num_region_pixels = 0
    sum_temperature = 0
    for y, x in np.ndindex(ir.shape):
        if mask[y, x] != 0:
            sum_temperature += ir[y, x]
            num_region_pixels += 1

    return sum_temperature / num_region_pixels


def show_histogram(single_channel_img: np.ndarray, title: str) -> None:
    plt.hist(single_channel_img.ravel(), bins='auto')
    plt.title(title)
    plt.show()
