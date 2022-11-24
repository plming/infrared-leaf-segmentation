import os
from array import array
from typing import Optional

import cv2
import jenkspy
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from numpy import genfromtxt, ndarray


def load_ir_in_dat(path: str) -> NDArray:
    assert os.path.splitext(path)[-1] == '.dat'

    WIDTH = 160
    HEIGHT = 120

    with open(path, 'rb') as file:
        USHORT_BYTE = 2
        byte_stream = file.read(HEIGHT*WIDTH*USHORT_BYTE)

    ir_array = array('H', byte_stream)

    result: ndarray = np.array(ir_array).reshape(HEIGHT, WIDTH)

    divisor = np.max(result) - np.min(result)
    result = (result - np.min(result))/divisor*255
    result = result.astype(np.uint8)
    return result


def load_ir_in_csv(path: str) -> NDArray[np.float64]:
    assert os.path.splitext(path)[1] == '.csv'

    WIDTH = 320
    HEIGHT = 240

    result = genfromtxt(path, delimiter=',', skip_header=2,
                        usecols=range(1, WIDTH + 1))
    assert result.shape == (HEIGHT, WIDTH)
    return result


def show_images(*images: ndarray) -> None:
    for image in images:
        dimension = image.ndim

        if dimension == 2:
            plt.imshow(image, cmap='plasma')
        elif dimension == 3:
            plt.imshow(image)
        else:
            assert False, "Unknown dimension"

        plt.axis(False)
        plt.show()


def show_histogram(single_channel_image: ndarray, title: Optional[str] = None) -> None:
    assert single_channel_image.ndim == 2
    assert single_channel_image.dtype == np.float64

    plt.hist(single_channel_image.ravel(), bins=256)
    if title is not None:
        plt.title(title)
    plt.show()


def get_excess_green(rgb: ndarray) -> ndarray:
    assert rgb.ndim == 3
    assert rgb.dtype == np.uint8

    exg = np.zeros(shape=rgb.shape[:-1])

    for row, col, _channel in np.ndindex(rgb.shape):
        rgb_sum = rgb[row][col].sum()

        if rgb_sum == 0:
            r, g, b = 0, 0, 0
        else:
            r, g, b = rgb[row][col]/rgb_sum

        exg[row][col] = 2*g - r - b

    return exg


def load_rgb_in_jpeg(path: str) -> ndarray:
    WIDTH = 160
    HEIGHT = 120

    rgb_image = cv2.imread(path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image,
                           dsize=(WIDTH, HEIGHT),
                           interpolation=cv2.INTER_AREA)

    return rgb_image


def get_max_temperature(ir: ndarray, mask: ndarray) -> float:
    assert ir.shape == mask.shape and ir.ndim == 2
    assert ir.dtype == np.float64 and mask.dtype == np.bool8

    return np.max(ir[mask])


def get_min_temperature(ir: ndarray, mask: ndarray) -> float:
    assert ir.shape == mask.shape and ir.ndim == 2
    assert ir.dtype == np.float64 and mask.dtype == np.bool8

    return np.min(ir[mask])


def get_average_temperature(ir: ndarray, mask: ndarray) -> float:
    assert ir.shape == mask.shape and ir.ndim == 2
    assert ir.dtype == np.float64 and mask.dtype == np.bool8

    return np.mean(ir[mask])


def get_leaf_by_jenks(image: ndarray) -> ndarray:
    assert image.ndim == 2
    assert image.dtype == np.float64

    breaks = jenkspy.jenks_breaks(image.ravel(), nb_class=2)
    result = np.logical_and(image >= breaks[1],
                            image <= breaks[2])

    return result


def get_leaf_by_kmeans_with_coordination(ir: ndarray) -> ndarray:
    assert ir.ndim == 2
    assert ir.dtype == np.float64

    # region add indices as feature
    array_3d = np.zeros(shape=(ir.shape[0], ir.shape[1], 3))

    for row in range(ir.shape[0]):
        for col in range(ir.shape[1]):
            array_3d[row][col][0] = ir[row][col]
            array_3d[row][col][1] = row
            array_3d[row][col][2] = col
    # endregion

    x = array_3d.reshape(-1, 3)
    # region max-min normalization
    x[:, 0] -= x[:, 0].min()
    x[:, 0] /= x[:, 0].max() - x[:, 0].min()

    x[:, 1] -= x[:, 1].min()
    x[:, 1] /= x[:, 1].max() - x[:, 1].min()

    x[:, 2] -= x[:, 2].min()
    x[:, 2] /= x[:, 2].max() - x[:, 2].min()
    # endregion

    labels = KMeans(n_clusters=2).fit_predict(x).astype(np.bool8)

    return np.reshape(labels,
                      newshape=(ir.shape[0], ir.shape[1]))


def get_masked_image(rgb: ndarray, mask: ndarray) -> ndarray:
    # FIXME: not works. think about how to apply 2d mask at rgb(3d) image
    assert rgb.ndim == 3
    assert rgb.dtype == np.uint8
    assert mask.ndim == 2
    assert mask.dtype == np.bool8

    return rgb[mask]


def get_intersection_over_union(target: ndarray, predict: ndarray) -> float:
    assert target.shape == predict.shape and target.ndim == 2
    assert target.dtype == np.bool8 and predict.dtype == np.bool8

    intersection = np.logical_and(target, predict).sum()
    union = np.logical_or(target, predict).sum()
    assert intersection <= union

    return intersection/union
