import numpy as np
from numpy import float_
from numpy.typing import NDArray


def compute_excess_green(img: NDArray) -> NDArray[float_]:
    WEIGHTS_PER_CHANNEL = np.array([-1, 2, -1])

    weighted_sum_per_pixel = np.sum(img * WEIGHTS_PER_CHANNEL, axis=2)

    # replace 0 with 1 to avoid division by 0
    sum_per_pixel = np.sum(img, axis=2)
    sum_per_pixel[sum_per_pixel == 0] = 1

    return weighted_sum_per_pixel / sum_per_pixel
