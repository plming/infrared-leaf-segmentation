import numpy as np
from numpy.typing import NDArray


def linear_scaling(x: NDArray, min_value: float, max_value: float) -> NDArray:
    result = (x - np.min(x)) * (max_value - min_value) / (np.max(x) - np.min(x)) + min_value
    assert (np.logical_and(result >= min_value, result <= max_value).all())

    return result


def standardization(x: NDArray) -> NDArray:
    return (x - x.mean()) / x.std()
