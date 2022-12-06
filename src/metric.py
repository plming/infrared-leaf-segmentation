from numpy.typing import NDArray
import numpy as np
from numpy import float64, bool8


def pixel_accuracy(x: NDArray[bool8], y: NDArray[bool8]) -> float64:
    assert x.shape == y.shape
    assert x.dtype == bool8
    assert y.dtype == bool8

    matches = np.logical_and(x, y).sum()
    total = x.size
    assert matches <= total

    return matches / total


def intersection_over_union(x: NDArray[bool8], y: NDArray[bool8]) -> float64:
    assert x.shape == y.shape
    assert x.dtype == bool8
    assert y.dtype == bool8

    intersection = np.logical_and(x, y).sum()
    union = np.logical_or(x, y).sum()
    assert intersection <= union

    return intersection / union
