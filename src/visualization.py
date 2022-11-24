from typing import Optional

import matplotlib.pyplot as plt
from numpy import ndarray
from numpy.typing import NDArray


def show_image(image: NDArray, title: str) -> None:
    if image.ndim == 2:
        plt.imshow(image, cmap='plasma')
    elif image.ndim == 3:
        plt.imshow(image)
    else:
        assert False, "Unknown dimension"

    plt.title(title)
    plt.axis(False)
    plt.show()


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

    plt.hist(single_channel_image.ravel(), bins=256)
    if title is not None:
        plt.title(title)
    plt.show()
