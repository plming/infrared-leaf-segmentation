import cv2
from numpy import uint8
from numpy.typing import NDArray

from src.rgb_image import RgbImage


class GinsengRgbImage(RgbImage):
    def __init__(self, path: str):
        super().__init__(path)
        self._rgb = GinsengRgbImage.__calibrate_image(self._rgb)

    @staticmethod
    def __calibrate_image(img: NDArray[uint8]) -> NDArray[uint8]:
        # region zoom in image
        zoom_ratio = 1.3  # 1.5
        img = cv2.resize(img,
                         dsize=None,
                         fx=zoom_ratio,
                         fy=zoom_ratio,
                         interpolation=cv2.INTER_CUBIC)
        # endregion
        # region crop image
        assert img.ndim == 3
        y, x, _ = img.shape

        WIDTH = 160
        HEIGHT = 120

        offset_x = 2
        offset_y = 7
        start_x = x // 2 - (WIDTH // 2) + offset_x
        start_y = y // 2 - (HEIGHT // 2) + offset_y
        # endregion

        return img[start_y:start_y + HEIGHT, start_x:start_x + WIDTH]
