import cv2

from src.ir_image import IrImage
from src.model import Model


class OtsuModel(Model):
    def predict(self, x: IrImage):
        _thresh, mask = cv2.threshold(x.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.astype(bool)
