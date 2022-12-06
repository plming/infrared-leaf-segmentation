from cv2 import threshold, THRESH_BINARY, THRESH_OTSU

from src.ir_image import IrImage
from src.model import Model


class OtsuModel(Model):
    def predict(self, x: IrImage):
        _thresh, mask = threshold(x.image,
                                  thresh=None,
                                  maxval=255,
                                  type=THRESH_BINARY + THRESH_OTSU)

        return mask.astype(bool)
