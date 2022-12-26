from cv2 import threshold, THRESH_BINARY, THRESH_OTSU

from src.model import Model


class OtsuModel(Model):
    def predict(self, x):
        _thresh, mask = threshold(src=x,
                                  thresh=None,
                                  maxval=255,
                                  type=THRESH_BINARY + THRESH_OTSU)

        return mask.astype(bool)
