from jenkspy import jenks_breaks
from numpy import logical_and

from src.ir_image import IrImage
from src.model import Model


class JenksModel(Model):
    def predict(self, x: IrImage):
        x = x.image
        assert x.ndim == 2

        breaks = jenks_breaks(x.ravel(), n_classes=2)
        result = logical_and(x >= breaks[1], x <= breaks[2])

        return result.astype(bool)
