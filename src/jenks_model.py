from jenkspy import jenks_breaks
from numpy import logical_and

from src.model import Model


class JenksModel(Model):
    def predict(self, x):
        breaks = jenks_breaks(x.ravel(), n_classes=2)
        result = logical_and(x >= breaks[1], x <= breaks[2])

        return result.astype(bool)
