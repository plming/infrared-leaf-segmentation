from model import Model
from numpy import logical_and
from jenkspy import jenks_breaks


class JenksModel(Model):
    def predict(self, x):
        assert x.ndim == 2

        breaks = jenks_breaks(x.ravel(), nb_class=2)
        result = logical_and(x >= breaks[1], x <= breaks[2])

        return result.astype(bool)
