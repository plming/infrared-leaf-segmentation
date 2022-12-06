from abc import ABC, abstractmethod

from numpy import bool_
from numpy.typing import NDArray

from src.ir_image import IrImage


class Model(ABC):
    @abstractmethod
    def predict(self, x: IrImage) -> NDArray[bool_]:
        pass
