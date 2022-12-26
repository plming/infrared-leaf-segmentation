from abc import ABC, abstractmethod

from numpy import bool8, uint8
from numpy.typing import NDArray


class Model(ABC):
    @abstractmethod
    def predict(self, x: NDArray[uint8]) -> NDArray[bool8]:
        pass
