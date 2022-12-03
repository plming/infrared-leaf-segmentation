from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Model(ABC):
    @abstractmethod
    def predict(self, x: NDArray[float]) -> NDArray[bool]:
        pass
