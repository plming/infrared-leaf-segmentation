from numpy import float64
from numpy.typing import NDArray

import src.util as util


class IrImage:
    def __init__(self, path: str):
        self.__ir = util.load_ir_in_dat(path)
        assert self.__ir.ndim == 2

    @property
    def image(self) -> NDArray[float64]:
        return self.__ir.copy()
