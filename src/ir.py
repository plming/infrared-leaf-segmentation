import os
from array import array

import numpy as np
from numpy import uint8
from numpy.typing import NDArray

from src.normalization import linear_scaling


def load_ir_in_dat(path: str) -> NDArray[uint8]:
    assert os.path.splitext(path)[-1] == '.dat'

    WIDTH = 160
    HEIGHT = 120

    with open(path, 'rb') as file:
        USHORT_BYTE = 2
        byte_stream = file.read(HEIGHT * WIDTH * USHORT_BYTE)

    ir_array = array('H', byte_stream)

    result = np.array(ir_array).reshape(HEIGHT, WIDTH)
    result = linear_scaling(result, 0, 255)

    return result.astype(np.uint8)
