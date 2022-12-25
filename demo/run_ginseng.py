import time
from typing import List

from src.ir_image import IrImage
from src.jenks_model import JenksModel
from src.metric import intersection_over_union, pixel_accuracy
from src.model import Model
from src.otsu_model import OtsuModel
from src.rgb_image import RgbImage

start = time.time()
ir = IrImage("../ginseng-ir/irimage_20220512_0800.dat")
rgb = RgbImage("../ginseng-rgb/camimage_20220512_0800.jpg")

models: List[Model] = [JenksModel(),
                       OtsuModel(), ]

for model in models:
    print(f"[{model.__class__.__name__}]")

    predict = model.predict(ir)

    print(f"IOU: {intersection_over_union(predict, rgb.label) * 100:.2f}%")
    print(f"PA: {pixel_accuracy(predict, rgb.label) * 100:.2f}%")

print(f"Time: {time.time() - start:.2f}s")
