import time
from typing import List

from src.ir import load_ir_in_dat
from src.jenks_model import JenksModel
from src.metric import intersection_over_union, pixel_accuracy
from src.model import Model
from src.otsu_model import OtsuModel
from src.rgb_image import RgbImage

program_started = time.time()
ir = load_ir_in_dat("../ginseng-ir/irimage_20220512_0800.dat")
rgb = RgbImage("../ginseng-rgb/camimage_20220512_0800.jpg")

models: List[Model] = [JenksModel(),
                       OtsuModel(), ]

for model in models:
    model_started = time.time()
    print(f"[{model.__class__.__name__}]")

    predict = model.predict(ir)

    print(f"IOU: {intersection_over_union(predict, rgb.label) * 100:.2f}%")
    print(f"PA: {pixel_accuracy(predict, rgb.label) * 100:.2f}%")
    print(f"Elapsed: {time.time() - model_started:.2f}s")

print(f"Total Elapsed Time: {time.time() - program_started:.2f}s")
