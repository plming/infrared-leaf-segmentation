import cv2
import numpy as np
from jenkspy import jenks_breaks
from src.metric import intersection_over_union, pixel_accuracy

import src.util as util
import src.visualization as visualization
from src.ginseng_rgb_image import GinsengRgbImage
from src.ir_image import IrImage
from src.jenks_model import JenksModel

ir = IrImage("../ginseng-ir/irimage_20220512_0800.dat")
rgb = GinsengRgbImage("../ginseng-rgb/camimage_20220512_0800.jpg")
model = JenksModel()

# get preprocessed ir image
# preprocessed = cv2.GaussianBlur(ir.image, (5, 5), 0)
preprocessed = cv2.medianBlur(ir.image, 5)

predicted = model.predict(ir)
print("Without preprocessing")
print(f"pa: {pixel_accuracy(predicted, rgb.label) * 100:.2f}")
print(f"iou: {intersection_over_union(predicted, rgb.label) * 100:.2f}")
visualization.show_image(ir.image, "without preprocessing")
from jenkspy import jenks_breaks

breaks = jenks_breaks(preprocessed.flatten(), n_classes=2)
predicted = np.logical_and(preprocessed >= breaks[1], preprocessed <= breaks[2]).astype(np.bool_)
print("With preprocessing")
print(f"pa: {pixel_accuracy(predicted, rgb.label) * 100:.2f}")
print(f"iou: {intersection_over_union(predicted, rgb.label) * 100:.2f}")
visualization.show_image(preprocessed, "with preprocessing")
