import src.visualization as visualization
from src.metric import intersection_over_union, pixel_accuracy
from src.ginseng_rgb_image import GinsengRgbImage
from src.ir_image import IrImage
from src.jenks_model import JenksModel

ir = IrImage("./ginseng-ir/irimage_20220512_0800.dat")
rgb = GinsengRgbImage("./ginseng-rgb/camimage_20220512_0800.jpg")
model = JenksModel()

print(f"정확도(IOU, %): {intersection_over_union(model.predict(ir), rgb.label) * 100:.2f}")
print(f"정확도(PA, %): {pixel_accuracy(model.predict(ir), rgb.label) * 100:.2f}")

visualization.show_image(ir.image, "ir")
visualization.show_image(rgb.image, "rgb")
visualization.show_image(model.predict(ir), "predicted")
