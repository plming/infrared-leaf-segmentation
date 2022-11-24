import cv2
import numpy as np

import visualization
import util

# region load images
ir_image = util.load_ir_in_dat("./ginseng-ir/irimage_20220728_0903.dat")
rgb_image = util.load_rgb_in_jpeg("./ginseng-rgb/1.jpeg")
# endregion

exg = util.get_excess_green(rgb_image)
label = util.get_leaf_by_jenks(exg)
predict = util.get_leaf_by_jenks(ir_image)

visualization.show_image(ir_image, "IR image")
visualization.show_image(rgb_image, "RGB image")
visualization.show_histogram(ir_image, "IR histogram")
visualization.show_image(exg, "EXG image")
visualization.show_image(label, "Label leaf region")
visualization.show_image(predict, "Predicted leaf region")

accuracy = util.get_intersection_over_union(label, predict)
print(f"정확도(IOU, %): {accuracy*100:.2f}")

max_temperature = util.get_max_temperature(ir_image, predict)
average_temperature = util.get_average_temperature(ir_image, predict)
min_temperature = util.get_min_temperature(ir_image, predict)

# TODO: not works because pixel's range in byte
print(f"잎의 최고 온도(C): {max_temperature:.2f}")
print(f"잎의 평균 온도(C): {average_temperature:.2f}")
print(f"잎의 최저 온도(C): {min_temperature:.2f}")

# region cluster with false color
false_color_image = cv2.applyColorMap(ir_image, cv2.COLORMAP_PLASMA)
false_color_image = cv2.cvtColor(false_color_image, cv2.COLOR_BGR2RGB)
visualization.show_image(false_color_image, "False color image")

# kmeans cluster for false_color_image
kmeans = cv2.kmeans(false_color_image.astype(np.float32).reshape(-1, 3),
                    K=2,
                    bestLabels=None,
                    criteria=None,
                    attempts=10,
                    flags=cv2.KMEANS_RANDOM_CENTERS)

new_predict = kmeans[1].reshape(false_color_image.shape[:2]).astype(np.bool8)
visualization.show_image(new_predict, "predicted leaf region from false color image")
print(f"정확도: {util.get_intersection_over_union(label, new_predict)*100:.2f}")
# endregion
