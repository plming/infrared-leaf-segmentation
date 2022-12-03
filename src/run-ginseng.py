import cv2
import numpy as np

import visualization
import util
from jenks_model import JenksModel

jenks = JenksModel()


def preprocess(image):
    image = cv2.medianBlur(image, 5)

    # dilate and erode the image
    kernel = np.ones((3, 3), np.uint8)

    MORPH_COUNT = 3
    image = cv2.erode(image, kernel, iterations=MORPH_COUNT)
    image = cv2.dilate(image, kernel, iterations=MORPH_COUNT)
    return image


# region load images
ir_image = util.load_ir_in_dat("./ginseng-ir/irimage_20220512_0800.dat")
rgb_image = util.load_rgb_in_jpg("./ginseng-rgb/camimage_20220512_0800.jpg")
# endregion

# region zoom in and crop rgb_image to fit ir_image
zoom_ratio = 1.3  # 1.5
rgb_image = cv2.resize(rgb_image,
                       dsize=None,
                       fx=zoom_ratio,
                       fy=zoom_ratio,
                       interpolation=cv2.INTER_CUBIC)

rgb_image = util.crop_img_from_center(rgb_image,
                                      offset_x=2,
                                      offset_y=7,
                                      cropx=ir_image.shape[1],
                                      cropy=ir_image.shape[0])
# endregion

visualization.show_image(ir_image, "before preprocessing")

preprocessed = preprocess(ir_image.copy())

# region create label(ground truth) image
exg = util.get_excess_green(rgb_image)
label = jenks.predict(exg)
# endregion

predict = jenks.predict(ir_image)
heuristic = jenks.predict(preprocessed)
visualization.show_image(heuristic, "heuristic")

predict = np.bitwise_and(predict, heuristic).astype(np.bool8)
visualization.show_image(predict, "predict")

# region show metrics
accuracy = util.get_intersection_over_union(label, predict)
print(f"정확도(IOU, %): {accuracy*100:.2f}")
# TODO: not works because pixel's range in byte
print(f"잎의 최고 온도(C): {util.get_max_temperature(ir_image, predict):.2f}")
print(f"잎의 평균 온도(C): {util.get_average_temperature(ir_image, predict):.2f}")
print(f"잎의 최저 온도(C): {util.get_min_temperature(ir_image, predict):.2f}")
# endregion

# region show images
visualization.show_image(predict, "predict")
visualization.show_image(label, "label")
