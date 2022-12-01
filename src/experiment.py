import cv2
import numpy as np
import util
import visualization

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

visualization.show_image(ir_image, "before noise reduction")

# reduce noise in ir_image
ir_image = cv2.medianBlur(ir_image, 3)

visualization.show_image(ir_image, "after noise reduction")
